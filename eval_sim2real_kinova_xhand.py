# eval_sim2real_kinova_xhand.py
"""
Evaluation script for deploying simulation-trained diffusion policy to real Kinova+XHand robot.

Both sim and real use 19D actions (7 arm joints + 12 xhand joints).
The real environment automatically pads robot1 with home position internally.

This script handles observation translation:
- Sim policy expects: head_cam [3,256,256], agent_pos [19]
- Real robot provides: agentview_image, robot2_joint_qpos [7] + robot2_xhand_qpos [12]

Usage:
    python eval_sim2real_kinova_xhand.py \
        --input /path/to/sim_checkpoint.ckpt \
        --output /path/to/output_dir \
        --robot_ip 192.168.1.15 \
        --frequency 30 \
        --max_duration 30 \
        --num_episodes 5
"""

import sys
sys.path.insert(0, '/home/hshadow/wild_human')
sys.path.insert(0, '/usr/local/lib/python3.10/dist-packages')
# Add DDDC Benchmark path for loading sim-trained checkpoints (dp.runner.dp_runner.DPRunner)
sys.path.insert(0, '/home/hshadow/diffusion_policy_kinova_xhand/Dynamic-Dexterous-Digital-Cousin-Benchmark/roboverse_learn/il')

import multiprocessing as mp
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import pyzed.sl as sl
from omegaconf import OmegaConf

from diffusion_policy.real_world.real_env_xhand import RealXhandEnv
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import get_real_obs_dict
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.real_world.sim2real_adapter import (
    Sim2RealObservationAdapter,
    get_sim2real_shape_meta
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


@click.command()
@click.option('--input', '-i', required=True, help='Path to sim policy checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri', required=True, help="NUC IP address (e.g., 192.168.1.15)")
@click.option('--frequency', '-f', default=30, type=float, help="Control frequency in Hz")
@click.option('--max_duration', '-md', default=30, type=float, help='Max duration per episode in seconds')
@click.option('--num_episodes', '-n', default=1, type=int, help='Number of episodes to collect')
@click.option('--n_obs_steps', '-nos', default=3, type=int, help='Number of observation steps (should match sim training)')



def main(input, output, robot_ip, frequency, max_duration, num_episodes, n_obs_steps):

    # ===================== Load Sim Policy =====================
    print(f"Loading sim policy checkpoint from {input}")
    payload = torch.load(open(input, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    # DDDC Benchmark checkpoint structure:
    # - model_config: contains policy configuration
    # - state_dicts: contains 'model', 'ema_model', 'optimizer'
    model_cfg = cfg.model_config

    # Print shape info from checkpoint
    print(f"\n=== Checkpoint Shape Meta ===")
    print(f"  Image (head_cam): {model_cfg.shape_meta.obs.head_cam.shape}")
    print(f"  Proprio (agent_pos): {model_cfg.shape_meta.obs.agent_pos.shape}")
    print(f"  Action: {model_cfg.shape_meta.action.shape}")

    # Replace dp.models path with diffusion_policy path for compatibility
    model_cfg_dict = OmegaConf.to_container(model_cfg, resolve=True)
    if model_cfg_dict['_target_'].startswith('dp.models.'):
        # Map DDDC Benchmark class names to diffusion_policy equivalents
        # dp.models.ddpm_unet_image_policy -> diffusion_policy.policy.diffusion_unet_image_policy
        model_cfg_dict['_target_'] = model_cfg_dict['_target_'].replace(
            'dp.models.', 'diffusion_policy.policy.'
        ).replace(
            'ddpm_unet_image_policy', 'diffusion_unet_image_policy'
        )
        print(f"  Remapped _target_: {model_cfg_dict['_target_']}")

    # Instantiate the policy
    policy = hydra.utils.instantiate(model_cfg_dict)

    # Load model weights from state_dicts
    state_dicts = payload['state_dicts']
    use_ema = False
    if hasattr(cfg, 'train_config') and hasattr(cfg.train_config, 'training_params'):
        use_ema = cfg.train_config.training_params.use_ema

    if use_ema and 'ema_model' in state_dicts:
        print("Loading EMA model weights")
        policy.load_state_dict(state_dicts['ema_model'])
    elif 'model' in state_dicts:
        print("Loading model weights")
        policy.load_state_dict(state_dicts['model'])
    else:
        raise KeyError("Checkpoint does not contain 'model' or 'ema_model' state dict")

    device = torch.device('cuda')
    policy.eval().to(device)

    # Set inference parameters
    policy.num_inference_steps = 16  # DDIM steps

    # Try to get n_action_steps from policy or config
    if hasattr(policy, 'n_action_steps'):
        n_action_steps = int(policy.n_action_steps)
    elif hasattr(cfg, 'n_action_steps'):
        n_action_steps = int(cfg.n_action_steps)
    else:
        n_action_steps = 4  # Default

    # Override n_obs_steps from policy if available
    if hasattr(policy, 'n_obs_steps'):
        n_obs_steps = int(policy.n_obs_steps)
    elif hasattr(cfg, 'n_obs_steps'):
        n_obs_steps = int(cfg.n_obs_steps)

    # Use shape_meta from checkpoint for observation transformation
    sim_shape_meta = OmegaConf.to_container(model_cfg.shape_meta, resolve=True)
    # Get image resolution from checkpoint (H, W from shape [C, H, W])
    img_shape = sim_shape_meta['obs']['head_cam']['shape']
    obs_res = (img_shape[1], img_shape[2])  # (H, W)
    action_dim = sim_shape_meta['action']['shape'][0]
    proprio_dim = sim_shape_meta['obs']['agent_pos']['shape'][0]

    print(f"\n{'='*60}")
    print(f"Sim2Real Policy Configuration:")
    print(f"  Checkpoint: {input}")
    print(f"  Action dim: {action_dim}")
    print(f"  Proprio dim: {proprio_dim}")
    print(f"  Observation steps: {n_obs_steps}")
    print(f"  Action steps per inference: {n_action_steps}")
    print(f"  Image resolution: {obs_res} (H, W)")
    print(f"  Control frequency: {frequency} Hz")
    print(f"  Control mode: joint (required for sim2real)")
    print(f"{'='*60}\n")

    # ===================== Setup Observation Adapter =====================
    obs_adapter = Sim2RealObservationAdapter(
        sim_image_key="head_cam",
        sim_proprio_key="agent_pos",
        real_image_key="agentview_image",
        real_arm_key="robot2_joint_qpos",
        real_hand_key="robot2_xhand_qpos"
    )

    print("Observation adapter initialized:")
    print(f"  agentview_image -> head_cam (resize handled explicitly in eval loop)")
    print(f"  robot2_joint_qpos[7] + robot2_xhand_qpos[12] -> agent_pos[{proprio_dim}]")
    if proprio_dim > 19:
        print(f"  NOTE: Will pad {proprio_dim - 19} zeros to match checkpoint's expected dimension")
    print(f"  Action: {action_dim}D from policy -> 19D for robot (truncate if needed)")
    print()

    # ===================== Setup Environment =====================
    dt = 1.0 / frequency

    # Camera captures directly at policy's expected size (no distortion)
    policy_image_size = obs_res  # (H, W) from checkpoint

    serials = [d.serial_number for d in sl.Camera.get_device_list()]
    print("Detected ZED serials:", serials)
    assert len(serials) > 0, "No ZED cameras detected by SDK."

    print(f"Policy expected image size: {policy_image_size} (H, W)")

    with SharedMemoryManager() as shm_manager:
        with RealXhandEnv(
            output_dir=output,
            robot_ip=robot_ip,
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            control_mode='joint',  # CRITICAL: must be joint mode for sim2real
            obs_image_resolution=policy_image_size,  # Capture at policy's expected size
            obs_float32=False,
            enable_multi_cam_vis=True,
            record_raw_video=True,
            thread_per_video=3,
            video_crf=21,
            shm_manager=shm_manager,
            camera_serial_numbers=serials
        ) as env:

            cv2.setNumThreads(1)

            print("Waiting for cameras and robot to be ready...")
            time.sleep(1.0)

            print("\n[OK] Ready for sim2real evaluation!")
            print("Press 'C' to start episode, 'S' to stop episode, 'Q' to quit\n")

            # ===================== Collection Loop =====================
            episodes_collected = 0

            while episodes_collected < num_episodes:
                # ============ Wait for Start Command ============
                print(f"\n[Episode {episodes_collected + 1}/{num_episodes}] Waiting for start command...")

                while True:
                    obs = env.get_obs()
                    vis_img = obs['agentview_image'][-1]

                    # Add status text
                    text = f'Sim2Real Ready - Press C | Episode {episodes_collected}/{num_episodes}'
                    cv2.putText(vis_img, text, (10, 30),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.6, thickness=2, color=(0, 255, 0))

                    cv2.imshow('Sim2Real Control', vis_img[..., ::-1])
                    key = cv2.waitKey(10)

                    if key == ord('q'):
                        print("Exiting...")
                        return
                    elif key == ord('c'):
                        break

                # ============ Policy Control Loop ============
                try:
                    print("Starting episode...")
                    policy.reset()

                    # Start episode with delay
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    # Wait to sync with episode start
                    frame_latency = 1.0 / 30.0  # Camera frame latency
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)

                    print("Policy in control!")
                    iter_idx = 0

                    while True:
                        t_cycle_end = t_start + (iter_idx + n_action_steps) * dt

                        # -------- Get Real Observations --------
                        real_obs = env.get_obs()
                        obs_timestamps = real_obs['timestamp']

                        # Log real_obs structure (first iteration only)
                        if iter_idx == 0:
                            print("\n=== real_obs structure ===")
                            for key, val in real_obs.items():
                                if isinstance(val, np.ndarray):
                                    print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
                                else:
                                    print(f"  {key}: {type(val)}")
                            print()

                        # -------- Adapt Observations for Sim Policy --------
                        # Note: ZedWorker now outputs at policy_image_size directly, no resize needed
                        sim_obs = obs_adapter.adapt_observation(real_obs)

                        # Log sim_obs structure after adaptation (first iteration only)
                        if iter_idx == 0:
                            print("=== sim_obs structure (after adapter) ===")
                            for key, val in sim_obs.items():
                                if isinstance(val, np.ndarray):
                                    print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
                                else:
                                    print(f"  {key}: {type(val)}")
                            print()

                            # Print image size comparison
                            print("=== Image Size Check ===")
                            real_img_shape = sim_obs['head_cam'].shape  # [T, H, W, C]
                            print(f"  Checkpoint expects: [3, {policy_image_size[0]}, {policy_image_size[1]}] (C, H, W)")
                            print(f"  Real robot sends:   {real_img_shape} (T, H, W, C)")
                            if real_img_shape[1] == policy_image_size[0] and real_img_shape[2] == policy_image_size[1]:
                                print(f"  [OK] Image sizes match!")
                            else:
                                print(f"  [WARN] Image size MISMATCH! Policy expects {policy_image_size}, got ({real_img_shape[1]}, {real_img_shape[2]})")
                            print()

                        # Handle dimension mismatch between real (19D) and sim (20D)
                        # Sim has right_hand_ee_joint at index 7 (dummy joint always 0)
                        # Real robot: [arm(7), hand(12)] = 19D
                        # Sim policy: [arm(7), ee_dummy(1), hand(12)] = 20D
                        real_proprio_dim = sim_obs['agent_pos'].shape[-1]

                        # Log proprio dimensions (first iteration only)
                        if iter_idx == 0:
                            print("=== Proprio Dimension Check ===")
                            print(f"  proprio_dim (from checkpoint): {proprio_dim}")
                            print(f"  real_proprio_dim (from robot): {real_proprio_dim}")
                            print(f"  agent_pos shape before transform: {sim_obs['agent_pos'].shape}")
                            print(f"  agent_pos values[0]: {sim_obs['agent_pos'][0]}")

                        # Insert ee_dummy (0.0) at index 7 if checkpoint expects 20D
                        if real_proprio_dim == 19 and proprio_dim == 20:
                            T = sim_obs['agent_pos'].shape[0]
                            agent_pos = sim_obs['agent_pos']
                            # Split: arm[0:7], hand[7:19]
                            arm_joints = agent_pos[:, :7].copy()  # [T, 7]
                            hand_joints_real = agent_pos[:, 7:19]  # [T, 12] in real order

                            # Reorder hand joints from real to sim (alphabetically sorted)
                            # Real order:  [Thumb_rot, Thumb_MCP, Thumb_PIP, Index_abd, Index_MCP, Index_PIP,
                            #               Middle_MCP, Middle_PIP, Ring_MCP, Ring_PIP, Pinky_MCP, Pinky_PIP]
                            # Sim order:   [index_bend, index_j1, index_j2, mid_j1, mid_j2, pinky_j1, pinky_j2,
                            #               ring_j1, ring_j2, thumb_bend, thumb_rota1, thumb_rota2]
                            REAL_TO_SIM_HAND = [3, 4, 5, 6, 7, 10, 11, 8, 9, 1, 0, 2]
                            hand_joints = hand_joints_real[:, REAL_TO_SIM_HAND]  # [T, 12] in sim order

                            # Adjust arm joint angles for sim-to-real offset
                            # Real robot has different zero positions than simulation
                            arm_joints[:, 0] -= 2 * np.pi  # joint[0]: subtract 2π
                            arm_joints[:, 3] -= 2 * np.pi  # joint[3]: subtract 2π
                            arm_joints[:, 4] -= 2 * np.pi  # joint[4]: subtract 2π
                            arm_joints[:, 6] -= np.pi      # joint[-1]: subtract π

                            ee_dummy = np.zeros((T, 1), dtype=agent_pos.dtype)  # [T, 1]
                            # Concatenate: [arm(7), ee_dummy(1), hand(12)] = 20D
                            sim_obs['agent_pos'] = np.concatenate(
                                [arm_joints, ee_dummy, hand_joints], axis=-1
                            )

                            # print("arm joints************************")
                            # print(arm_joints)
                            # print("hand joints************************")
                            # print(hand_joints)


                            if iter_idx == 0:
                                print(f"  [EE_DUMMY] Inserted 0.0 at index 7 for right_hand_ee_joint")
                                print(f"  [EE_DUMMY] agent_pos shape after: {sim_obs['agent_pos'].shape}")

                        if iter_idx == 0:
                            print()

                        # -------- Run Policy Inference --------
                        with torch.no_grad():
                            inference_start = time.time()

                            obs_dict_np = get_real_obs_dict(
                                env_obs=sim_obs,
                                shape_meta=sim_shape_meta
                            )
                            obs_dict = dict_apply(
                                obs_dict_np,
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device)
                            )

                            # Log obs_dict structure right before policy inference
                            if iter_idx == 0:
                                print("=== obs_dict structure (fed into policy) ===")
                                for key, val in obs_dict.items():
                                    if isinstance(val, torch.Tensor):
                                        print(f"  {key}: shape={list(val.shape)}, dtype={val.dtype}, device={val.device}")
                                    elif isinstance(val, dict):
                                        print(f"  {key}: (dict)")
                                        for k2, v2 in val.items():
                                            if isinstance(v2, torch.Tensor):
                                                print(f"    {k2}: shape={list(v2.shape)}, dtype={v2.dtype}")
                                            else:
                                                print(f"    {k2}: {type(v2)}")
                                    else:
                                        print(f"  {key}: {type(val)}")
                                print()

                            result = policy.predict_action(obs_dict)
                            sim_actions = result['action'][0].detach().cpu().numpy()

                            inference_time = time.time() - inference_start

                        # -------- Validate Actions --------
                        assert sim_actions.shape[-1] == action_dim, \
                            f"Expected {action_dim}D action, got {sim_actions.shape[-1]}D"

                        # Convert 20D sim actions to 19D real actions
                        # Sim: [arm(7), ee_dummy(1), hand(12)] = 20D
                        # Real: [arm(7), hand(12)] = 19D
                        # Remove index 7 (ee_dummy joint) and reorder hand joints
                        if action_dim == 20:
                            # Extract arm[0:7] and hand[8:20], skip ee_dummy[7]
                            arm_actions = sim_actions[:, :7]       # [H, 7]
                            hand_actions_sim = sim_actions[:, 8:20]  # [H, 12] in sim order

                            # Reorder hand actions from sim to real order (inverse of REAL_TO_SIM_HAND)
                            # Sim order:  [index_bend, index_j1, index_j2, mid_j1, mid_j2, pinky_j1, pinky_j2,
                            #              ring_j1, ring_j2, thumb_bend, thumb_rota1, thumb_rota2]
                            # Real order: [Thumb_rot, Thumb_MCP, Thumb_PIP, Index_abd, Index_MCP, Index_PIP,
                            #              Middle_MCP, Middle_PIP, Ring_MCP, Ring_PIP, Pinky_MCP, Pinky_PIP]
                            SIM_TO_REAL_HAND = [10, 9, 11, 0, 1, 2, 3, 4, 7, 8, 5, 6]
                            hand_actions = hand_actions_sim[:, SIM_TO_REAL_HAND]  # [H, 12] in real order

                            actions = np.concatenate([arm_actions, hand_actions], axis=-1)  # [H, 19]
                            if iter_idx == 0:
                                print(f"  [EE_DUMMY] Removed index 7 from 20D action -> 19D")
                                print(f"  [HAND_REORDER] Reordered hand actions from sim to real order")
                        else:
                            # For other dimensions, just take first 19D
                            actions = sim_actions[:, :19]

                        # Apply safety limits to actions (now 19D)
                        # Kinova Gen3 joint limits (radians)
                        joint_min = np.array([-3.14, -2.41, -3.14, -2.66, -3.14, -2.23, -3.14])
                        joint_max = np.array([3.14, 2.41, 3.14, 2.66, 3.14, 2.23, 3.14])
                        actions[:, :7] = np.clip(actions[:, :7], joint_min, joint_max)
                        actions[:, 7:19] = np.clip(actions[:, 7:19], 0.0, 1.0)  # xhand

                        # -------- Calculate Action Timestamps --------
                        action_timestamps = (
                            np.arange(len(actions), dtype=np.float64) * dt
                            + obs_timestamps[-1]
                        )

                        # Filter actions that are in the future
                        curr_time = time.time()
                        action_exec_latency = 0.01
                        is_future = action_timestamps > (curr_time + action_exec_latency)

                        if not np.any(is_future):
                            # All actions are in the past - use last action
                            actions = actions[[-1]]
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamps = np.array([eval_t_start + next_step_idx * dt])
                            print(f"  [WARN] Over time budget!")
                        else:
                            actions = actions[is_future]
                            action_timestamps = action_timestamps[is_future]

                        # -------- Execute Actions (19D, env auto-pads robot1) --------
                        print(f"[sim2real] Step {iter_idx}: Sending {len(actions)} actions (19D)")
                        print(f"  Arm joints (rad): {actions[0, :7]}")
                        print(f"  Hand joints: {actions[0, 7:19]}")

                        # env.exec_actions(actions=actions, timestamps=action_timestamps)  # DRY RUN: commented out

                        # -------- Visualization --------
                        vis_img = real_obs['agentview_image'][-1]
                        elapsed_time = time.monotonic() - t_start

                        # Status text
                        text = f'Ep {episodes_collected + 1}/{num_episodes} | Time: {elapsed_time:.1f}s | FPS: {1/inference_time:.1f}'
                        cv2.putText(vis_img, text, (10, 30),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.6, thickness=2, color=(0, 255, 0))

                        text2 = f'Actions: {len(actions)} | Press S to stop'
                        cv2.putText(vis_img, text2, (10, 60),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, thickness=1, color=(255, 255, 255))

                        cv2.imshow('Sim2Real Control', vis_img[..., ::-1])
                        key = cv2.pollKey()

                        if key == ord('s'):
                            print("Stopped by user")
                            break

                        # -------- Check Termination --------
                        if elapsed_time > max_duration:
                            print(f"Episode terminated: reached max duration ({max_duration}s)")
                            break

                        # -------- Timing --------
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += n_action_steps

                        # Print stats periodically
                        if iter_idx % 10 == 0:
                            print(f"  Step {iter_idx} | Inference: {inference_time*1000:.1f}ms | "
                                  f"Actions executed: {len(actions)}")

                    # End episode
                    env.end_episode()
                    episodes_collected += 1
                    print(f"[OK] Episode {episodes_collected} saved!\n")

                except KeyboardInterrupt:
                    print("\nInterrupted by user")
                    env.end_episode()
                    break

                except Exception as e:
                    print(f"\n[ERROR] Error during episode: {e}")
                    import traceback
                    traceback.print_exc()
                    env.end_episode()

                    import builtins
                    # Ask if user wants to continue
                    print("\nContinue to next episode? (y/n)")
                    if builtins.input().lower() != 'y':
                        break

            print(f"\n{'='*60}")
            print(f"Sim2Real evaluation complete!")
            print(f"Collected {episodes_collected} episodes")
            print(f"Data saved to: {output}")
            print(f"{'='*60}")


if __name__ == '__main__':
    main()
