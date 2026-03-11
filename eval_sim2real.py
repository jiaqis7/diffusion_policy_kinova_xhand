# eval_sim2real.py
# Deploy a simulation-trained DDDC diffusion policy on real Kinova Gen3 + XHand

import sys
import os

# Add wild_human path for robot controller dependencies
sys.path.insert(0, '/home/hshadow/wild_human')

# Add DDDC paths so we can import the policy model classes without metasim
DDDC_ROOT = os.path.join(os.path.dirname(__file__), "Dynamic-Dexterous-Digital-Cousin-Benchmark")
sys.path.insert(0, os.path.join(DDDC_ROOT, "roboverse_learn", "il"))  # for 'dp' package
sys.path.insert(0, DDDC_ROOT)  # for roboverse_learn

import multiprocessing as mp
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

import time
import pickle
from collections import deque
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pyzed.sl as sl
from omegaconf import OmegaConf

from diffusion_policy.real_world.real_env_xhand import RealXhandEnv
from diffusion_policy.common.precise_sleep import precise_wait

OmegaConf.register_new_resolver("eval", eval, replace=True)


# ===================== Sim Joint Names for v2 Trajectory Format =====================
# These are the 20 joint names in sim order for gen3_xhand_right robot
# Format: 7 arm joints + 1 ee_dummy + 12 hand joints (alphabetically sorted)
# NOTE: IsaacSim uses "Actuator1-7" for arm joints, not "joint_1-7"
SIM_JOINT_NAMES = [
    # Kinova Gen3 arm joints (7) - IsaacSim naming convention
    "Actuator1", "Actuator2", "Actuator3", "Actuator4", "Actuator5", "Actuator6", "Actuator7",
    # XHand ee dummy joint (1)
    "right_hand_ee_joint",
    # XHand finger joints in alphabetical order (12)
    "right_hand_index_bend_joint",
    "right_hand_index_joint1",
    "right_hand_index_joint2",
    "right_hand_mid_joint1",
    "right_hand_mid_joint2",
    "right_hand_pinky_joint1",
    "right_hand_pinky_joint2",
    "right_hand_ring_joint1",
    "right_hand_ring_joint2",
    "right_hand_thumb_bend_joint",
    "right_hand_thumb_rota_joint1",
    "right_hand_thumb_rota_joint2",
]

ROBOT_NAME = "gen3_xhand_right"


def action_array_to_dict(action_20d):
    """Convert 20D action array to dict format for v2 trajectory.

    Args:
        action_20d: numpy array of shape (20,) in sim order

    Returns:
        dict mapping joint names to joint values
    """
    return {name: float(action_20d[i]) for i, name in enumerate(SIM_JOINT_NAMES)}


def build_v2_init_state(first_action_20d):
    """Build v2 format init_state from first action.

    Args:
        first_action_20d: numpy array of shape (20,) - first action in sim order

    Returns:
        dict in v2 format for init_state: {robot_name: {'pos': [...], 'rot': [...], 'dof_pos': {...}}}
    """
    # Use the first action as the initial dof_pos (already in sim order)
    dof_pos_dict = action_array_to_dict(first_action_20d)

    # Default robot pose (will be overridden by task default if needed)
    return {
        ROBOT_NAME: {
            'pos': [0.0, 0.0, 0.0],           # Default position
            'rot': [1.0, 0.0, 0.0, 0.0],      # Default rotation (wxyz quaternion, identity)
            'dof_pos': dof_pos_dict
        }
    }


def build_v2_trajectory_file(trajectory_actions):
    """Build v2 format trajectory file from list of actions.

    Args:
        trajectory_actions: list of {'dof_pos_target': {...}} dicts

    Returns:
        dict in v2 format: {robot_name: [{'init_state': {...}, 'actions': [...]}]}
    """
    if len(trajectory_actions) == 0:
        return None

    # Get first action to build init_state
    first_action = trajectory_actions[0]['dof_pos_target']
    first_action_array = np.array([first_action[name] for name in SIM_JOINT_NAMES])
    v2_init_state = build_v2_init_state(first_action_array)

    # Build the trajectory data structure
    # Format: {robot_name: [demo_0, demo_1, ...]}
    # Each demo: {'init_state': {...}, 'actions': [...]}
    demo_data = {
        'init_state': v2_init_state,
        'actions': trajectory_actions
    }

    return {
        ROBOT_NAME: [demo_data]  # List of demos (we have 1 demo)
    }


# ===================== Module 1: DDDC Checkpoint Loader =====================

def load_dddc_policy(ckpt_path, device='cuda'):
    """Load a DDDC-trained policy directly from checkpoint, bypassing DPRunner."""
    print(f"Loading DDDC checkpoint from {ckpt_path}")
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    # Get model config and remap target path to avoid dp.models import issues
    model_cfg = cfg.model_config
    model_cfg_dict = OmegaConf.to_container(model_cfg, resolve=True)

    # Remap dp.models -> diffusion_policy.policy
    if model_cfg_dict['_target_'].startswith('dp.models.'):
        model_cfg_dict['_target_'] = model_cfg_dict['_target_'].replace(
            'dp.models.', 'diffusion_policy.policy.'
        ).replace(
            'ddpm_unet_image_policy', 'diffusion_unet_image_policy'
        )
        print(f"  Remapped _target_: {model_cfg_dict['_target_']}")

    # Create model from remapped config
    model = hydra.utils.instantiate(model_cfg_dict)

    # Load weights
    state_dicts = payload['state_dicts']
    use_ema = False
    if hasattr(cfg, 'train_config') and hasattr(cfg.train_config, 'training_params'):
        use_ema = cfg.train_config.training_params.get('use_ema', True)

    if use_ema and 'ema_model' in state_dicts:
        print("  Loading EMA model weights")
        model.load_state_dict(state_dicts['ema_model'])
    elif 'model' in state_dicts:
        print("  Loading model weights")
        model.load_state_dict(state_dicts['model'])
    else:
        raise KeyError("Checkpoint does not contain 'model' or 'ema_model' state dict")

    model.eval().to(device)

    print(f"  Model type: {type(model).__name__}")
    print(f"  n_obs_steps: {cfg.n_obs_steps}")
    print(f"  n_action_steps: {cfg.n_action_steps}")
    print(f"  horizon: {cfg.horizon}")
    print(f"  obs agent_pos shape: {cfg.shape_meta.obs.agent_pos.shape}")
    print(f"  action shape: {cfg.shape_meta.action.shape}")

    return model, cfg


# ===================== Module 2: Observation Adapter =====================

def sim2real_obs_adapter(env_obs, target_res=(640, 368)):
    """
    Convert real env observations to sim policy format.

    Real env provides:
      - agentview_image: (T, H, W, 3) uint8
      - robot2_joint_qpos: (T, 7)
      - robot2_xhand_qpos: (T, 12)

    Sim policy expects:
      - head_cam: (T, 3, 368, 640) float32 [0, 1]
      - agent_pos: (T, 20) float32  [joint(7) + xhand_ee_pad(1) + xhand(12)]

    Transformations applied:
      1. Arm joint angle offset correction (real robot has different zero positions)
      2. Hand joint reordering (real order -> sim alphabetical order)
      3. Insert ee_dummy (0.0) at index 7
    """
    # head_cam: resize + normalize + THWC -> TCHW
    imgs = env_obs['agentview_image']  # (T, H, W, 3) uint8
    w, h = target_res
    resized = np.stack([cv2.resize(img, (w, h)) for img in imgs])
    head_cam = resized.astype(np.float32) / 255.0
    head_cam = np.moveaxis(head_cam, -1, 1)  # (T, 3, H, W)

    # agent_pos: joint(7) + ee_dummy(1) + xhand(12) = 20
    arm_joints = env_obs['robot2_joint_qpos'].copy()   # (T, 7)
    hand_joints_real = env_obs['robot2_xhand_qpos']    # (T, 12) in real order
    T = arm_joints.shape[0]

    # 1. Arm joint angle offset correction (real robot has different zero positions than sim)
    arm_joints[:, 0] -= 2 * np.pi  # joint[0]: subtract 2π
    arm_joints[:, 3] -= 2 * np.pi  # joint[3]: subtract 2π
    arm_joints[:, 4] -= 2 * np.pi  # joint[4]: subtract 2π
    arm_joints[:, 6] -= np.pi      # joint[6]: subtract π

    # 2. Hand joint reordering: real order -> sim order (alphabetically sorted)
    # Real order (0-11):  [Thumb_rot, Thumb_MCP, Thumb_PIP, Index_abd, Index_MCP, Index_PIP,
    #                      Middle_MCP, Middle_PIP, Ring_MCP, Ring_PIP, Pinky_MCP, Pinky_PIP]
    # Sim order (0-11):   [index_bend, index_j1, index_j2, mid_j1, mid_j2, pinky_j1, pinky_j2,
    #                      ring_j1, ring_j2, thumb_bend, thumb_rota1, thumb_rota2]
    # Mapping: sim[i] = real[REAL_TO_SIM_HAND[i]]
    #   sim[9] (thumb_bend)  <- real[0] (Thumb_rot)
    #   sim[10] (thumb_rota1) <- real[1] (Thumb_MCP)
    #   sim[11] (thumb_rota2) <- real[2] (Thumb_PIP)
    REAL_TO_SIM_HAND = [3, 4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2]
    hand_joints = hand_joints_real[:, REAL_TO_SIM_HAND]  # (T, 12) in sim order

    # 3. Insert ee_dummy (0.0) at index 7
    ee_dummy = np.zeros((T, 1), dtype=np.float32)

    # Concatenate: [arm(7), ee_dummy(1), hand(12)] = 20D
    agent_pos = np.concatenate([arm_joints, ee_dummy, hand_joints], axis=-1)


    # print(f"&&&&&&&&&&agent_pos: {agent_pos} &&&&&&&&&&")

    return {'head_cam': head_cam, 'agent_pos': agent_pos.astype(np.float32)}


# ===================== Module 3: Temporal Stacking =====================

def stack_obs(obs_history, n_obs_steps, device='cuda'):
    """
    Stack observation history into batched tensors for policy input.

    Each obs in history is a dict with:
      - head_cam: (3, H, W)
      - agent_pos: (20,)

    Returns dict with:
      - head_cam: (1, n_obs_steps, 3, H, W) torch tensor
      - agent_pos: (1, n_obs_steps, 20) torch tensor
    """
    result = {}
    for key in obs_history[0].keys():
        frames = [obs[key] for obs in obs_history]
        n_have = len(frames)

        # Pad with first frame if not enough history
        if n_have < n_obs_steps:
            pad_frames = [frames[0]] * (n_obs_steps - n_have)
            frames = pad_frames + frames

        # Take last n_obs_steps
        frames = frames[-n_obs_steps:]
        stacked = np.stack(frames, axis=0)  # (n_obs_steps, ...)
        result[key] = torch.from_numpy(stacked).unsqueeze(0).to(device).float()

    return result


# ===================== Module 4: Action Adapter =====================

def sim_action_to_real(action_20d):
    """
    Convert 20D sim action to 19D real action.

    Sim: [arm_joints(7), xhand_ee(1), xhand_joints(12)] = 20D
    Real: [arm_joints(7), xhand_joints(12)] = 19D

    Transformations applied:
      1. Remove index 7 (xhand_ee placeholder)
      2. Add back arm joint offsets (inverse of observation adapter)
      3. Reorder hand joints from sim order to real order
    """
    if isinstance(action_20d, torch.Tensor):
        action_20d = action_20d.detach().cpu().numpy()

    arm = action_20d[:, 0:7].copy()    # (n_steps, 7) - copy to avoid modifying original
    xhand_sim = action_20d[:, 8:20]    # (n_steps, 12) in sim order

    # Add back arm joint offsets (inverse of observation adapter)
    # Observation adapter: real → sim (subtract offset)
    # Action adapter: sim → real (add offset back)
    arm[:, 0] += 0  # joint[0]: add 2π
    arm[:, 3] += 0  # joint[3]: add 2π
    arm[:, 4] += 0  # joint[4]: add 2π
    arm[:, 6] += np.pi      # joint[6]: add π

    # Reorder hand joints from sim order to real order (inverse of REAL_TO_SIM_HAND)
    # Sim order (0-11):  [index_bend, index_j1, index_j2, mid_j1, mid_j2, pinky_j1, pinky_j2,
    #                     ring_j1, ring_j2, thumb_bend, thumb_rota1, thumb_rota2]
    # Real order (0-11): [Thumb_rot, Thumb_MCP, Thumb_PIP, Index_abd, Index_MCP, Index_PIP,
    #                     Middle_MCP, Middle_PIP, Ring_MCP, Ring_PIP, Pinky_MCP, Pinky_PIP]
    # Mapping: real[i] = sim[SIM_TO_REAL_HAND[i]]
    #   real[0] (Thumb_rot)  <- sim[9] (thumb_bend)
    #   real[1] (Thumb_MCP)  <- sim[10] (thumb_rota1)
    #   real[2] (Thumb_PIP)  <- sim[11] (thumb_rota2)
    SIM_TO_REAL_HAND = [9, 10, 11, 0, 1, 2, 3, 4, 7, 8, 5, 6]
    xhand_real = xhand_sim[:, SIM_TO_REAL_HAND]  # (n_steps, 12) in real order

    action_19d = np.concatenate([arm, xhand_real], axis=-1)  # (n_steps, 19)

    # Safety clipping
    # Kinova Gen3 joint limits (radians)
    q_min = np.array([-3.14, -2.41, -3.14, -2.66, -3.14, -2.23, -3.14], dtype=np.float32)
    q_max = np.array([3.14, 2.41, 3.14, 2.66, 3.14, 2.23, 3.14], dtype=np.float32)
    action_19d[:, 0:7] = np.clip(action_19d[:, 0:7], q_min, q_max)

    print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@Action adapter output (first step) xhand: {action_19d[0, 7:19]}")

    action_19d[:, 7:19] = np.clip(action_19d[:, 7:19], 0.0, 2.0)

    print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@Action adapter output (after clipping) xhand: {action_19d[0, 7:19]}")

    return action_19d


# ===================== Module 5: Main Eval Loop =====================

@click.command()
@click.option('--ckpt', '-c', required=True, help='Path to DDDC checkpoint (.ckpt)')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri', required=True, help="NUC IP address (e.g., 192.168.1.15)")
@click.option('--frequency', '-f', default=30, type=float, help="Control frequency in Hz")
@click.option('--max_duration', '-md', default=30, type=float, help='Max duration per episode in seconds')
@click.option('--num_episodes', '-n', default=1, type=int, help='Number of episodes to collect')
@click.option('--img_width', default=640, type=int, help='Sim policy image width')
@click.option('--img_height', default=368, type=int, help='Sim policy image height')
def main(ckpt, output, robot_ip, frequency, max_duration, num_episodes, img_width, img_height):

    device = torch.device('cuda')
    target_res = (img_width, img_height)

    # ===================== Load Policy =====================
    policy, cfg = load_dddc_policy(ckpt, device=device)
    n_obs_steps = int(cfg.n_obs_steps)
    n_action_steps = int(cfg.n_action_steps)
    action_dim = int(cfg.shape_meta.action.shape[0])

    print(f"\n{'='*50}")
    print(f"Sim2Real Policy Configuration:")
    print(f"  Sim action dim: {action_dim} -> Real action dim: {action_dim - 1}")
    print(f"  Observation steps: {n_obs_steps}")
    print(f"  Action steps per inference: {n_action_steps}")
    print(f"  Image resolution: {img_width}x{img_height}")
    print(f"  Control frequency: {frequency} Hz")
    print(f"{'='*50}\n")

    # ===================== Setup Environment =====================
    dt = 1.0 / frequency
    obs_res = (img_height, img_width)  # (H, W) format for ZedWorker

    serials = [d.serial_number for d in sl.Camera.get_device_list()]
    print("Detected ZED serials:", serials)
    assert len(serials) > 0, "No ZED cameras detected by SDK."

    with SharedMemoryManager() as shm_manager:
        with RealXhandEnv(
            output_dir=output,
            robot_ip=robot_ip,
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            control_mode='joint',
            obs_image_resolution=obs_res,
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

            print("\nReady! Press 'C' to start, 'S' to stop episode, 'Q' to quit\n")

            # ===================== Collection Loop =====================
            episodes_collected = 0

            while episodes_collected < num_episodes:
                print(f"\n[Episode {episodes_collected + 1}/{num_episodes}] Waiting for start command...")

                # Wait for start
                while True:
                    obs = env.get_obs()
                    vis_img = obs['agentview_image'][-1]
                    text = f'Ready - Press C to start | Episode {episodes_collected}/{num_episodes}'
                    cv2.putText(vis_img, text, (10, 30),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.6, thickness=2, color=(0, 255, 0))
                    cv2.imshow('Sim2Real Policy Control', vis_img[..., ::-1])
                    key = cv2.waitKey(10)
                    if key == ord('q'):
                        print("Exiting...")
                        return
                    elif key == ord('c'):
                        break

                # ============ Policy Control Loop ============
                # Action rollout storage for replay (initialized before try block)
                trajectory_actions = []
                init_state = None

                try:
                    print("Starting episode...")
                    obs_history = deque(maxlen=n_obs_steps)

                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    frame_latency = 1.0 / 30.0
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)

                    print("Policy in control!")
                    iter_idx = 0

                    while True:
                        t_cycle_end = t_start + (iter_idx + n_action_steps) * dt

                        # -------- Get & Adapt Observations --------
                        raw_obs = env.get_obs()
                        obs_timestamps = raw_obs['timestamp']

                        # Capture initial state on first iteration
                        if init_state is None:
                            init_state = {
                                'robot2_joint_qpos': raw_obs['robot2_joint_qpos'][-1].copy(),
                                'robot2_xhand_qpos': raw_obs['robot2_xhand_qpos'][-1].copy(),
                            }
                            print(f"[Trajectory] Captured init_state: robot2_joint_qpos={init_state['robot2_joint_qpos']}")

                        adapted = sim2real_obs_adapter(raw_obs, target_res=target_res)

                        # Take last frame for temporal stacking
                        single_obs = {
                            'head_cam': adapted['head_cam'][-1],    # (3, H, W)
                            'agent_pos': adapted['agent_pos'][-1],  # (20,)
                        }
                        obs_history.append(single_obs)

                        # -------- Stack & Run Policy --------
                        with torch.no_grad():
                            inference_start = time.time()

                            stacked_obs = stack_obs(obs_history, n_obs_steps, device=device)

                            # Log agent_pos right before policy inference (first iteration only)
                            if iter_idx == 0:
                                agent_pos_np = stacked_obs['agent_pos'][0].cpu().numpy()  # (n_obs_steps, 20)
                                print(f"[DEBUG] agent_pos sent to policy (shape: {agent_pos_np.shape}):")
                                for t_idx in range(agent_pos_np.shape[0]):
                                    print(f"  t={t_idx}: {agent_pos_np[t_idx]}")

                            # Save image right before policy inference (first iteration only)
                            if iter_idx == 0:
                                # stacked_obs['head_cam'] is (1, n_obs_steps, 3, H, W), values in [0, 1]
                                img_tensor = stacked_obs['head_cam'][0, -1]  # Take last frame: (3, H, W)
                                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                                img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, 3)
                                img_bgr = img_np[..., ::-1]  # RGB -> BGR for cv2
                                save_path = os.path.join(output, 'policy_input_image.png')
                                cv2.imwrite(save_path, img_bgr)
                                print(f"[DEBUG] Saved policy input image to: {save_path}")
                                print(f"  Image shape: {img_np.shape}, range: [{img_np.min()}, {img_np.max()}]")

                            result = policy.predict_action(stacked_obs)
                            actions_20d = result['action'][0].detach().cpu().numpy()  # (n_action_steps, 20)
                            print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@Policy output (first step) xhand: {actions_20d[0, 8:20]}")

                            inference_time = time.time() - inference_start

                        # -------- Adapt Actions --------
                        actions_19d = sim_action_to_real(actions_20d)  # (n_action_steps, 19)

                        # Store actions for replay in v2 format (dict with joint names)
                        for act_20d in actions_20d:
                            trajectory_actions.append({
                                'dof_pos_target': action_array_to_dict(act_20d)
                            })

                        # -------- Timestamps & Future Filtering --------
                        action_timestamps = (
                            np.arange(len(actions_19d), dtype=np.float64) * dt
                            + obs_timestamps[-1]
                        )

                        curr_time = time.time()
                        action_exec_latency = 0.01
                        is_future = action_timestamps > (curr_time + action_exec_latency)

                        if not np.any(is_future):
                            actions_19d = actions_19d[[-1]]
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamps = np.array([eval_t_start + next_step_idx * dt])
                            print(f"  Warning: over time budget!")
                        else:
                            actions_19d = actions_19d[is_future]
                            action_timestamps = action_timestamps[is_future]

                        # -------- Prepend robot1 home & Execute --------
                        robot1_home = np.array([
                            # 7 joint angles (radians)
                            2.1118, 0.9076, -2.9671, -1.9897, -1.0123, -1.0821, -3.1241,
                            # 12 xhand joints (zeros)
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                        ], dtype=np.float32)

                        robot1_block = np.tile(robot1_home, (len(actions_19d), 1))
                        actions_full = np.concatenate([robot1_block, actions_19d], axis=1)  # (n, 38)

                        print(f"  Real Arm2 joints: {actions_19d[0, 0:7]}")
                        print(f"  Real Arm2 xhand:  {actions_19d[0, 7:19]}")


                        env.exec_actions(actions=actions_full, timestamps=action_timestamps)  # DRY RUN: commented out

                        # -------- Visualization --------
                        vis_img = raw_obs['agentview_image'][-1]
                        elapsed_time = time.monotonic() - t_start

                        text = f'Episode {episodes_collected + 1}/{num_episodes} | Time: {elapsed_time:.1f}s | FPS: {1/inference_time:.1f}'
                        cv2.putText(vis_img, text, (10, 30),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.6, thickness=2, color=(0, 255, 0))
                        text2 = f'Actions: {len(actions_19d)} | Press S to stop'
                        cv2.putText(vis_img, text2, (10, 60),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, thickness=1, color=(255, 255, 255))

                        cv2.imshow('Sim2Real Policy Control', vis_img[..., ::-1])
                        key = cv2.pollKey()

                        if key == ord('s'):
                            print("Stopped by user")
                            break

                        if elapsed_time > max_duration:
                            print(f"Episode terminated: reached max duration ({max_duration}s)")
                            break

                        # -------- Timing --------
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += n_action_steps

                        if iter_idx % 10 == 0:
                            print(f"  Step {iter_idx} | Inference: {inference_time*1000:.1f}ms | "
                                  f"Actions: {len(actions_19d)}")

                    env.end_episode()
                    episodes_collected += 1

                    # Save trajectory as v2 pkl for DDDC replay in simulation
                    if len(trajectory_actions) > 0:
                        # Build v2 format trajectory file
                        trajectory_data = build_v2_trajectory_file(trajectory_actions)

                        # Use _v2.pkl suffix so replay script recognizes it
                        traj_filename = f'trajectory_ep{episodes_collected:03d}_v2.pkl'
                        traj_path = os.path.join(output, traj_filename)
                        with open(traj_path, 'wb') as f:
                            pickle.dump(trajectory_data, f)
                        print(f"[Trajectory] Saved {len(trajectory_actions)} actions (v2 format) to: {traj_path}")

                    print(f"Episode {episodes_collected} saved!\n")

                except KeyboardInterrupt:
                    print("\nInterrupted by user")
                    # Save partial trajectory if any actions collected
                    if len(trajectory_actions) > 0:
                        # Build v2 format trajectory file
                        trajectory_data = build_v2_trajectory_file(trajectory_actions)

                        traj_filename = f'trajectory_ep{episodes_collected + 1:03d}_partial_v2.pkl'
                        traj_path = os.path.join(output, traj_filename)
                        with open(traj_path, 'wb') as f:
                            pickle.dump(trajectory_data, f)
                        print(f"[Trajectory] Saved partial trajectory ({len(trajectory_actions)} actions, v2 format) to: {traj_path}")
                    env.end_episode()
                    break

                except Exception as e:
                    print(f"\nError during episode: {e}")
                    import traceback
                    traceback.print_exc()
                    env.end_episode()

                    import builtins
                    print("\nContinue to next episode? (y/n)")
                    if builtins.input().lower() != 'y':
                        break

            print(f"\n{'='*50}")
            print(f"Collection complete! Collected {episodes_collected} episodes")
            print(f"Data saved to: {output}")
            print(f"{'='*50}")


if __name__ == '__main__':
    main()
