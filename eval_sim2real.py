# eval_sim2real.py
# Deploy a simulation-trained DDDC diffusion policy on real Kinova Gen3 + XHand

import sys
import os

# Add DDDC paths so we can import the policy model classes without metasim
DDDC_ROOT = os.path.join(os.path.dirname(__file__), "Dynamic-Dexterous-Digital-Cousin-Benchmark")
sys.path.insert(0, os.path.join(DDDC_ROOT, "roboverse_learn", "il"))  # for 'dp' package
sys.path.insert(0, DDDC_ROOT)  # for roboverse_learn

import multiprocessing as mp
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

import copy
import time
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


# ===================== Module 1: DDDC Checkpoint Loader =====================

def load_dddc_policy(ckpt_path, device='cuda'):
    """Load a DDDC-trained policy directly from checkpoint, bypassing DPRunner."""
    print(f"Loading DDDC checkpoint from {ckpt_path}")
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    # Create model from config (no metasim dependency)
    model = hydra.utils.instantiate(cfg.model_config)

    # Create EMA model
    ema_model = copy.deepcopy(model)

    # Load weights (includes normalizer)
    model.load_state_dict(payload['state_dicts']['model'])
    if 'ema_model' in payload['state_dicts']:
        ema_model.load_state_dict(payload['state_dicts']['ema_model'])

    # Use EMA model if available (better for inference)
    use_ema = cfg.train_config.training_params.get('use_ema', True)
    policy = ema_model if (use_ema and 'ema_model' in payload['state_dicts']) else model

    policy.eval().to(device)

    print(f"  Model type: {type(policy).__name__}")
    print(f"  n_obs_steps: {cfg.n_obs_steps}")
    print(f"  n_action_steps: {cfg.n_action_steps}")
    print(f"  horizon: {cfg.horizon}")
    print(f"  obs agent_pos shape: {cfg.shape_meta.obs.agent_pos.shape}")
    print(f"  action shape: {cfg.shape_meta.action.shape}")

    return policy, cfg


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
    """
    # head_cam: resize + normalize + THWC -> TCHW
    imgs = env_obs['agentview_image']  # (T, H, W, 3) uint8
    w, h = target_res
    resized = np.stack([cv2.resize(img, (w, h)) for img in imgs])
    head_cam = resized.astype(np.float32) / 255.0
    head_cam = np.moveaxis(head_cam, -1, 1)  # (T, 3, H, W)

    # agent_pos: joint(7) + pad(1) + xhand(12) = 20
    joint_qpos = env_obs['robot2_joint_qpos']   # (T, 7)
    xhand_qpos = env_obs['robot2_xhand_qpos']   # (T, 12)
    T = joint_qpos.shape[0]
    pad = np.zeros((T, 1), dtype=np.float32)
    agent_pos = np.concatenate([joint_qpos, pad, xhand_qpos], axis=-1)  # (T, 20)

    return {'head_cam': head_cam, 'agent_pos': agent_pos}


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

    Remove index 7 (xhand_ee placeholder).
    """
    if isinstance(action_20d, torch.Tensor):
        action_20d = action_20d.detach().cpu().numpy()

    arm = action_20d[:, 0:7]    # (n_steps, 7)
    xhand = action_20d[:, 8:20]  # (n_steps, 12)
    action_19d = np.concatenate([arm, xhand], axis=-1)  # (n_steps, 19)

    # Safety clipping
    # Kinova Gen3 joint limits (radians)
    q_min = np.array([-3.14, -2.41, -3.14, -2.66, -3.14, -2.23, -3.14], dtype=np.float32)
    q_max = np.array([3.14, 2.41, 3.14, 2.66, 3.14, 2.23, 3.14], dtype=np.float32)
    action_19d[:, 0:7] = np.clip(action_19d[:, 0:7], q_min, q_max)

    # XHand joint limits [0, 1]
    action_19d[:, 7:19] = np.clip(action_19d[:, 7:19], 0.0, 1.0)

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
    obs_res = (img_width, img_height)

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
                            result = policy.predict_action(stacked_obs)
                            actions_20d = result['action'][0].detach().cpu().numpy()  # (n_action_steps, 20)

                            inference_time = time.time() - inference_start

                        # -------- Adapt Actions --------
                        actions_19d = sim_action_to_real(actions_20d)  # (n_action_steps, 19)

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

                        print(f"[sim2real] actions shape: {actions_full.shape}")
                        print(f"  Arm2 joints: {actions_19d[0, 0:7]}")
                        print(f"  Arm2 xhand:  {actions_19d[0, 7:19]}")

                        env.exec_actions(actions=actions_full, timestamps=action_timestamps)

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
                    print(f"Episode {episodes_collected} saved!\n")

                except KeyboardInterrupt:
                    print("\nInterrupted by user")
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
