# eval_kinova_xhand.py

import sys
sys.path.insert(0, '/home/hshadow/wild_human')
sys.path.insert(0, '/usr/local/lib/python3.10/dist-packages') 



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
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


OmegaConf.register_new_resolver("eval", eval, replace=True)


@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri', required=True, help="NUC IP address (e.g., 192.168.1.15)")
@click.option('--frequency', '-f', default=30, type=float, help="Control frequency in Hz")
@click.option('--max_duration', '-md', default=30, type=float, help='Max duration per episode in seconds')
@click.option('--num_episodes', '-n', default=1, type=int, help='Number of episodes to collect')
def main(input, output, robot_ip, frequency, max_duration, num_episodes):
    
    # ===================== Load Policy =====================
    print(f"Loading checkpoint from {input}")
    payload = torch.load(open(input, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    # Create workspace and load weights
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # Get policy
    policy: BaseImagePolicy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device('cuda')
    policy.eval().to(device)
    
    # Set inference parameters
    policy.num_inference_steps = 16  # DDIM steps
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    
    # Get configuration
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    # n_obs_steps = cfg.n_obs_steps
    n_obs_steps = int(policy.n_obs_steps)
    action_dim = cfg.task.shape_meta.action.shape[0]
    
    print(f"\n{'='*50}")
    print(f"Policy Configuration:")
    print(f"  Action dimension: {action_dim}")
    print(f"  Observation steps: {n_obs_steps}")
    print(f"  Action steps per inference: {policy.n_action_steps}")
    print(f"  Control frequency: {frequency} Hz")
    print(f"{'='*50}\n")
    
    assert action_dim == 19, f"Expected 19D bimanual actions, got {action_dim}D"
    
    
    # ===================== Setup Environment =====================         
    dt = 1.0 / frequency

    serials = [d.serial_number for d in sl.Camera.get_device_list()]
    print("Detected ZED serials:", serials)
    assert len(serials) > 0, "No ZED cameras detected by SDK."
    
    with SharedMemoryManager() as shm_manager:

        with RealXhandEnv(
            output_dir=output,
            robot_ip=robot_ip,
            frequency=frequency,
            n_obs_steps=n_obs_steps,
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
            
            # ===================== Warm up Policy =====================
            # print("Warming up policy inference...")
            # obs = env.get_obs()
            # with torch.no_grad():
            #     policy.reset()
            #     obs_dict_np = get_real_obs_dict(
            #         env_obs=obs, shape_meta=cfg.task.shape_meta)
            #     obs_dict = dict_apply(obs_dict_np,
            #         lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
            #     result = policy.predict_action(obs_dict)
            #     action = result['action'][0].detach().to('cpu').numpy()
            #     print(f"Policy output shape: {action.shape}")
            #     assert action.shape[-1] == 38, f"Expected 38D actions, got {action.shape[-1]}D"
            #     del result
            
            print("\n✓ Ready to collect episodes!")
            print("Press 'C' to start, 'S' to stop episode, 'Q' to quit\n")
            
            # ===================== Collection Loop =====================
            episodes_collected = 0
            
            while episodes_collected < num_episodes:
                # ============ Wait for Start Command ============
                print(f"\n[Episode {episodes_collected + 1}/{num_episodes}] Waiting for start command...")
                
                while True:
                    obs = env.get_obs()
                    vis_img = obs['agentview_image'][-1]
                    
                    # Add status text
                    text = f'Ready - Press C to start | Episode {episodes_collected}/{num_episodes}'
                    cv2.putText(vis_img, text, (10, 30),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.6, thickness=2, color=(0, 255, 0))
                    
                    cv2.imshow('Policy Control', vis_img[..., ::-1])
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
                        t_cycle_start = time.monotonic()
                        t_cycle_end = t_start + (iter_idx + policy.n_action_steps) * dt
                        
                        # -------- Get Observations --------
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        needed = ['agentview_image', 'robot2_eef_pos', 'robot2_eef_quat', 'robot2_xhand_qpos', 'timestamp']
                        obs = {k: v for k, v in obs.items() if k in needed or k == 'timestamp'}
                        
                        # -------- Run Policy Inference --------
                        with torch.no_grad():
                            inference_start = time.time()
                            
                            obs_dict_np = get_real_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np,
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            
                            print(obs_dict)
                            
                            result = policy.predict_action(obs_dict)
                            actions = result['action'][0].detach().to('cpu').numpy()
                            
                            inference_time = time.time() - inference_start
                        
                        # -------- Prepare Actions --------
                        assert actions.shape[-1] == 19, f"Wrong action dim: {actions.shape[-1]}"
                        
                        # Calculate action timestamps
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
                            print(f"  ⚠ Over time budget!")
                        else:
                            actions = actions[is_future]
                            action_timestamps = action_timestamps[is_future]
                        
                        # Optional: Clip workspace (adjust bounds for your robot!)
                        

                        print("[eval] Predicted actions:")
                        print(f"  Shape: {actions.shape}")
                        print(f"  Arm2 pos: {actions[0, 0:3]}")
                        print(f"  Arm2 quat: {actions[0, 3:7]}")
                        print(f"  Arm2 xhand: {actions[0, 7:19]}")


                        
                        # -------- Execute Actions --------

                        print("[eval] about to call env.exec_actions", flush=True)


                        pos_min = np.array([0.02, 0.6, 0.05], dtype=np.float32)
                        pos_max = np.array([ 0.1, 0.7, 0.2], dtype=np.float32)
                        actions[:, 0:3] = np.clip(actions[:, 0:3], pos_min, pos_max)


                        # fixed robot1
                        robot1_home = np.array([
                            -0.02, -0.65, 0.18,
                            0.7071, 0.7071, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                        ], dtype=np.float32)

                        robot1_block = np.tile(robot1_home, (actions.shape[0], 1))
                        actions_full = np.concatenate([robot1_block, actions], axis=1)

                        env.exec_actions(actions=actions_full, timestamps=action_timestamps)



                        
                        # -------- Visualization --------
                        vis_img = obs['agentview_image'][-1]
                        elapsed_time = time.monotonic() - t_start
                        
                        # Status text
                        text = f'Episode {episodes_collected + 1}/{num_episodes} | Time: {elapsed_time:.1f}s | FPS: {1/inference_time:.1f}'
                        cv2.putText(vis_img, text, (10, 30),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.6, thickness=2, color=(0, 255, 0))
                        
                        text2 = f'Actions: {len(actions)} | Press S to stop'
                        cv2.putText(vis_img, text2, (10, 60),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, thickness=1, color=(255, 255, 255))
                        
                        cv2.imshow('Policy Control', vis_img[..., ::-1])
                        key = cv2.pollKey()
                        
                        if key == ord('s'):
                            print("Stopped by user")
                            break
                        
                        # -------- Check Termination --------
                        if elapsed_time > max_duration:
                            print(f"Episode terminated: reached max duration ({max_duration}s)")
                            break
                        
                        # You can add custom termination conditions here
                        # Example:
                        # if is_task_complete(obs):
                        #     print("Task completed!")
                        #     break
                        
                        # -------- Timing --------
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += policy.n_action_steps
                        
                        # Print stats periodically
                        if iter_idx % 10 == 0:
                            print(f"  Step {iter_idx} | Inference: {inference_time*1000:.1f}ms | "
                                  f"Actions executed: {len(actions)}")
                    
                    # End episode
                    env.end_episode()
                    episodes_collected += 1
                    print(f"✓ Episode {episodes_collected} saved!\n")
                    
                except KeyboardInterrupt:
                    print("\nInterrupted by user")
                    env.end_episode()
                    break
                
                except Exception as e:
                    print(f"\n✗ Error during episode: {e}")
                    import traceback
                    traceback.print_exc()
                    env.end_episode()
                    
                    import builtins
                    # Ask if user wants to continue
                    print("\nContinue to next episode? (y/n)")
                    if builtins.input().lower() != 'y':
                        break
            
            print(f"\n{'='*50}")
            print(f"Collection complete! Collected {episodes_collected} episodes")
            print(f"Data saved to: {output}")
            print(f"{'='*50}")


if __name__ == '__main__':
    main()