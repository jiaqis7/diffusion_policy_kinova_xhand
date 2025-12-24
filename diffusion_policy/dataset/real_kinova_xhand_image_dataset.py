# real_kinova_image_dataset.py
from typing import Dict, Any, Optional

import copy
import numpy as np
import torch
import zarr
from filelock import FileLock
from threadpoolctl import threadpool_limits

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer as DPLinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_dataset_masks
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats,
)

# Optional resize backstop (should be a no-op if your converter already wrote 256x256)
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    from PIL import Image
    _HAS_CV2 = False

register_codecs()


class RealKinovaXhandImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: Optional[int] = None,
        n_action_steps: Optional[int] = None,
        abs_action: bool = False,
        use_legacy_normalizer: bool = False,
        seed: int = 42,
        val_ratio: float = 0.0,
        load_to_memory: bool = True,
        dataset_mask_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if dataset_mask_kwargs is None:
            dataset_mask_kwargs = {}

        # Load Zarr (ZipStore for .zarr.zip)
        cache_zarr_path = dataset_path
        cache_lock_path = cache_zarr_path + ".lock"
        print("Acquiring lock on dataset.")


        with FileLock(cache_lock_path):
            print("Loading dataset from disk.")
            
            # Check if it's a zip file or a directory
            import os
            if os.path.isdir(cache_zarr_path):
                print(f"Loading from Zarr directory: {cache_zarr_path}")
                if cache_zarr_path.endswith('.zarr.zip'):
                    cache_zarr_path = cache_zarr_path.replace('.zarr.zip', '.zarr')
                
                src_store = zarr.DirectoryStore(cache_zarr_path)
                
                # Check if data is nested under 'data' group and flatten if needed
                temp_root = zarr.open(src_store, mode='r')
                # needs_flattening = 'data' in temp_root and isinstance(temp_root['data'], zarr.Group)
                needs_flattening = not ('data' in temp_root and isinstance(temp_root['data'], zarr.Group))

                if needs_flattening:
                    print("Detected nested 'data' group structure. Flattening to ReplayBuffer format...")
                    flat_store = zarr.MemoryStore()
                    flat_root = zarr.group(store=flat_store, overwrite=True)
                    
                    # Copy meta and fix episode_ends if needed
                    if 'meta' in temp_root:
                        meta_group = flat_root.create_group('meta', overwrite=True)
                        episode_ends = temp_root['meta']['episode_ends'][:]
                        
                        # Check if episode_ends needs fixing (off-by-one)
                        max_data_len = max(temp_root['data'][key].shape[0] for key in temp_root['data'].keys())
                        if episode_ends[-1] != max_data_len:
                            print(f"Fixing episode_ends: {episode_ends[-1]} -> {max_data_len}")
                            episode_ends = episode_ends + 1  # Add 1 to all episodes
                        
                        meta_group.array('episode_ends', episode_ends, dtype=episode_ends.dtype)
                    
                    # Copy data arrays to root level (not under 'data' group)
                    # for key in temp_root['data'].keys():
                    #     print(f"  Copying {key} to flat structure...")
                    #     zarr.copy(temp_root['data'][key], flat_root, name=key, log=None)

                    # Copy data arrays to root level (not under 'data' group)
                    for key in temp_root['data'].keys():
                        print(f"  Copying {key} to flat structure...")
                        zarr.copy(temp_root['data'][key], flat_root, name=key, log=None)
                    
                    print("Creating ReplayBuffer from flattened structure...")
                    replay_buffer = ReplayBuffer(root=flat_root)
                else:
                    # Original flat structure
                    if load_to_memory:
                        print("Loading dataset to memory...")
                        store = zarr.MemoryStore()
                        replay_buffer = ReplayBuffer.copy_from_store(src_store=src_store, store=store)
                    else:
                        replay_buffer = ReplayBuffer.copy_from_store(src_store=src_store, store=src_store)
            else:
                # It's a zip file - handle as ZipStore
                print(f"Loading from Zarr zip: {cache_zarr_path}")
                zip_store = zarr.ZipStore(cache_zarr_path, mode="r")
                
                # Check for nested structure in zip too
                temp_root = zarr.open(zip_store, mode='r')
                needs_flattening = 'data' in temp_root and isinstance(temp_root['data'], zarr.Group)
                
                if needs_flattening:
                    print("Detected nested 'data' group structure in zip. Flattening...")
                    flat_store = zarr.MemoryStore()
                    flat_root = zarr.group(store=flat_store, overwrite=True)
                    
                    # Copy meta and fix episode_ends if needed
                    if 'meta' in temp_root:
                        meta_group = flat_root.create_group('meta', overwrite=True)
                        episode_ends = temp_root['meta']['episode_ends'][:]
                        
                        max_data_len = max(temp_root['data'][key].shape[0] for key in temp_root['data'].keys())
                        if episode_ends[-1] != max_data_len:
                            print(f"Fixing episode_ends: {episode_ends[-1]} -> {max_data_len}")
                            episode_ends = episode_ends + 1
                        
                        meta_group.array('episode_ends', episode_ends, dtype=episode_ends.dtype)
                    
                    # Copy data arrays to root level
                    for key in temp_root['data'].keys():
                        zarr.copy(temp_root['data'][key], flat_root, name=key, log=None)
                    
                    replay_buffer = ReplayBuffer(root=flat_root)
                    zip_store.close()
                else:
                    if load_to_memory:
                        print("Loading dataset to memory...")
                        store = zarr.MemoryStore()
                        replay_buffer = ReplayBuffer.copy_from_store(src_store=zip_store, store=store)
                        zip_store.close()
                    else:
                        replay_buffer = ReplayBuffer.copy_from_store(src_store=zip_store, store=zip_store)
            
            print("Loaded!")

        # Parse keys from shape_meta (these now match Zarr)
        rgb_keys, lowdim_keys = [], []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            typ = attr.get("type", "low_dim")
            if typ == "rgb":
                rgb_keys.append(key)
            else:
                lowdim_keys.append(key)

        # Target image size derived from shape_meta (e.g., [3, 256, 256])
        self.target_img_size = None
        for k in rgb_keys:
            shp = obs_shape_meta[k]["shape"]
            if len(shp) == 3:
                _, h, w = shp
                self.target_img_size = (h, w)
                break

        # Let sampler keep only the first n_obs_steps of obs keys (no mapping needed)
        key_first_k = {}
        if n_obs_steps is not None:
            for k in (rgb_keys + lowdim_keys):
                key_first_k[k] = n_obs_steps

        # Masks
        train_mask, val_mask, holdout_mask = get_dataset_masks(
            dataset_path=dataset_path,
            num_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
            **dataset_mask_kwargs,
        )

        # Sampler
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        # Members
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.holdout_mask = holdout_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer
        self._dataset_path = dataset_path
        self._dataset_mask_kwargs = dataset_mask_kwargs

        # Visualization flags
        self._return_image = False
        self._render_obs_key = None

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        val_set.train_mask = self.val_mask
        return val_set

    def get_holdout_dataset(self):
        holdout_set = copy.copy(self)
        holdout_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.holdout_mask,
        )
        holdout_set.train_mask = self.holdout_mask
        return holdout_set

    def get_normalizer(self, **kwargs) -> DPLinearNormalizer:
        normalizer = DPLinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer["action"])
        if self.abs_action:
            if stat["mean"].shape[-1] > 10:
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer["action"] = this_normalizer

        # low-dim obs (suffix-based routing)
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            if key.endswith("pos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("quat"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("qpos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError(f"unsupported low-dim key: {key}")
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def _resize_thwc(self, thwc: np.ndarray) -> np.ndarray:
        """(T, H, W, C) → resize to target_img_size if needed."""
        if self.target_img_size is None:
            return thwc
        Ht, Wt = self.target_img_size
        H, W = thwc.shape[1], thwc.shape[2]
        if (H, W) == (Ht, Wt):
            return thwc
        if _HAS_CV2:
            out = np.stack([cv2.resize(fr, (Wt, Ht), interpolation=cv2.INTER_LINEAR) for fr in thwc], axis=0)
        else:
            from PIL import Image
            out = np.stack([np.array(Image.fromarray(fr).resize((Wt, Ht), resample=Image.BILINEAR)) for fr in thwc], axis=0)
        return out


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # Take only the first n_obs_steps of observations
        T_slice = slice(self.n_obs_steps) if self.n_obs_steps is not None else slice(None)

        obs_dict = {}

        # RGB: (T, H, W, C) uint8 -> resize (optional) -> (T, C, H, W) float32 in [0,1]
        for key in self.rgb_keys:
            thwc = data[key][T_slice]  # (T, H, W, C)
            thwc = self._resize_thwc(thwc)
            chw = np.moveaxis(thwc, -1, 1).astype(np.float32) / 255.0
            obs_dict[key] = chw
            # free memory unless we want raw imgs for visualization
            if not (self._return_image and key == self._render_obs_key):
                del data[key]

        # Low-dim obs as float32
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        # Actions: optionally trim time to n_action_steps (e.g., 14)
        act = data["action"].astype(np.float32)  # (T, D)
        
        
        if self.n_action_steps is not None:
            act = act[-self.n_action_steps:]      # take last n_action_steps
            
        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(act),
        }

        assert torch_data["action"].ndim == 2, f"action must be 2D, got {tuple(torch_data['action'].shape)}"
        assert torch_data["action"].shape[-1] == 19, f"action dim must be 19, got {torch_data['action'].shape[-1]}"
        if self.n_action_steps is not None:
            assert torch_data["action"].shape[0] == self.n_action_steps, \
                f"action T must be {self.n_action_steps}, got {torch_data['action'].shape[0]}"

        # Optional raw image for visualization
        if self._return_image:
            assert isinstance(self._render_obs_key, str), "render obs key must be a string!"
            torch_data["img"] = data[self._render_obs_key][T_slice].astype(np.uint8)
            del data[self._render_obs_key]

        return torch_data


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat["max"].max(), np.abs(stat["min"]).max())
    scale = np.full_like(stat["max"], fill_value=1 / max_abs)
    offset = np.zeros_like(stat["max"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat,
    )
