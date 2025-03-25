import numpy as np
import torch
import random
from scipy.ndimage import gaussian_filter
from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    RandFlipd,
    RandSpatialCropd,
    ResizeWithPadOrCropd,
    RandRotated,
    ToDeviced,
    Zoomd
)

aug_params = {
    "patch_size": (100,100,100),
    "final_size":   (100,100,100),
    "flip_prob":  0.5,
    "rot_prob":   0.5,
    "scale_prob": 1.0,
    "rot_range":  np.pi,
    "scale_range": 0.2
}

def get_heatmap(shape, pts, radius):
    volume = np.zeros(shape, dtype=np.float32)
    for pt in pts: 
        pt = tuple(pt)
        volume[pt] = 1.0
    volume = gaussian_filter(
        volume, 
        radius, 
        axes=(0,1,2)
        )
    volume = volume / np.max(volume)
    return volume.astype(np.float16)

def rand_aug(
    sample:dict,
    aug_params=None,
    gpu:bool=True
):
    """
    Augment 3D volume and mask with cropping, normalization, padding, flipping, and rotation.
    The implementation allows for efficient data loading.
    Input should be memory mapped torch tensors on CPU

    Parameters:
        src (torch.tensor): source 3D volume (shape: D x H x W).
        tgt (torch.tensor): target 3D volume (shape: D x H x W).
        aug_params (Dict): 
    Returns:
        transformed source / target dict
    """
    device='cpu'
    if gpu: device='cuda'

    if len(sample['src'].shape) == 3: sample['src'] = sample['src'].unsqueeze(0)
    if len(sample['tgt'].shape) == 3: sample['tgt'] = sample['tgt'].unsqueeze(0)

    # Convert to tensors and move to the specified device

    keys = ["src", "tgt"]
    mode = ['bilinear', 'bilinear']

    scale_range = [random.uniform(1.0-aug_params['scale_range'], 1.0+aug_params['scale_range']) for i in range(3)]

    augment = Compose([
        RandSpatialCropd(
            keys=keys, 
            roi_size=aug_params["patch_size"], 
            random_center=True, 
            random_size=False
        ),
        ToDeviced(
            keys=keys,
            device=device
        ),
        NormalizeIntensityd( # need in inference
            subtrahend=torch.mean(sample['src']),
            divisor=torch.std(sample['src']),
            keys="src"
        ), 
        RandFlipd(
            keys=keys,
            spatial_axis=[0, 1, 2], 
            prob=aug_params["flip_prob"]
        ),
        Zoomd(
            keys=keys,
            zoom=scale_range,
            mode='trilinear',
            keep_size=False,
            padding_mode='zeros'
        ),
        RandRotated(
            keys=keys, 
            range_x=aug_params["rot_range"], 
            prob=aug_params["rot_prob"],  
            keep_size=False,
            padding_mode='zeros',
            mode=mode
        ),
        ResizeWithPadOrCropd(
            keys=keys, 
            spatial_size=aug_params["final_size"], 
            method="symmetric",
            mode="constant"
        )
    ])
    
    return augment(sample)