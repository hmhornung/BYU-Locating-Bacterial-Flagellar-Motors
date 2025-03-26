import numpy as np
import torch
import gc
import random
from scipy.ndimage import gaussian_filter
import SimpleITK as sitk
from monai.transforms import MapTransform
from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    RandFlipd,
    RandSpatialCropd,
    ResizeWithPadOrCropd,
    RandRotated,
    ToDeviced,
    Zoomd,
    SqueezeDimd
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

def tuple_op(operator, tuple1, tuple2):
    if len(tuple1) != len(tuple2):
        raise ValueError("Tuples must have the same length")
    
    # Apply the operator element-wise using zip and a list comprehension
    return tuple(operator(x, y) for x, y in zip(tuple1, tuple2))

class RandCropMMapd(MapTransform):
    def __init__(self, keys, roi_size):
        super().__init__(keys)
        self.roi_size = roi_size

    def __call__(self, data):
        shapes = [data[self.keys[0]][i].shape for i in range(len(data['src']))]
        ranges = [tuple_op(lambda x,y: x-y, shape, self.roi_size) for shape in shapes]
        start  = [tuple(random.randint(0, r[i]) for i in range(3)) for r in ranges]
        stop   = [tuple_op(lambda x,y: x+y, s, self.roi_size) for s in start]
        result = {}
        for key in self.keys:
            crops = [data[key][i][tuple(slice(j, k) for j, k in zip(start[i], stop[i]))].copy() for i in range(len(data[key]))]
            result[key] = np.expand_dims(np.stack(crops), axis=1)
        return result

class ToTorchd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            data[key] = torch.from_numpy(data[key])
        return data

def gaussian_filter_sitk(volume, radius):
    image = sitk.GetImageFromArray(volume)
    smoothed_image = sitk.SmoothingRecursiveGaussian(image, radius)
    return sitk.GetArrayFromImage(smoothed_image)

def get_heatmap(shape, pts, radius): 
    volume = np.zeros(shape, dtype=np.float32)
    for pt in pts: 
        pt = tuple(np.rint(pt).astype(int))
        if all(0 <= pt[i] < shape[i] for i in range(3)):
            volume[pt] = 1.0
    volume = gaussian_filter_sitk(volume, radius)
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
        src (list of mmap .npy): source 3D volumes (shape: D x H x W).
        tgt (list of mmap .npy): target 3D volumes (shape: D x H x W).
        aug_params (Dict): 
    Returns:
        transformed source / target dict
    """
    device='cpu'
    if gpu: device='cuda'

    # Convert to tensors and move to the specified device

    keys = ["src", "tgt"]
    mode = ['bilinear', 'bilinear']

    scale_range = [random.uniform(1.0-aug_params['scale_range'], 1.0+aug_params['scale_range']) for i in range(3)]

    augment1 = Compose([
        RandCropMMapd(
            keys=keys,
            roi_size=aug_params["patch_size"]
        ),
        ToTorchd(
            keys=keys
        ),
        ToDeviced(
            keys=keys,
            device=device
        ),
        NormalizeIntensityd( # need in inference
            keys="src"
        ), 
        RandFlipd(
            keys=keys,
            spatial_axis=[0, 1, 2], 
            prob=aug_params["flip_prob"]
        ),
        SqueezeDimd(
            keys=keys,
            dim=1
        )
        ])
    augment2 = Compose([
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
    data1 = augment1(sample)
    print(data1['src'].shape)
    print(data1['tgt'].shape)
    data2 = augment2(data1)
    # return augment(data2)

#testing

def get_heatmap_old(shape, pts, radius):
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