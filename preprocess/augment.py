import numpy as np

from scipy.ndimage import gaussian_filter
from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    RandFlipd,
    RandAffined,
    RandSpatialCropd,
    SpatialPadd,
    ResizeWithPadOrCropd,
    RandRotated,
    SqueezeDimd
)

def get_heatmap(shape, pts, radius):
    volume = np.zeros(shape, dtype=np.float32)
    for pt in pts: 
        pt = tuple(pt)
        volume[pt] = 1.0
    volume = gaussian_filter(volume, radius, axes=(0,1,2))
    volume = volume / np.max(volume)
    return volume.astype(np.float16)

