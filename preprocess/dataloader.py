import torch
from torch.utils.data import Dataset
import numpy as np
import augment
import gc
import os

def release_mmap_array(mmap_array):
    '''
    something that ChatGPT created to release mmap files quicker
    the mmap delete was discovered as a large bottleneck in the pipeline
    '''
    if hasattr(mmap_array, 'base') and mmap_array.base is not None:
        try:
            mmap_array.base.close()  # Close the file descriptor explicitly
        except AttributeError:
            pass  # Some numpy versions may not expose .base.close()
    del mmap_array

class MMapDataset(Dataset):
    def __init__(self, names, path='../data/hm30rad/', aug_params=None, gpu=False, zero_threshold = 0.0):
        src_dir=os.path.join(path,'src')
        tgt_dir=os.path.join(path,'tgt')
        self.src_files = [os.path.join(src_dir,f"{f}.npy") for f in names]
        self.tgt_files = [os.path.join(tgt_dir,f"{f}.npy") for f in names]
        self.zero_threshold = zero_threshold
        if aug_params is None:
            self.aug_params = {
                "patch_size": (96,96,96),
                "final_size":   (96,96,96),
                "flip_prob":  0.5,
                "rot_prob":   0.5,
                "scale_prob": 1.0,
                "rot_range":  np.pi,
                "scale_range": 0.2
            }
        else: self.aug_params = aug_params
        self.gpu = gpu

    def __len__(self):
        return len(self.src_files)

    def __getitem__(self, idx):
        sample = {
            'src': np.load(self.src_files[idx], mmap_mode='r'),
            'tgt': np.load(self.tgt_files[idx], mmap_mode='r'),
        }
        
        aug = augment.rand_aug(
            sample=sample,
            aug_params=self.aug_params,
            gpu=self.gpu
        )
        
        for key in aug.keys():
            release_mmap_array(sample[key])
        if torch.max(aug['tgt']) < self.zero_threshold:
            aug['tgt'] = torch.zeros(aug['src'].shape, dtype=torch.float32)
        return aug
    
# Custom collate function
def custom_collate(batch):
    src = torch.stack([sample['src'] for sample in batch])
    tgt = torch.stack([sample['tgt'] for sample in batch])
    return {'src': src, 'tgt': tgt}

class Scheduler:
    def __init__(self, warmup_epochs = 0, warmup_value = 1.0, init_value = 1.0, factor = 1.0, stop=10):
        self.warmup_epochs = warmup_epochs
        self.warmup = False
        self.value = 1.0
        self.factor = factor
        self.epoch = 0
        self.init_value = init_value
        self.stop=stop
        if self.warmup_epochs > 0:
            self.warmup = True
            self.value = warmup_value
        else:
            self.value = init_value
    def step(self):
        self.epoch += 1
        if self.epoch == self.warmup_epochs:
            self.value = self.init_value
        elif self.warmup and self.epoch > self.warmup_epochs and self.epoch < (self.warmup_epochs + self.stop):
            self.value *= self.factor
            self.stop += 1
        else:
            self.value *= self.factor
            self.stop += 1
    def __call__(self):
        return self.value