import torch
from torch.utils.data import Dataset
import numpy as np
import augment
import gc
import os

class MMapDataset(Dataset):
    def __init__(self, names, path='../data/hm30rad/', aug_params=None, gpu=False):
        src_dir=os.path.join(path,'src')
        tgt_dir=os.path.join(path,'tgt')
        self.src_files = [os.path.join(src_dir,f"{f}.npy") for f in names]
        self.tgt_files = [os.path.join(tgt_dir,f"{f}.npy") for f in names]
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
            del sample[key]
            gc.collect()
        return aug
    
# Custom collate function
def custom_collate(batch):
    src = torch.stack([sample['src'] for sample in batch])
    tgt = torch.stack([sample['tgt'] for sample in batch])
    return {'src': src, 'tgt': tgt}