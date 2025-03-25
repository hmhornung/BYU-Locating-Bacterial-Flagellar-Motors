import numpy as np
import pandas as pd
import torch
import PIL.Image as Image
import os
import augment

def load_tomo(path):
    filenames = os.listdir(path)
    slices = []
    for file in filenames: 
        slices.append(Image.open(os.path.join(path,file)))
    return np.stack(slices, dtype=np.float16)

def load_pts(name, path='../data/train_labels.csv'):
    df = pd.read_csv(path)
    rows = df.loc[df['tomo_id'] == 'name']
    print(rows)
    pts = []
    for i in range(len(rows)):
        pts.append([rows.iloc[i][f'Motor axis {j}'] for j in range(3)])
    return pts

def save_sample(path, name:str, sample:dict):
    '''
    Save a sample dict of torch tensor src/tgt to disk
    Parameters:
        path
        name (string): name of file (exclude .pt)
        sample: (dict): dict{'src','tgt'} mapping to torch tensor volumes
    '''
    src_dest = os.path.join(path, 'src/', name + '.pt')
    tgt_dest = os.path.join(path, 'tgt/', name + '.pt')
    torch.save(sample['src'], src_dest)
    torch.save(sample['tgt'], tgt_dest)

def gen_heatmap_dataset(dest:os.path, tomo_src:os.path='../data/train', pt_src='../data/train_labels.csv', radius=30):
    tomos = os.listdir(tomo_src)
    for tomo in tomos:
        src = load_tomo(os.path.join(tomo_src, tomo))
        pts = load_pts(tomo, path=pt_src)
        tgt = augment.get_heatmap(
            shape=src.shape, 
            pts=pts, 
            radius=radius
            )

        src = torch.Tensor(src).to(torch.float16)
        tgt = torch.Tensor(tgt).to(torch.float16)
        sample = {'src':src, 'tgt':tgt}
        save_sample(
            path=dest, 
            name=tomo, 
            sample=sample
            )
