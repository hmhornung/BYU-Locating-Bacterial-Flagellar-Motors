import numpy as np
import pandas as pd
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
    rows = df.loc[df['tomo_id'] == name]
    pts = []
    for i in range(len(rows)):
        pts.append([int(rows.iloc[i][f'Motor axis {j}']) for j in range(3)])
    if -1 in pts[0]: return []
    else: return pts

def save_sample(path, name:str, sample:dict):
    '''
    Save a sample dict of torch tensor src/tgt to disk
    Parameters:
        path
        name (string): name of file (exclude .npy)
        sample: (dict): dict{'src','tgt'} mapping to torch tensor volumes
    '''
    src_dest = os.path.join(path, 'src/', name + '.npy')
    tgt_dest = os.path.join(path, 'tgt/', name + '.npy')
    np.save(src_dest, sample['src'])
    np.save(tgt_dest, sample['tgt'])

def gen_heatmap_dataset(dest:os.path, tomo_src:os.path='../data/train', pt_src='../data/train_labels.csv', radius=30):
    tomos = os.listdir(tomo_src)
    for tomo in tomos:
        src = load_tomo(os.path.join(tomo_src, tomo))
        pts = load_pts(tomo, path=pt_src)
        if not pts:
            tgt = np.zeros(src.shape, dtype=np.float16)
            print(f'{tomo} heatmap empty')
        else: 
            tgt = augment.get_heatmap(
                shape=src.shape, 
                pts=pts, 
                radius=radius
                )
            print(f'{tomo} heatmap genned')

        sample = {'src':src, 'tgt':tgt}
        save_sample(
            path=dest, 
            name=tomo, 
            sample=sample
            )
        print(f'{tomo} saved')
