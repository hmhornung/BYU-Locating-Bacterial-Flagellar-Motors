import numpy as np
import PIL.Image as Image
import os

def load_tomo(path):
    filenames = os.listdir(path)
    slices = []
    for file in filenames: 
        slices.append(Image.open(os.path.join(path,file)))
    return np.stack(slices, dtype=np.float16)