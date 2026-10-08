from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

def _read(p):
    p=Path(p)
    if p.suffix.lower()=='.npy': return np.load(p)
    return np.asarray(Image.open(p))

def load_channel(p,size):
    a=_read(p)
    if a.ndim==3: a=a[...,0]
    a=a.astype(np.float32)
    if a.shape!=(size,size): a=np.asarray(Image.fromarray(a).resize((size,size),Image.BILINEAR))
    lo,hi=np.nanpercentile(a,[1,99]); a=np.clip((a-lo)/(hi-lo+1e-6),0,1)
    return a

def load_mask(p,size):
    a=_read(p)
    if a.ndim==3: a=a[...,0]
    if a.shape!=(size,size): a=np.asarray(Image.fromarray(a).resize((size,size),Image.NEAREST))
    return (a>127).astype(np.float32)

class PairedSegDataset(Dataset):
    def __init__(self,root,items,size=256): self.root=Path(root); self.items=items; self.size=size
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        d=self.items[i]
        a=load_channel(self.root/d['a'],self.size); b=load_channel(self.root/d['b'],self.size); y=load_mask(self.root/d['mask'],self.size)
        return torch.from_numpy(a[None]),torch.from_numpy(b[None]),torch.from_numpy(y[None])
