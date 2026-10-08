import torch
from src.model import AquaSentinel

def test_forward_shape():
    m=AquaSentinel(); a=torch.randn(2,1,256,256); b=torch.randn(2,1,256,256); s,z=m(a,b)
    assert s.shape==(2,1,256,256); assert z.shape==(2,1)
