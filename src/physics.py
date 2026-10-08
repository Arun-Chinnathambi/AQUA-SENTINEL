import numpy as np
import torch
import torch.nn.functional as F

def physics_features(x: torch.Tensor) -> torch.Tensor:
    # x: BxCxHxW; use first channel as grayscale/backscatter intensity.
    g = x[:, :1].float()
    # Per-scene min/max normalization to [0,255] for Eq. (1) implementation.
    mn = g.amin(dim=(2,3), keepdim=True)
    mx = g.amax(dim=(2,3), keepdim=True)
    I = (g-mn)/(mx-mn+1e-6)*255.0
    c = 255.0/torch.log1p(I.amax(dim=(2,3), keepdim=True).clamp_min(1.0))
    p1 = c*torch.log1p(I)
    k = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]],device=x.device,dtype=x.dtype).view(1,1,3,3)
    lap = F.conv2d(I,k,padding=1)
    p2 = lap.abs()/(lap.abs().amax(dim=(2,3),keepdim=True)+1e-6)
    return torch.cat([p1/255.0,p2],dim=1)
