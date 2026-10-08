import numpy as np
from scipy.ndimage import distance_transform_edt, zoom

def _laplacian(a):
    gy,gx=np.gradient(a.astype(np.float32)); return np.gradient(gx,axis=1)+np.gradient(gy,axis=0)

def plausibility(water_mask, img):
    wm=water_mask.astype(bool)
    D=distance_transform_edt(wm); W=D/(D.max()+1e-8) if D.max()>0 else D
    h,w=img.shape; scales=[]
    for s in (1,2,4):
        small=img[::s,::s]
        v=np.abs(_laplacian(small));
        v=zoom(v,(h/v.shape[0],w/v.shape[1]),order=1)
        scales.append(v)
    L=np.minimum.reduce(scales); T=1-L/(L.max()+1e-8)
    return W*T

def apply_constraint(raw_prob, rho, water_mask, img, theta_max=.60, theta_min=.20):
    phi=plausibility(water_mask,img); gamma=float(1-rho); theta=theta_max-(theta_max-theta_min)*gamma
    B=raw_prob>=.50; return (B&(phi>=theta)).astype(np.uint8),phi,theta
