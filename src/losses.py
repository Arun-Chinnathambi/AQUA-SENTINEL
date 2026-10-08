import torch
import torch.nn.functional as F

def dice_loss(logits, y, eps=1.0):
    p=torch.sigmoid(logits)
    inter=(p*y).sum(dim=(1,2,3))
    den=p.sum(dim=(1,2,3))+y.sum(dim=(1,2,3))
    return (1-(2*inter+eps)/(den+eps)).mean()

def focal_loss(logits,y,alpha=.8,gamma=2.):
    b=F.binary_cross_entropy_with_logits(logits,y,reduction='none')
    p=torch.sigmoid(logits); pt=p*y+(1-p)*(1-y)
    a=alpha*y+(1-alpha)*(1-y)
    return (a*(1-pt).pow(gamma)*b).mean()

def total_loss(seg_logits, fp_logit, y, dice_w=.5,bce_w=.3,focal_w=.2,aux_w=.3,alpha=.8,gamma=2.):
    ld=dice_loss(seg_logits,y); lb=F.binary_cross_entropy_with_logits(seg_logits,y)
    lf=focal_loss(seg_logits,y,alpha,gamma)
    # Auxiliary target: 1 if frame has no spill, 0 otherwise.
    no_spill=(y.sum(dim=(1,2,3))==0).float().view(-1,1)
    lfp=F.binary_cross_entropy_with_logits(fp_logit,no_spill)
    return dice_w*ld+bce_w*lb+focal_w*lf+aux_w*lfp, {'dice':ld.item(),'bce':lb.item(),'focal':lf.item(),'aux':lfp.item()}
