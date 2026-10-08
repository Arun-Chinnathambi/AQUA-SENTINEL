import numpy as np

def pooled_counts(pred,y):
    p=pred.astype(bool); t=y.astype(bool)
    return int((p&t).sum()),int((p&~t).sum()),int((~p&t).sum())

def pooled_metrics(pred,y):
    tp,fp,fn=pooled_counts(pred,y)
    P=tp/(tp+fp+1e-12); R=tp/(tp+fn+1e-12)
    F1=2*P*R/(P+R+1e-12)
    return {'precision':P,'recall':R,'f1':F1,'fdr':1-P,'tp':tp,'fp':fp,'fn':fn}

def per_image_overlap(pred,y):
    ds=[]; ios=[]
    for p,t in zip(pred,y):
        p=p.astype(bool); t=t.astype(bool); inter=(p&t).sum(); union=(p|t).sum()
        if union==0: ds.append(1.0); ios.append(1.0)
        else: ds.append(2*inter/(p.sum()+t.sum()+1e-12)); ios.append(inter/(union+1e-12))
    return float(np.mean(ds)),float(np.mean(ios))
