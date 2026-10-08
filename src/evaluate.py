import argparse,json
from pathlib import Path
import numpy as np,torch,yaml
from .model import AquaSentinel
from .data import PairedSegDataset
from .metrics import pooled_metrics,per_image_overlap

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--config',required=True); p.add_argument('--manifest',required=True); p.add_argument('--data-root',required=True); p.add_argument('--out',required=True); p.add_argument('--thresholds',default='0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70'); a=p.parse_args()
 c=yaml.safe_load(open(a.config)); items=json.loads(Path(a.manifest).read_text()); ds=PairedSegDataset(a.data_root,items,c['image_size']); dev='cuda' if torch.cuda.is_available() else 'cpu'; m=AquaSentinel(use_physics=c['physics'],use_fp_head=c['fp_head']).to(dev); m.load_state_dict(torch.load(a.checkpoint,map_location=dev),strict=True); m.eval()
 ps=[]; ys=[]
 with torch.no_grad():
  for aa,bb,y in torch.utils.data.DataLoader(ds,batch_size=c['batch_size']):
   s,_=m(aa.to(dev),bb.to(dev)); ps.append(torch.sigmoid(s).cpu().numpy()); ys.append(y.numpy())
 p=np.concatenate(ps,0)[:,0]; y=np.concatenate(ys,0)[:,0]; out=[]
 for th in map(float,a.thresholds.split(',')):
  b=(p>=th); mm=pooled_metrics(b,y); d,i=per_image_overlap(b,y); mm.update({'threshold':th,'dice_per_image':d,'iou_per_image':i}); out.append(mm)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
