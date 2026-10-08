import argparse,yaml,random,json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from .model import AquaSentinel
from .losses import total_loss
from .data import PairedSegDataset

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def read_items(path): return json.loads(Path(path).read_text())
def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--manifest',required=True); p.add_argument('--data-root',required=True); p.add_argument('--out',required=True); p.add_argument('--no-physics',action='store_true'); p.add_argument('--no-fp-head',action='store_true'); p.add_argument('--warm-start'); args=p.parse_args(); c=yaml.safe_load(open(args.config)); seed_all(c['seed']); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
 items=read_items(args.manifest); ds=PairedSegDataset(args.data_root,items,c['image_size']); dl=DataLoader(ds,batch_size=c['batch_size'],shuffle=True,num_workers=0)
 dev='cuda' if torch.cuda.is_available() else 'cpu'; m=AquaSentinel(use_physics=c['physics'] and not args.no_physics,use_fp_head=c['fp_head'] and not args.no_fp_head).to(dev)
 if args.warm_start: m.load_state_dict(torch.load(args.warm_start,map_location=dev),strict=False)
 opt=torch.optim.Adam(m.parameters(),lr=c['lr'],weight_decay=c['weight_decay']); sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=c['scheduler_factor'],patience=c['scheduler_patience'],mode='min'); best=1e9
 for ep in range(c['epochs']):
  for par in m.ea.parameters(): par.requires_grad=ep>=c['freeze_epochs']
  for par in m.eb.parameters(): par.requires_grad=ep>=c['freeze_epochs']
  m.train(); losses=[]
  for a,b,y in dl:
   a,b,y=a.to(dev),b.to(dev),y.to(dev); opt.zero_grad(set_to_none=True); s,z=m(a,b); loss,_=total_loss(s,z,y,c['loss']['dice'],c['loss']['bce'],c['loss']['focal'],c['loss']['auxiliary'],c['focal_alpha'],c['focal_gamma']); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),c['grad_clip']); opt.step(); losses.append(loss.item())
  v=float(np.mean(losses)); sch.step(v); print(f'epoch {ep+1}/{c["epochs"]} loss={v:.5f}')
  if v<best: best=v; torch.save(m.state_dict(),out/'best.pt')
 print('best',best)
if __name__=='__main__': main()
