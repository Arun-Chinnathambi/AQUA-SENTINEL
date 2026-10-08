import argparse,json,random
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--out',required=True); p.add_argument('--train',type=float,default=.70); p.add_argument('--val',type=float,default=.15); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
items=json.loads(Path(a.manifest).read_text()); random.Random(a.seed).shuffle(items); n=len(items); nt=int(n*a.train); nv=int(n*a.val); d={'train':items[:nt],'val':items[nt:nt+nv],'test':items[nt+nv:]}; Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(d,indent=2)); print({k:len(v) for k,v in d.items()})
