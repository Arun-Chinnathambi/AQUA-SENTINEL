import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--images',required=True); p.add_argument('--masks',required=True); p.add_argument('--out',required=True); p.add_argument('--secondary',required=True); a=p.parse_args()
# Generic stem matching; adjust suffix rules if the downloaded release uses a different naming convention.
imgs={x.stem:x for x in Path(a.images).rglob('*') if x.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff'}}
masks={x.stem:x for x in Path(a.masks).rglob('*') if x.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff'}}
sec={x.stem:x for x in Path(a.secondary).rglob('*') if x.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff','.npy'}}
items=[]
for stem,pth in imgs.items():
 if stem in masks and stem in sec: items.append({'a':str(pth.relative_to(Path(a.images).parent)),'b':str(sec[stem].relative_to(Path(a.secondary).parent)),'mask':str(masks[stem].relative_to(Path(a.masks).parent))})
Path(a.out).write_text(json.dumps(items,indent=2)); print('paired items:',len(items))
