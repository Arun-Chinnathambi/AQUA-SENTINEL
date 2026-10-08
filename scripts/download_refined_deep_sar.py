#!/usr/bin/env python3
import argparse,requests
from pathlib import Path
URLS={
'images.zip':'https://zenodo.org/api/records/15298010/files/images.zip/content',
'masks.zip':'https://zenodo.org/api/records/15298010/files/masks.zip/content'}
def dl(url,out):
 r=requests.get(url,stream=True,timeout=120); r.raise_for_status(); total=int(r.headers.get('content-length',0)); n=0
 with open(out,'wb') as f:
  for ch in r.iter_content(1024*1024): f.write(ch); n+=len(ch); print(f'\r{n/1e6:.1f} MB / {total/1e6:.1f} MB',end='')
 print()
p=argparse.ArgumentParser(); p.add_argument('--out',default='data/raw/refined_deep_sar'); p.add_argument('--files',nargs='+',choices=URLS.keys(),default=list(URLS)); a=p.parse_args(); d=Path(a.out); d.mkdir(parents=True,exist_ok=True)
for name in a.files: print('Downloading',name); dl(URLS[name],d/name)
print('Dataset DOI: https://doi.org/10.5281/zenodo.15298010')
