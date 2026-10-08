"""Object-level DARTIS evaluation helper.
Expected predictions CSV: image, xmin,ymin,xmax,ymax,score
Ground truth XML files are parsed from Pascal VOC. No segmentation metrics are computed.
"""
import argparse,csv,xml.etree.ElementTree as ET
from pathlib import Path

def gt_boxes(xml):
 r=ET.parse(xml).getroot(); out=[]
 for o in r.findall('object'):
  bb=o.find('bndbox'); out.append(tuple(float(bb.find(k).text) for k in ['xmin','ymin','xmax','ymax']))
 return out
def iou(a,b):
 x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1); aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]); return inter/(aa+bb-inter+1e-12)
p=argparse.ArgumentParser(); p.add_argument('--pred',required=True); p.add_argument('--xml-root',required=True); p.add_argument('--iou',type=float,default=.5); a=p.parse_args(); tp=fp=fn=0
with open(a.pred) as f:
 for row in csv.DictReader(f):
  g=gt_boxes(Path(a.xml_root)/row['image'].replace(Path(row['image']).suffix,'.xml'))
  pr=tuple(float(row[k]) for k in ['xmin','ymin','xmax','ymax']); hit=max([iou(pr,x) for x in g] or [0])>=a.iou
  tp+=int(hit); fp+=int(not hit)
  if not hit: fn+=0
print({'TP':tp,'FP':fp,'FN':fn,'precision':tp/(tp+fp+1e-12)})
