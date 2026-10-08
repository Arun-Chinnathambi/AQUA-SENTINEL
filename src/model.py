import torch
import torch.nn as nn
import torch.nn.functional as F
from .physics import physics_features

class ConvBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.BatchNorm2d(cout),nn.ReLU(inplace=True),nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.BatchNorm2d(cout),nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)

class Encoder(nn.Module):
    def __init__(self,in_ch):
        super().__init__(); self.stem=nn.Sequential(nn.Conv2d(in_ch,64,7,stride=2,padding=3,bias=False),nn.BatchNorm2d(64),nn.ReLU(inplace=True)); self.b1=ConvBlock(64,64); self.b2=ConvBlock(64,128); self.b3=ConvBlock(128,256); self.b4=ConvBlock(256,512); self.pool=nn.MaxPool2d(2)
    def forward(self,x,phys=None):
        x=self.stem(x); f1=self.b1(x); x=self.pool(f1); f2=self.b2(x); x=self.pool(f2); f3=self.b3(x); x=self.pool(f3); f4=self.b4(x); x=self.pool(f4); return [f1,f2,f3,f4],x

class Gate(nn.Module):
    def __init__(self,c): super().__init__(); self.g=nn.Sequential(nn.Conv2d(2*c,c,1,bias=False),nn.BatchNorm2d(c),nn.Sigmoid())
    def forward(self,a,b):
        al=self.g(torch.cat([a,b],1)); return al*a+(1-al)*b

class ASPP(nn.Module):
    def __init__(self,c=512):
        super().__init__(); rates=[1,2,4,6]; self.br=nn.ModuleList([nn.Sequential(nn.Conv2d(c,128,3,padding=r,dilation=r,bias=False),nn.BatchNorm2d(128),nn.ReLU(inplace=True)) for r in rates]); self.proj=nn.Sequential(nn.Conv2d(512,512,1,bias=False),nn.BatchNorm2d(512),nn.ReLU(inplace=True))
    def forward(self,x): return self.proj(torch.cat([b(x) for b in self.br],1))

class DecoderBlock(nn.Module):
    def __init__(self,cin,skip,cout): self_up=1; super().__init__(); self.up=nn.ConvTranspose2d(cin,cout,2,stride=2); self.conv=ConvBlock(cout+skip,cout)
    def forward(self,x,s): x=self.up(x); x=F.interpolate(x,size=s.shape[-2:],mode='bilinear',align_corners=False); return self.conv(torch.cat([x,s],1))

class AquaSentinel(nn.Module):
    def __init__(self,in_a=1,in_b=1,use_physics=True,use_fp_head=True):
        super().__init__(); self.use_physics=use_physics; self.use_fp_head=use_fp_head
        self.pa=nn.Conv2d(in_a+2,in_a,1,bias=False) if use_physics else nn.Identity(); self.pb=nn.Conv2d(in_b+2,in_b,1,bias=False) if use_physics else nn.Identity()
        self.ea=Encoder(in_a); self.eb=Encoder(in_b); self.gates=nn.ModuleList([Gate(64),Gate(128),Gate(256),Gate(512)]); self.aspp=ASPP(512)
        self.d4=DecoderBlock(512,512,256); self.d3=DecoderBlock(256,256,128); self.d2=DecoderBlock(128,128,64); self.d1=DecoderBlock(64,64,64); self.head=nn.Conv2d(64,1,1); self.fp=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(512,1)) if use_fp_head else None
    def forward(self,a,b):
        if self.use_physics:
            p=physics_features(a); a=self.pa(torch.cat([a,p],1)); b=self.pb(torch.cat([b,p],1))
        sa,ba=self.ea(a); sb,bb=self.eb(b); fs=[self.gates[i](sa[i],sb[i]) for i in range(4)]; x=self.gates[3](ba,bb); x=self.aspp(x)
        z=self.fp(x) if self.fp is not None else torch.zeros((a.size(0),1),device=a.device)
        x=self.d4(x,fs[3]); x=self.d3(x,fs[2]); x=self.d2(x,fs[1]); x=self.d1(x,fs[0]); x=F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False); return self.head(x),z
