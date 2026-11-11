"""GoldWorm v8 directly extends OWN gwlm_v6.py: only RoPE/context and SDPA change.
Byte vocab, affine pre-LN residual blocks, bias-free QKVO/MLP GELU, tied head are reused.
No imported architecture or pretrained external weights.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))  # release layout: gwlm_v6.py sits next to this file (original: parent.parent)
import torch
from torch.nn import functional as F
from gwlm_v6 import GoldWormV6, GoldWormV6Config, CausalSelfAttention
class RopeAttention(CausalSelfAttention):
 def __init__(self,d_model,n_head,dropout):
  super().__init__(d_model,n_head,dropout)
  assert self.d_head%2==0
  self.register_buffer('rope_freq',1/(10000**(torch.arange(0,self.d_head,2).float()/self.d_head)),persistent=False)
 def rotate(self,x,pos):
  a=pos.float()[:,None]*self.rope_freq[None,:];c=a.cos().to(x.dtype)[None,None];s=a.sin().to(x.dtype)[None,None];e,o=x[...,::2],x[...,1::2]
  return torch.stack((e*c-o*s,e*s+o*c),-1).flatten(-2)
 def forward(self,x,cache=None):
  B,T,D=x.shape
  q=self.wq(x).view(B,T,self.n_head,self.d_head).transpose(1,2);k=self.wk(x).view(B,T,self.n_head,self.d_head).transpose(1,2);v=self.wv(x).view(B,T,self.n_head,self.d_head).transpose(1,2)
  start=cache[0].shape[2] if cache else 0;pos=torch.arange(start,start+T,device=x.device);q=self.rotate(q,pos);k=self.rotate(k,pos)
  if cache:k=torch.cat((cache[0],k),2);v=torch.cat((cache[1],v),2)
  if cache is not None:cache[:]=[k,v]
  y=F.scaled_dot_product_attention(q,k,v,is_causal=T>1,dropout_p=self.dropout if self.training else 0)
  return self.resid_dropout(self.wo(y.transpose(1,2).contiguous().view(B,T,D)))
class GoldWormV8(GoldWormV6):
 def __init__(self,cfg=None):
  cfg=cfg or GoldWormV6Config(ctx=2048,d_model=768,n_layer=12,n_head=12,dropout=0.0)
  assert cfg.vocab_size==256 and cfg.tie_embeddings
  super().__init__(cfg)
  del self.pos_emb
  for block in self.blocks:
   attn=RopeAttention(cfg.d_model,cfg.n_head,cfg.dropout);attn.load_state_dict(block.attn.state_dict());block.attn=attn
 def forward(self,idx,targets=None,cache=None):
  B,T=idx.shape;start=cache[0][0].shape[2] if cache and cache[0] else 0;assert start+T<=self.cfg.ctx
  x=self.drop(self.tok_emb(idx))
  if cache is None:
   for block in self.blocks:x=block(x)
  else:
   for i,block in enumerate(self.blocks):
    x=x+block.attn(block.ln1(x),cache[i]);x=x+block.mlp(block.ln2(x))
  logits=self.head(self.ln_final(x))
  if targets is not None:return logits,F.cross_entropy(logits.reshape(-1,self.cfg.vocab_size),targets.reshape(-1))
  return logits
