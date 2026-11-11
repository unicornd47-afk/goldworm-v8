"""Bounded 45-minute pretrain + 10-minute SFT; selection solely on dev CE."""
import os,sys,time,json,math,random,signal
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
import numpy as np
import torch
import torch.nn.functional as F
from goldworm_v8 import GoldWormV8,GoldWormV6Config
ROOT=Path(__file__).resolve().parent.parent;P=ROOT/'v8/trial1';torch.set_num_threads(4);torch.manual_seed(8801);rng=random.Random(8801);stop=False

def stopped(sig,frame):
 global stop;stop=True
signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
def dump(x,path):
 tmp=str(path)+'.tmp';json.dump(x,open(tmp,'w'),indent=2);os.replace(tmp,path)
def save(m,opt,path,meta):
 obj=dict(cfg=asdict(m.cfg),model_state_dict=m.state_dict(),metadata=meta,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),python_rng=rng.getstate())
 if opt is not None:obj['optimizer']=opt.state_dict()
 tmp=str(path)+'.tmp';torch.save(obj,tmp);os.replace(tmp,path)
def main():
 global stop
 train=torch.from_numpy(np.load(P/'train.npy'));dev=torch.from_numpy(np.load(P/'dev.npy'));S=json.load(open(P/'sft_train.json'));D=json.load(open(P/'sft_dev.json'));groups=defaultdict(list)
 for d in S:groups[d['prompt']].append(d)
 groupvals=list(groups.values());gdev=defaultdict(list)
 for d in D:gdev[d['prompt']].append(d)
 # Fixed nonduplicate prompt dev representatives; no test-set outcomes used for selection.
 D=[gdev[p][0] for p in sorted(gdev)]
 m=GoldWormV8(GoldWormV6Config(ctx=2048,d_model=768,n_layer=12,n_head=12,dropout=0.0)).cuda();opt=torch.optim.AdamW(m.parameters(),lr=2e-4,betas=(.9,.95),weight_decay=.1)
 def general_batch(data,bs=2,L=2048,fixed=None):
  ids=torch.tensor([rng.randrange(len(data)) for _ in range(bs)]) if fixed is None else torch.arange(fixed,min(fixed+bs,len(data)))
  x=data[ids,:L+1].long().cuda();return x[:,:-1],x[:,1:],None
 def pair_batch(rows):
  seq=[(d['prompt']+d['body']+'\n').encode()[:2048] for d in rows];L=max(len(b) for b in seq);X=torch.zeros(len(rows),L,dtype=torch.long);mask=torch.zeros(len(rows),L-1)
  for i,(d,b) in enumerate(zip(rows,seq)):
   X[i,:len(b)]=torch.tensor(list(b));mask[i,max(0,len(d['prompt'].encode())-1):len(b)-1]=1
  return X[:,:-1].cuda(),X[:,1:].cuda(),mask.cuda()
 def loss_for(batch):
  x,y,mask=batch
  with torch.autocast('cuda',dtype=torch.bfloat16):lg=m(x)
  l=F.cross_entropy(lg.float().reshape(-1,256),y.reshape(-1),reduction='none').reshape_as(y)
  return l.mean() if mask is None else (l*mask).sum()/mask.sum().clamp_min(1)
 def validate(phase):
  m.eval();ls=[]
  with torch.no_grad():
   if phase=='pretrain':
    for i in range(0,min(32,len(dev)),2):ls.append(float(loss_for(general_batch(dev,fixed=i))))
   else:
    for d in D:ls.append(float(loss_for(pair_batch([d]))))
  m.train();return sum(ls)/len(ls)
 log=[];totalstart=time.monotonic()
 for phase,budget in [('pretrain',2700),('sft',600)]:
  if stop:break
  if phase=='sft':
   c=torch.load(P/'pretrain_best.pt',map_location='cpu',weights_only=True);m.load_state_dict(c['model_state_dict']);del c;opt=torch.optim.AdamW(m.parameters(),lr=1e-5,weight_decay=.01,betas=(.9,.95))
  initial=validate(phase);best=initial;start=time.monotonic();step=0;presented=0;lastval=start;lastsave=start;losses=[]
  meta=dict(phase=phase,step=step,dev_loss=best,phase_s=0,init='scratch random seed8801; no smoke weights (fresh concept-filtered lineage)' if phase=='pretrain' else 'pretrain_best.pt')
  save(m,None,P/(phase+'_best.pt'),meta);log.append(meta);print('PHASE_START',meta,flush=True)
  while time.monotonic()-start<budget and not stop:
   opt.zero_grad(set_to_none=True);progress=min(1,(time.monotonic()-start)/budget)
   lr=(2e-4*(.15+.85*(1+math.cos(math.pi*progress))/2)*min(1,(step+1)/50)) if phase=='pretrain' else 1e-5
   for group in opt.param_groups:group['lr']=lr
   if phase=='pretrain':batch=general_batch(train)
   elif rng.random()<.2:batch=general_batch(train,bs=4,L=256)
   else:batch=pair_batch([rng.choice(rng.choice(groupvals)) for _ in range(4)])
   loss=loss_for(batch)
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();step+=1;presented+=batch[0].numel();losses.append(float(loss.detach()))
   if step%100==0:print(phase,'step',step,'loss',sum(losses[-100:])/len(losses[-100:]),'elapsed',time.monotonic()-start,flush=True)
   now=time.monotonic();interval=300 if phase=='pretrain' else 60
   if now-lastval>=interval:
    v=validate(phase);meta=dict(phase=phase,step=step,dev_loss=v,train_loss_mean=sum(losses[-100:])/len(losses[-100:]),phase_s=time.monotonic()-start,presented_bytes=presented,lr=lr);log.append(meta);print('DEV',meta,flush=True)
    if v<best:best=v;save(m,None,P/(phase+'_best.pt'),meta)
    dump(dict(status='RUNNING',phase=phase,remaining_phase_s=max(0,budget-(time.monotonic()-start)),log=log),P/'progress.json');lastval=time.monotonic()
   if now-lastsave>=600:save(m,opt,P/'last.pt',dict(phase=phase,step=step,phase_s=now-start));lastsave=time.monotonic()
  v=validate(phase);meta=dict(phase=phase,step=step,dev_loss=v,phase_s=time.monotonic()-start,presented_bytes=presented,bytes_s=presented/(time.monotonic()-start),train_loss_first100=sum(losses[:100])/max(1,len(losses[:100])),train_loss_last100=sum(losses[-100:])/max(1,len(losses[-100:])),stopped=stop);log.append(meta)
  if v<best:best=v;save(m,None,P/(phase+'_best.pt'),meta)
  save(m,opt,P/'last.pt',meta);dump(dict(status='TRAINING_STOPPED' if stop else 'PHASE_DONE',phase=phase,log=log,elapsed_s=time.monotonic()-totalstart),P/'progress.json');print('PHASE_END',meta,flush=True)
 dump(dict(status='INTERRUPTED' if stop else 'TRAIN_DONE',selected_checkpoint='sft_best.pt' if not stop else 'last.pt',elapsed_s=time.monotonic()-totalstart,log=log),P/'TRAIN_DONE.json')
 if stop:sys.exit(130)
if __name__=='__main__':main()
