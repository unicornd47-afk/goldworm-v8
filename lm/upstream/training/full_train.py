"""Full v8 run: continue pretrain from trial1/pretrain_best.pt for 350M byte positions, then SFT on teacher_v4. Resumable; dev-CE selection only."""
import os,sys,time,json,math,random,signal
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
import numpy as np, torch, torch.nn.functional as F
from goldworm_v8 import GoldWormV8,GoldWormV6Config
ROOT=Path(__file__).resolve().parent.parent;T=ROOT/'v8/trial1';P=ROOT/'v8/full1';TV=ROOT/'v8/teacher_v4'
torch.set_num_threads(4);rng=random.Random(9901);torch.manual_seed(9901);stop=False
PRE_POS=350_000_000;SFT_S=1200;PEAK=1.5e-4;MINF=.1
def stopped(s,f):
 global stop;stop=True
signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
def dump(x,p):
 t=str(p)+'.tmp';json.dump(x,open(t,'w'),indent=2);os.replace(t,p)
MET=open(P/'metrics.jsonl','a')
def metric(**k):
 k['t']=time.time();MET.write(json.dumps(k)+'\n');MET.flush()
def save(m,opt,path,meta,state=None):
 o=dict(cfg=asdict(m.cfg),model_state_dict=m.state_dict(),metadata=meta,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),python_rng=rng.getstate())
 if opt is not None:o['optimizer']=opt.state_dict()
 if state is not None:o['state']=state
 t=str(path)+'.tmp';torch.save(o,t);os.replace(t,path)
def main():
 global stop
 train=torch.from_numpy(np.load(T/'train.npy'));dev=torch.from_numpy(np.load(T/'dev.npy'))
 S=json.load(open(TV/'train_data.json'))['data'];Dall=json.load(open(TV/'dev_data.json'))['data']
 groups=defaultdict(list)
 for d in S:groups[d['prompt']].append(d)
 groupvals=list(groups.values());gdev=defaultdict(list)
 for d in Dall:gdev[d['prompt']].append(d)
 D=[gdev[p][0] for p in sorted(gdev)]
 m=GoldWormV8(GoldWormV6Config(ctx=2048,d_model=768,n_layer=12,n_head=12,dropout=0.0)).cuda()
 def general_batch(data,bs=2,L=2048,fixed=None):
  ids=torch.tensor([rng.randrange(len(data)) for _ in range(bs)]) if fixed is None else torch.arange(fixed,min(fixed+bs,len(data)))
  x=data[ids,:L+1].long().cuda();return x[:,:-1],x[:,1:],None
 def pair_batch(rows):
  seq=[(d['prompt']+d['body']+'\n').encode()[:2048] for d in rows];L=max(len(b) for b in seq);X=torch.zeros(len(rows),L,dtype=torch.long);mask=torch.zeros(len(rows),L-1)
  for i,(d,b) in enumerate(zip(rows,seq)):
   X[i,:len(b)]=torch.tensor(list(b));mask[i,max(0,len(d['prompt'].encode())-1):len(b)-1]=1
  return X[:,:-1].cuda(),X[:,1:].cuda(),mask.cuda()
 def loss_for(b):
  x,y,mask=b
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
 # resume
 st=None;opt_sd=None
 if (P/'last.pt').exists():
  c=torch.load(P/'last.pt',map_location='cpu',weights_only=False);m.load_state_dict(c['model_state_dict']);st=c.get('state');opt_sd=c.get('optimizer')
  torch.set_rng_state(c['torch_rng']);torch.cuda.set_rng_state(c['cuda_rng']);rng.setstate(c['python_rng']);del c;print('RESUME',st,flush=True);metric(event='resume',state=st)
 else:
  c=torch.load(T/'pretrain_best.pt',map_location='cpu',weights_only=True);m.load_state_dict(c['model_state_dict']);del c
 log=json.load(open(P/'progress.json'))['log'] if (st and (P/'progress.json').exists()) else []
 phases=['pretrain','sft']
 if st and st['phase']=='done':print('ALREADY_DONE');return
 startidx=phases.index(st['phase']) if st else 0
 for phase in phases[startidx:]:
  if stop:break
  resumed=bool(st) and st['phase']==phase and not st.get('fresh')
  if phase=='sft' and not resumed:
   c=torch.load(P/'pretrain_best.pt',map_location='cpu',weights_only=True);m.load_state_dict(c['model_state_dict']);del c
  opt=torch.optim.AdamW(m.parameters(),lr=PEAK if phase=='pretrain' else 1e-5,betas=(.9,.95),weight_decay=.1 if phase=='pretrain' else .01)
  if resumed and opt_sd:opt.load_state_dict(opt_sd)
  if resumed:step,presented,best,elapsed_prev=st['step'],st['presented'],st['best'],st['phase_s']
  else:
   step=presented=0;elapsed_prev=0.;best=validate(phase)
   meta=dict(phase=phase,step=0,dev_loss=best,phase_s=0,init='trial1/pretrain_best.pt (own v8 lineage)' if phase=='pretrain' else 'full1/pretrain_best.pt');save(m,None,P/(phase+'_best.pt'),meta);log.append(meta);metric(event='dev',**meta);print('PHASE_START',meta,flush=True)
  st=None;start=time.monotonic()-elapsed_prev;lastval=lastsave=time.monotonic();losses=[];tw=time.monotonic();pw=presented
  def done():return presented>=PRE_POS if phase=='pretrain' else time.monotonic()-start>=SFT_S
  while not done() and not stop:
   opt.zero_grad(set_to_none=True)
   if phase=='pretrain':prog=presented/PRE_POS;lr=PEAK*(MINF+(1-MINF)*(1+math.cos(math.pi*prog))/2)*min(1,(step+1)/200)
   else:lr=1e-5
   for g in opt.param_groups:g['lr']=lr
   if phase=='pretrain':b=general_batch(train)
   elif rng.random()<.2:b=general_batch(train,bs=4,L=256)
   else:b=pair_batch([rng.choice(rng.choice(groupvals)) for _ in range(4)])
   loss=loss_for(b)
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();step+=1;presented+=b[0].numel();losses.append(float(loss.detach()))
   if step%100==0:
    now=time.monotonic();bps=(presented-pw)/(now-tw);tw,pw=now,presented;tl=sum(losses[-100:])/100
    tot=PRE_POS if phase=='pretrain' else None
    metric(event='train',phase=phase,step=step,loss=tl,lr=lr,bytes_s=bps,presented=presented,total=tot,phase_s=now-start,sft_budget_s=SFT_S)
    print(phase,'step',step,'loss',round(tl,4),'bytes/s',round(bps),'presented',presented,flush=True)
   now=time.monotonic()
   if now-lastval>=(300 if phase=='pretrain' else 60):
    v=validate(phase);meta=dict(phase=phase,step=step,dev_loss=v,train_loss_mean=sum(losses[-100:])/max(1,len(losses[-100:])),phase_s=time.monotonic()-start,presented_bytes=presented,lr=lr);log.append(meta);metric(event='dev',**meta);print('DEV',meta,flush=True)
    if v<best:best=v;save(m,None,P/(phase+'_best.pt'),meta);metric(event='ckpt',file=phase+'_best.pt',step=step)
    dump(dict(status='RUNNING',phase=phase,log=log),P/'progress.json');lastval=time.monotonic()
   if now-lastsave>=600:
    save(m,opt,P/'last.pt',dict(phase=phase,step=step),state=dict(phase=phase,step=step,presented=presented,best=best,phase_s=now-start));metric(event='ckpt',file='last.pt',step=step);lastsave=time.monotonic()
  now=time.monotonic();v=validate(phase);meta=dict(phase=phase,step=step,dev_loss=v,phase_s=now-start,presented_bytes=presented,stopped=stop);log.append(meta);metric(event='dev',**meta)
  if v<best:best=v;save(m,None,P/(phase+'_best.pt'),meta)
  save(m,opt if stop else None,P/'last.pt',meta,state=dict(phase=phase,step=step,presented=presented,best=best,phase_s=now-start) if stop else dict(phase='sft',fresh=True) if phase=='pretrain' else dict(phase='done'))
  dump(dict(status='STOPPED' if stop else 'PHASE_DONE',phase=phase,log=log),P/'progress.json');print('PHASE_END',meta,flush=True);metric(event='phase_end',phase=phase)
 if stop:sys.exit(130)
 dump(dict(status='TRAIN_DONE',selected_checkpoint='sft_best.pt',log=log),P/'TRAIN_DONE.json');metric(event='train_done')
if __name__=='__main__':main()
