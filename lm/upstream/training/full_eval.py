"""Frozen k16/.8/top40/max160 protocol; paired task bootstrap B10000/seed777."""
import os,sys,json,time,hashlib,math
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
import torch.nn.functional as F
from goldworm_v8 import GoldWormV8,GoldWormV6Config
ROOT=Path(__file__).resolve().parent.parent;P=ROOT/'v8/full1';sys.path.insert(0,str(ROOT));torch.set_num_threads(4)
from goldworm_agent.heal import heal
from goldworm_agent.sandbox import run_tests,valid

def dump(obj,path):
 t=str(path)+'.tmp';json.dump(obj,open(t,'w'),indent=2);os.replace(t,path)
@torch.no_grad()
def sample(m,prompt,seed):
 p=list(prompt.encode());g=torch.Generator().manual_seed(seed);cache=[[] for _ in m.blocks];x=torch.tensor([p]*16,device='cuda');out=[[] for _ in range(16)];done=[False]*16
 for t in range(min(160,m.cfg.ctx-len(p))):
  # Sampling arithmetic float32 like the archived baselines; CUDA SDPA may differ numerically.
  logits=m(x,cache=cache)[:,-1]/.8;v,_=logits.topk(40);logits[logits<v[:,-1:]]=-float('inf');nx=torch.multinomial(logits.softmax(-1).cpu(),1,generator=g)[:,0]
  for i,n in enumerate(nx.tolist()):
   if not done[i]:
    out[i].append(n);s=bytes(out[i])
    if len(s)>=2 and s[-2:-1]==b'\n' and s[-1:] not in (b' ',b'\t',b'\n'):out[i].pop();done[i]=True
  if all(done):break
  x=nx[:,None].cuda()
 return [bytes(o).decode('utf8','replace') for o in out]
@torch.no_grad()
def bpc(m,prompt,body):
 b=list((prompt+body).encode())[:m.cfg.ctx];p=len(prompt.encode())
 if len(b)<=p:return float('inf')
 x=torch.tensor([b[:-1]],device='cuda');y=torch.tensor(b[1:],device='cuda');lg=m(x)[0].float();lp=lg.log_softmax(-1).gather(-1,y[:,None])[:,0];return float(-lp[p-1:].mean()/math.log(2))
def per_samples(rows):
 out=[]
 for r in rows:
  ss=r['samples'];v=[s['valid'] for s in ss];ssvalid=[s for s in ss if s['valid']]
  out.append(dict(id=r['id'],p1=float(np.mean([s['passed'] for s in ss])),p16=float(any(s['passed'] for s in ss)),valid=float(np.mean(v)),ranked=float(min(ss,key=lambda s:s['bpc'])['passed']),ranked_valid=float(min(ssvalid,key=lambda s:s['bpc'])['passed']) if ssvalid else 0.))
 return out
def bootstrap(base,new,keys):
 assert [r['id'] for r in base]==[r['id'] for r in new]
 g=np.random.default_rng(777);n=len(base);ix=g.integers(0,n,(10000,n));out={}
 for k in keys:
  a=np.array([r[k] for r in base]);b=np.array([r[k] for r in new]);d=b-a;out[k]=dict(base=float(a.mean()),new=float(b.mean()),diff=float(d.mean()),ci95=np.quantile(d[ix].mean(1),[.025,.975],method='linear').tolist())
 return out
def main():
 t0=time.monotonic();checkpoint=P/'sft_best.pt';c=torch.load(checkpoint,map_location='cpu',weights_only=True);m=GoldWormV8(GoldWormV6Config(**c['cfg']));m.load_state_dict(c['model_state_dict']);m.cuda().eval();selected=c['metadata'];del c;data=json.load(open(ROOT/'v8/trial1/data_stats.json'));allresults={}
 baseline_paths={'eval':dict(gen0=('gen0_harness_ruby.json','heal_gen0_ruby.json'),v2=('eval1_ruby_ftv2.json','eval1_heal_ftv2.json')),'eval2':dict(gen0=('eval2_ruby_gen0.json','eval2_heal_gen0.json'),v2=('eval2_ruby_ftv2.json','eval2_heal_ftv2.json'))}
 for name in ['eval2','eval']:
  raw=(ROOT/name/'tasks.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==data['eval_hashes'][name];tasks=json.loads(raw);rows=[];hrows=[];start=time.monotonic()
  partial=P/(name+'_samples.partial.json')
  if partial.exists():
   previous=json.load(open(partial));rows=previous['rows'];assert [r['id'] for r in rows]==[t['id'] for t in tasks[:len(rows)]]
   for r in rows:
    base=per_samples([r])[0];ss=r['samples'];hrows.append(dict(**base,hp1=float(np.mean([s['heal_passed'] for s in ss])),hp16=float(any(s['heal_passed'] for s in ss)),hvalid=float(np.mean([s['heal_valid'] for s in ss])),healed=sum(s['completion']!=s['healed_completion'] for s in ss)))
   print('EVAL_RESUME',name,len(rows),'completed tasks',flush=True)
  for i,t in enumerate(tasks):
   if i<len(rows):continue
   comps=sample(m,t['prompt'],20261107+i);ss=[];hp=[];hv=[];hr=0
   for body in comps:
    code=t['prompt']+body;res=run_tests(code,t['tests']);fixed,rules=heal(t['prompt'],body);hres=run_tests(t['prompt']+fixed,t['tests']);hp.append(hres['passed']);hv.append(valid(t['prompt']+fixed));hr+=bool(rules)
    ss.append(dict(completion=body,valid=valid(code),passed=res['passed'],bpc=bpc(m,t['prompt'],body),err=res['err'],healed_completion=fixed,heal_passed=hres['passed'],heal_valid=hv[-1]))
   rows.append(dict(id=t['id'],tier=t['tier'],concept=t.get('concept'),seed=20261107+i,samples=ss));base=per_samples(rows[-1:])[0];hrows.append(dict(**base,hp1=float(np.mean(hp)),hp16=float(any(hp)),hvalid=float(np.mean(hv)),healed=hr))
   dump(dict(ckpt=str(checkpoint),k=16,rows=rows,wall_s=time.monotonic()-start),P/(name+'_samples.partial.json'));print('EVAL',name,t['id'],'passed',sum(s['passed'] for s in ss),'heal',sum(hp),flush=True)
  sampleout=dict(ckpt=str(checkpoint),k=16,rows=rows,wall_s=time.monotonic()-start);dump(sampleout,P/(name+'_samples.json'));keys=['p1','p16','valid','hp1','hp16','hvalid'];summary={k:float(np.mean([r[k] for r in hrows])) for k in keys};summary['solved_tasks']=sum(r['p16'] for r in hrows);summary['heal_solved_tasks']=sum(r['hp16'] for r in hrows)
  comparisons={}
  for baseline,(sf,hf) in baseline_paths[name].items():
   old=json.load(open(ROOT/'results'/sf));oldh=json.load(open(ROOT/'results'/hf));assert old['k']==16;assert [r['id'] for r in old['rows']]==[r['id'] for r in rows];assert [r['seed'] for r in old['rows']]==[r['seed'] for r in rows]
   comparisons[baseline]=dict(model_only=bootstrap(per_samples(old['rows']),per_samples(rows),['p1','p16','ranked','ranked_valid','valid']),with_heal=bootstrap(oldh['rows'],hrows,['p1','p16','hp1','hp16','valid','hvalid']),sample_source=sf,heal_source=hf)
  TR=ROOT/'v8/trial1';old=json.load(open(TR/(name+'_samples.json')));oldh=json.load(open(TR/(name+'_summary.json')))['rows'];assert [r['id'] for r in old['rows']]==[r['id'] for r in rows];assert [r['seed'] for r in old['rows']]==[r['seed'] for r in rows]
  comparisons['trial1']=dict(model_only=bootstrap(per_samples(old['rows']),per_samples(rows),['p1','p16','ranked','ranked_valid','valid']),with_heal=bootstrap(oldh,hrows,['p1','p16','hp1','hp16','valid','hvalid']),sample_source='v8/trial1/'+name+'_samples.json',heal_source='v8/trial1/'+name+'_summary.json')
  dump(dict(summary=summary,rows=hrows,comparisons=comparisons),P/(name+'_summary.json'));allresults[name]=dict(summary=summary,comparisons=comparisons,wall_s=time.monotonic()-start)
 train=json.load(open(P/'TRAIN_DONE.json'));evalsec=time.monotonic()-t0
 result=dict(status='FULL_DONE',architecture='Own GoldWormV8 d768/L12/H12 byte256 RoPE2048, no external weights',init='v8/trial1/pretrain_best.pt',pretrain_positions=350000000,sft_data='v8/teacher_v4 train 324 / dev 38',selected_checkpoint=str(checkpoint),selection=selected,training=train,results=allresults,eval_wall_s=evalsec,protocol=dict(k=16,temperature=.8,topk=40,maxnew=160,sampling_seed='20261107+task_index',bootstrap_B=10000,bootstrap_seed=777),push=False)
 dump(result,P/'DONE.json')
 L=['# v8 full1 Ergebnisse','',f"Checkpoint: {checkpoint} (Auswahl nur per Dev-CE: {selected})",'']
 for n,r in allresults.items():
  L.append(f"## {n}");L.append('Summary: '+json.dumps(r['summary']))
  for b,c in r['comparisons'].items():
   for k in ['p1','p16','valid']:x=c['model_only'][k];L.append(f"- vs {b} {k}: base {x['base']:.4f} new {x['new']:.4f} diff {x['diff']:+.4f} CI95 [{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}]")
  L.append('')
 L.append('Hinweis: teacher_v4 nutzt gelockerten Konzeptfilter; eval2 ist kein garantierter Test auf fremde Konzepte. Nichts gepusht.')
 open(P/'RESULTS_FULL.md','w').write('\n'.join(L));print('FULL_DONE',flush=True)
if __name__=='__main__':main()
