import ast,hashlib,importlib.util,importlib.metadata as M,json,re,random,sys,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent;OUT=ROOT/'v8/trial1';OUT.mkdir(exist_ok=True);(OUT/'licenses').mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'distill'))
from propose_tasks import BLOCK_EVAL1,BLOCK_EVAL2
PACKAGES=[('torch','BSD-3-Clause'),('transformers','Apache-2.0'),('sympy','BSD-3-Clause'),('scipy','BSD-3-Clause'),('numpy','BSD-3-Clause'),('datasets','Apache-2.0'),('gradio','Apache-2.0'),('huggingface_hub','Apache-2.0')]
CONCEPTS=['factorial','palindrome','fibonacci','fizzbuzz','fizz buzz','collatz','anagram','caesar','roman numeral','pascal triangle','pascal_triangle','ipv4','camel_to_snake','camel to snake','binary_search','binary search','merge_sorted','merge sorted','run length encod','run_length_encod','balanced bracket','balanced parenthes','greatest common divisor','gcd','transpose matrix']
def main():
 tasks=[];hashes={};G=set()
 for folder in ['eval','eval2']:
  b=(ROOT/folder/'tasks.json').read_bytes();hashes[folder]=hashlib.sha256(b).hexdigest();tasks+=json.loads(b)
 for t in tasks:
  for s in [t['prompt']]+t['tests']:
   b=s.encode();G.update(b[i:i+13] for i in range(len(b)-12))
 names={t['entry'] for t in tasks};namepat=re.compile(rb'\b(?:def|class)\s+(?:'+b'|'.join(n.encode() for n in sorted(names))+rb')\s*[(:]');conceptpat=re.compile('|'.join(re.escape(x) for x in CONCEPTS),re.I)
 def reject(b):
  if any(b[i:i+13] in G for i in range(len(b)-12)):return 'ngram'
  if namepat.search(b) or conceptpat.search(b.decode('utf8','replace')):return 'concept'
  return None
 train=[];dev=[];manifest=[];seen=set();versions=[];stats=dict(raw_bytes=0,ngram=0,concept=0,duplicate=0)
 for package,license in PACKAGES:
  d=M.distribution(package);base=Path(importlib.util.find_spec(package).origin).parent;lics=[]
  for f in d.files or []:
   if '.dist-info/' in str(f) and any(s in str(f).lower() for s in ['license','copying','notice']):
    src=d.locate_file(f)
    if src.is_file():dst=OUT/'licenses'/(package+'_'+str(f).replace('/','_'));shutil.copyfile(src,dst);lics.append(str(dst.relative_to(OUT)))
  if package=='numpy':shutil.copyfile('/usr/share/licenses/python3-numpy/LICENSE.txt',OUT/'licenses/numpy_LICENSE.txt');lics.append('licenses/numpy_LICENSE.txt')
  assert lics,(package,'missing license')
  versions.append(dict(package=package,version=d.version,license=license,notices=lics,source=str(base)))
  for p in sorted(base.rglob('*.py')):
   if p.is_symlink():continue
   b=p.read_bytes();stats['raw_bytes']+=len(b);parts=[]
   for j in range(0,len(b)-2048,2049):
    chunk=b[j:j+2049];reason=reject(chunk)
    if reason:stats[reason]+=1;continue
    digest=hashlib.sha256(chunk).hexdigest()
    if digest in seen:stats['duplicate']+=1;continue
    seen.add(digest);parts.append(np.frombuffer(chunk,np.uint8).copy())
   if parts:
    # file-separated deterministic dev; exact windows additionally deduped globally.
    digest=hashlib.sha256(b).hexdigest();split='dev' if int(digest[:8],16)%20==0 else 'train';(dev if split=='dev' else train).extend(parts);manifest.append(dict(package=package,path=str(p.relative_to(base)),sha256=digest,chunks=len(parts),split=split))
 np.save(OUT/'train.npy',np.stack(train));np.save(OUT/'dev.npy',np.stack(dev));json.dump(manifest,open(OUT/'manifest.json','w'))
 devprompts={d['prompt'] for n in ['dev_data','dev_data_v2'] for d in json.load(open(ROOT/'distill'/f'{n}.json'))['data']};tr=[];dv=[];seenpairs=set();pairrejected=0
 for name in ['train_data','train_data_v2','dev_data','dev_data_v2']:
  for d in json.load(open(ROOT/'distill'/f'{name}.json'))['data']:
   b=(d['prompt']+d['body']).encode();digest=hashlib.sha256(b).hexdigest()
   if digest in seenpairs:continue
   seenpairs.add(digest)
   if reject(b) or any(s in d['prompt'].lower() for s in BLOCK_EVAL1+BLOCK_EVAL2) or len(b)>2048:pairrejected+=1;continue
   (dv if d['prompt'] in devprompts else tr).append(d)
 json.dump(tr,open(OUT/'sft_train.json','w'));json.dump(dv,open(OUT/'sft_dev.json','w'))
 stats.update(train_bytes=len(train)*2049,dev_bytes=len(dev)*2049,train_chunks=len(train),dev_chunks=len(dev),sft_train=len(tr),sft_dev=len(dv),sft_unique_train_prompts=len({x['prompt'] for x in tr}),sft_unique_dev_prompts=len({x['prompt'] for x in dv}),sft_rejected=pairrejected,packages=versions,eval_hashes=hashes,concept_blocklist=CONCEPTS,entry_blocklist=sorted(names),guard='Every accepted full 2049-byte window or full teacher prompt+body excludes all prompt/test 13grams. Corpus also excludes heldout definitions and high-specificity concept phrases. Teacher prompts additionally use original broad BLOCK_EVAL1/2. Semantic independence cannot be guaranteed.')
 json.dump(stats,open(OUT/'data_stats.json','w'),indent=2);print(json.dumps(stats,indent=2),flush=True)
 assert stats['train_bytes']>10000000 and len(tr)>10 and len(dv)>5
if __name__=='__main__':main()
