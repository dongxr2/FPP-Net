"""Frozen record/domain protocols, preparation, fitting, and held-out diagnostics."""
from pathlib import Path
import json,hashlib,time,zipfile
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits
from fpp_net import FPPNet,spectral_features,load_checkpoint
ROOT=Path(__file__).resolve().parent
threadpool_limits(1);torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.use_deterministic_algorithms(True)
SEEDS=[20261014,20261015,20261016]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def digest(p,kind='sha256'):
 h=hashlib.new(kind)
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def datasets(a):return ['kaist','ceitec'] if a.dataset=='all' else [a.dataset]
def load(ds,folder):
 rows=read(ROOT/f'data/manifests/{ds}_records.json');parts=[]
 for r in rows:
  p=folder/ds/f"record_{r['record']:02d}.npz"
  if not p.exists():raise FileNotFoundError(f'{p}: extract the processed-data release at repository root or run prepare first')
  with np.load(p,allow_pickle=False) as z:parts.append({k:z[k] for k in z.files})
 d={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
 assert len(d['label'])==sum(r['windows'] for r in rows)
 return d
def splits(ds,fold=None):return [s for s in read(ROOT/'configs/splits.json') if s['dataset']==ds and (fold is None or s['target']==fold)]
def metrics(y,p):
 pred=p>.5;cm=np.array([[np.sum((y==i)&(pred==j)) for j in [0,1]] for i in [0,1]])
 return dict(ba=float(np.mean(cm.diagonal()/cm.sum(1))),accuracy=float(np.mean(y==pred)),confusion=cm.tolist())
def scores(d,ids,p):
 rr=d['record'][ids];y=d['label'][ids];rows=[]
 for r in np.unique(rr):
  m=rr==r;rows.append(dict(record=int(r),label=int(y[m][0]),fault_probability=float(p[m].mean())))
 return dict(window=metrics(y,p),record=metrics(np.array([r['label'] for r in rows]),np.array([r['fault_probability'] for r in rows])),records=rows)
def scorekey(s,ds):return [s['record']['ba'],s['window']['ba']] if ds=='kaist' else [s['window']['ba'],s['record']['ba']]
def predict(model,Z,device,batch=256):
 model.eval()
 with torch.inference_mode():return np.concatenate([model(torch.from_numpy(np.array(Z[i:i+batch])).to(device)).softmax(1)[:,1].cpu().numpy() for i in range(0,len(Z),batch)])

def check(a):
 report={}
 for ds in datasets(a):
  d=load(ds,a.data_dir);rows=read(ROOT/f'data/manifests/{ds}_records.json')
  for r in rows:
   ids=np.flatnonzero(d['record']==r['record']);assert len(ids)==r['windows'] and ids[0]==r['offset']
   assert hashlib.sha256(d['features'][ids].tobytes()).hexdigest()==r['sha256_features']
   assert np.all(d['label'][ids]==r['label'])
  covered=[]
  for s in splits(ds):
   tr,te=np.array(s['train']),np.array(s['test']);assert not set(d['record'][tr])&set(d['record'][te]);covered.extend(te)
   for inner in s['inner']:
    x,y=np.array(inner['train']),np.array(inner['val'])
    assert not set(d['record'][x])&set(d['record'][y]);assert not set(x)&set(te) and not set(y)&set(te)
   if ds=='ceitec':
    mapping={r['record']:r['load_nm'] for r in rows};loads=lambda ids:{mapping[int(r)] for r in d['record'][ids]}
    assert not loads(tr)&loads(te)
    for i in s['inner']:assert not loads(i['train'])&loads(i['val']) and not loads(i['val'])&loads(te)
  assert sorted(covered)==list(range(len(d['label'])))
  report[ds]=dict(records=len(rows),windows=len(d['label']),healthy_windows=int(np.sum(d['label']==0)),feature_hashes_match=True,record_domain_disjoint=True)
 dump(a.out/'data_check.json',report);print(json.dumps(report,indent=2))

def evaluate(a):
 idx=read(a.index);result=[]
 for ds in datasets(a):
  d=load(ds,a.data_dir)
  for s in splits(ds,a.fold):
   ids=np.array(s['test']);jobs=[j for j in idx if j['dataset']==ds and j['fold']==s['target']];assert len(jobs)==3,'Evaluation requires all three seeds for a fold'
   ps=[];replay=[]
   for j in jobs:
    ck=ROOT/j['checkpoint'] if not Path(j['checkpoint']).is_absolute() else Path(j['checkpoint'])
    model=load_checkpoint(ck,device=a.device);p=predict(model,d['features'][ids],a.device);ps.append(p)
    if j.get('predictions'):
     with np.load(ROOT/j['predictions'],allow_pickle=False) as z:
      assert np.array_equal(z['ids'],ids);diff=float(np.max(np.abs(p-z['prob'][:,1])));assert diff<1e-4,diff;replay.append(diff)
   p=np.mean(ps,axis=0);m=scores(d,ids,p);result.append(dict(dataset=ds,fold=s['target'],**m,reference_replay_max_abs_error=max(replay) if replay else None))
   out=a.out/ds;out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/f"fold{s['target']}_predictions.npz",ids=ids,record=d['record'][ids],label=d['label'][ids],fault_probability=p)
 summary={ds:{level+'_BA_percent':100*float(np.mean([r[level]['ba'] for r in result if r['dataset']==ds])) for level in ['window','record']} for ds in datasets(a)}
 dump(a.out/'evaluation.json',dict(summary=summary,folds=result));print(json.dumps(summary,indent=2))

def fit(d,ds,tr,va,seed,epochs,out,device):
 out.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 rng=np.random.default_rng(seed);m=FPPNet(d['features'].shape[1]//2).to(device);opt=torch.optim.Adam(m.parameters(),lr=.001)
 sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,120,eta_min=1e-5)
 Z=torch.from_numpy(d['features']).to(device);Y=torch.from_numpy(d['label']).long().to(device)
 groups={int(r):tr[d['record'][tr]==r] for r in np.unique(d['record'][tr])};byclass=[np.array([r for r,ids in groups.items() if d['label'][ids[0]]==c]) for c in [0,1]]
 hist=[];h=hashlib.sha256()
 for epoch in range(1,epochs+1):
  m.train();losses=[]
  for _ in range(16):
   ids=np.array([rng.choice(groups[int(rng.choice(byclass[c]))]) for c in rng.integers(0,2,96)]);h.update(ids.tobytes())
   opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(m(Z[ids]),Y[ids]);assert torch.isfinite(loss);loss.backward();opt.step();losses.append(float(loss.detach()))
  sched.step()
  if epoch%10==0 or epoch==epochs:
   item=dict(epoch=epoch,loss=float(np.mean(losses)))
   if va is not None:
    item['score']=scorekey(scores(d,va,predict(m,d['features'][va],device)),ds)
    torch.save(dict(state_dict=m.state_dict(),epoch=epoch),out/f'epoch{epoch}.pt')
   hist.append(item)
 torch.save(dict(state_dict=m.state_dict(),epoch=epochs),out/'final.pt');dump(out/'history.json',hist)
 dump(out/'training.json',dict(seed=seed,epochs=epochs,schedule_horizon=120,batch_hash=h.hexdigest(),train_records=np.unique(d['record'][tr]).tolist(),validation_records=[] if va is None else np.unique(d['record'][va]).tolist()))
 del Z,Y,m,opt
 if torch.cuda.is_available():torch.cuda.empty_cache()
 return hist

def train(a):
 index=[]
 for ds in datasets(a):
  d=load(ds,a.data_dir)
  for s in splits(ds,a.fold):
   for seed in SEEDS if a.seed is None else [a.seed]:
    out=a.out/ds/f"fold{s['target']}"/f'seed{seed}';hist=[]
    for i,inner in enumerate(s['inner']):hist.append(fit(d,ds,np.array(inner['train']),np.array(inner['val']),seed,a.epochs,out/f'inner{i}',a.device))
    avg=np.mean([[r['score'] for r in h] for h in hist],axis=0);best=max(range(len(avg)),key=lambda i:tuple(avg[i]));epoch=hist[0][best]['epoch']
    if ds=='kaist':
     fit(d,ds,np.array(s['train']),None,seed,epoch,out/'refit',a.device);ck=out/'refit/final.pt'
    else:ck=out/f'inner0/epoch{epoch}.pt'
    row=dict(dataset=ds,fold=s['target'],seed=seed,epoch=epoch,checkpoint=str(ck.resolve()),paper_epoch_budget=a.epochs==120,validation_score=avg[best].tolist());index.append(row)
    dump(out/'selected.json',row);dump(a.out/'index.json',index);print('TRAINED',ds,s['target'],seed,epoch,flush=True)

def baseline(a):
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import LogisticRegression,RidgeClassifier
 from sklearn.svm import SVC
 from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
 import joblib
 def model(param):
  if a.method=='lr':return make_pipeline(StandardScaler(),LogisticRegression(C=param,max_iter=4000,random_state=20261014))
  if a.method=='svm':return make_pipeline(StandardScaler(),SVC(C=param))
  if a.method=='ridge':return make_pipeline(StandardScaler(),RidgeClassifier(alpha=param,solver='lsqr',tol=.001,max_iter=1000))
  if a.method=='rf':return RandomForestClassifier(n_estimators=200,max_depth=param,min_samples_leaf=2,n_jobs=1,random_state=20261014)
  if a.method=='gb':return GradientBoostingClassifier(n_estimators=200,learning_rate=.05,max_depth=param,min_samples_leaf=10,random_state=20261014)
  from lightgbm import LGBMClassifier
  return LGBMClassifier(n_estimators=200,learning_rate=.05,num_leaves=param,min_child_samples=20,n_jobs=1,random_state=20261014,deterministic=True,force_col_wise=True,verbosity=-1)
 def prob(m,x):
  if hasattr(m,'predict_proba'):return m.predict_proba(x)[:,1]
  return 1/(1+np.exp(-np.clip(m.decision_function(x),-40,40)))
 result=[]
 for ds in datasets(a):
  d=load(ds,a.data_dir);X=d['features'].reshape(len(d['label']),-1,64,16).mean(-1).reshape(len(d['label']),-1)
  def fitted(ids,param):
   y=d['label'][ids];rr=d['record'][ids];w=np.empty(len(ids))
   for c in [0,1]:
    rs=np.unique(rr[y==c])
    for r in rs:
     w[rr==r]=(1 if a.method in ['lr','svm','rf'] else len(ids))/(2*len(rs)*np.sum(rr==r))
   if a.method in ['lr','svm','rf']:w*=len(ids)
   m=model(param);kw={'sample_weight':w}
   if a.method in ['lr','svm','ridge']:
    key={'lr':'logisticregression','svm':'svc','ridge':'ridgeclassifier'}[a.method];kw={key+'__sample_weight':w}
    if a.method=='ridge':kw['standardscaler__sample_weight']=w
   m.fit(X[ids],y,**kw);return m
  grid={'lr':[.1,1,10,100],'svm':[.1,1,10,100],'rf':[5,None],'ridge':[.1,1,10,100],'gb':[1,3],'lightgbm':[7,15]}[a.method]
  for s in splits(ds,a.fold):
   trials=[]
   for p in grid:
    vals=[]
    for i in s['inner']:
     m=fitted(np.array(i['train']),p);ids=np.array(i['val']);vals.append(scorekey(scores(d,ids,prob(m,X[ids])),ds))
    trials.append(dict(parameter=p,score=np.mean(vals,axis=0).tolist()))
   best=max(trials,key=lambda t:tuple(t['score']));m=fitted(np.array(s['train']),best['parameter']);ids=np.array(s['test']);r=scores(d,ids,prob(m,X[ids]));result.append(dict(dataset=ds,fold=s['target'],method=a.method,selection=best,trials=trials,**r))
   out=a.out/ds;out.mkdir(parents=True,exist_ok=True);joblib.dump(m,out/f"{a.method}_fold{s['target']}.joblib")
 summary={ds:{k+'_BA_percent':100*float(np.mean([r[k]['ba'] for r in result if r['dataset']==ds])) for k in ['window','record']} for ds in datasets(a)}
 dump(a.out/f'{a.method}_results.json',dict(summary=summary,folds=result));print(json.dumps(summary,indent=2))

def explain(a):
 index=read(a.index);out={}
 for ds in datasets(a):
  d=load(ds,a.data_dir);profiles=[];folds=[]
  for s in splits(ds,a.fold):
   ids=np.array(s['test']);jobs=[j for j in index if j['dataset']==ds and j['fold']==s['target']];assert len(jobs)==3
   models=[load_checkpoint(ROOT/j['checkpoint'],a.device) for j in jobs]
   for m in models:
    w=m.head[0].weight.detach().cpu().numpy().reshape(16,16,64);w=np.sqrt((w*w).sum((0,1)));profiles.append(w/w.sum())
   changes=np.empty((64,len(ids)),np.float32)
   with torch.inference_mode():
    for start in range(0,len(ids),256):
     z=torch.from_numpy(d['features'][ids[start:start+256]]).to(a.device);base=torch.stack([m(z).softmax(1)[:,1] for m in models]).mean(0)
     for band in range(64):
      zz=z.clone();zz[:,:,band*16:(band+1)*16]=0;p=torch.stack([m(zz).softmax(1)[:,1] for m in models]).mean(0);changes[band,start:start+len(z)]=(p-base).abs().cpu().numpy()
   folds.append(changes.mean(1).tolist());a.out.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out/f'{ds}_fold{s["target"]}_occlusion.npz',ids=ids,absolute_probability_change=changes)
  w=np.array(profiles).mean(0);occ=np.mean(folds,axis=0);out[ds]=dict(weight_profiles=np.array(profiles).tolist(),occlusion_fold_profiles=folds,half_positions=int(np.searchsorted(np.cumsum(np.sort(w)[::-1]),.5)+1),pearson_r=float(np.corrcoef(w,occ)[0,1]))
 dump(a.out/'frequency_position.json',out);print(json.dumps({ds:{k:v for k,v in x.items() if k in ['half_positions','pearson_r']} for ds,x in out.items()},indent=2))

def prepare(a):
 from scipy.io import loadmat
 for ds in datasets(a):
  rows=read(ROOT/f'data/manifests/{ds}_records.json');files={p.name:p for p in a.raw_dir.rglob('*') if p.is_file()}
  for r in rows:
   p=files.get(r['filename'])
   if p is None:raise FileNotFoundError(f"Missing {r['filename']} under {a.raw_dir}")
   assert digest(p)==r['sha256_raw_file'],f'Raw file checksum mismatch: {p}'
   if ds=='kaist':
    from nptdms import TdmsFile
    td=TdmsFile.read(p);chs=[c for g in td.groups() for c in g.channels()];assert len(chs)==1 and chs[0].path==r['channel']
    x=chs[0][:][:3072000].reshape(600,1,5120);starts=np.arange(600)*5120
   else:
    m=loadmat(p,variable_names=['Time','Speed_rpm']+['Currents_SubSys1_'+c for c in 'ABC']);assert np.allclose(np.diff(m['Time'].ravel()),.0001,rtol=1e-5,atol=1e-10)
    v=m['Speed_rpm'].ravel();current=np.stack([m['Currents_SubSys1_'+c].ravel() for c in 'ABC'])
    cand=[i for i in range(0,len(v)-1710+1,1710) if v[i:i+1710].min()>=500 and v[i:i+1710].max()<=4000]
    starts=np.array(cand)[np.linspace(0,len(cand)-1,min(160,len(cand)),dtype=int)];x=np.stack([current[:,i:i+1710] for i in starts]).astype(np.float32)
   Z=np.concatenate([spectral_features(x[i:i+128],r['sample_rate']) for i in range(0,len(x),128)])
   assert len(Z)==r['windows'];assert hashlib.sha256(Z.tobytes()).hexdigest()==r['sha256_features'],'Preprocessing differs from frozen release; check NumPy/SciPy versions'
   out=a.data_dir/ds;out.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(out/f"record_{r['record']:02d}.npz",features=Z,label=np.full(len(Z),r['label'],np.int64),record=np.full(len(Z),r['record'],np.int64),domain=np.full(len(Z),r['domain'],np.int64),start=starts,end=starts+r['window_samples'])
   print('PREPARED',ds,r['record'],len(Z),flush=True)

def download(a):
 import requests
 for ds in datasets(a):
  folder=a.raw_dir/ds;folder.mkdir(parents=True,exist_ok=True)
  for row in read(ROOT/f'data/sources/{ds}_downloads.json'):
   dest=folder/row['filename'];checksum=row.get('checksum');sha=row.get('sha256')
   def valid():return dest.exists() and dest.stat().st_size==row['size'] and (not checksum or digest(dest,checksum.split(':')[0])==checksum.split(':')[1]) and (not sha or digest(dest)==sha)
   if not valid():
    part=dest.with_suffix(dest.suffix+'.part')
    with requests.get(row['url'],stream=True,timeout=(30,120)) as response:
     response.raise_for_status()
     with part.open('wb') as f:
      for chunk in response.iter_content(1024*1024):f.write(chunk)
    part.replace(dest);assert valid(),f'Checksum/size failed: {dest}'
   if dest.suffix=='.zip':
    with zipfile.ZipFile(dest) as z:
     for entry in z.infolist():
      # Extract vibration files only; protect against archive traversal.
      if not entry.filename.lower().endswith('.tdms') or 'vibration' not in entry.filename.lower():continue
      target=(folder/entry.filename).resolve();assert target.is_relative_to(folder.resolve());z.extract(entry,folder)
   print('DOWNLOADED',ds,dest.name,flush=True)
