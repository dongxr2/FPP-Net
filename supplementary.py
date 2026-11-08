"""Reproduce the supplementary LR acquisition-group/split checks, sequentially."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import json,argparse
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
threadpool_limits(1)
ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf-8')


def ba(y,p):return float(np.mean([np.mean((p[y==k]>.5)==k) for k in (0,1)]))

def summarize(y,rec,p):
    rr=np.unique(rec);q=np.array([p[rec==r].mean() for r in rr]);yr=np.array([y[rec==r][0] for r in rr])
    return dict(window_BA=ba(y,p)*100,record_BA=ba(yr,q)*100,records=[dict(record=int(r),y=int(t),score=float(v)) for r,t,v in zip(rr,yr,q)])

def fit(X,y,rec,C):
    w=np.empty(len(y))
    for k in [0,1]:
        rr=np.unique(rec[y==k])
        for r in rr:w[rec==r]=len(y)/(2*len(rr)*np.sum(rec==r))
    sc=StandardScaler().fit(X)
    lr=LogisticRegression(C=C,max_iter=4000,random_state=20261014).fit(sc.transform(X),y,sample_weight=w)
    return sc,lr

def predict(model,X):return model[1].predict_proba(model[0].transform(X))[:,1]

def shortcuts():
    d=np.load(FEATURES);X,y,rec,motor,grp=[d[k] for k in ['X','y','rec','motor','group']]
    manifest={r['record']:r for r in read(ROOT/'data/manifests/kaist_all45.json')};res={}
    grid=[.1,1,10,100];rng=np.random.default_rng(20261014)
    te=np.zeros(len(y),bool)
    for k in (0,1):
        ids=np.flatnonzero(y==k);te[rng.choice(ids,round(.2*len(ids)),replace=False)]=True
    va=np.zeros(len(y),bool);tr=np.flatnonzero(~te);va[rng.choice(tr,round(.2*len(tr)),replace=False)]=True
    trials=[dict(C=C,score=ba(y[va],predict(fit(X[~te&~va],y[~te&~va],rec[~te&~va],C),X[va]))) for C in grid]
    C=max(trials,key=lambda z:z['score'])['C'];model=fit(X[~te],y[~te],rec[~te],C);p=predict(model,X[te])
    res['E1']=dict(window_BA=100*ba(y[te],p),C=C,trials=trials,train_records=len(np.unique(rec[~te])),test_records=len(np.unique(rec[te])),overlap_records=len(np.intersect1d(rec[te],rec[~te])))
    np.savez(OUT/'E1_predictions.npz',ids=np.flatnonzero(te),record=rec[te],y=y[te],prob=p,fit_ids=np.flatnonzero(~te&~va),val_ids=np.flatnonzero(va),train_ids=np.flatnonzero(~te));joblib.dump(model,OUT/'E1_model.joblib')
    print('E1',res['E1'],flush=True)
    for name,mask,labels in [('E2',np.ones(len(y),bool),y),('E3',grp!='C',(grp=='B').astype(int)),('E4',grp=='A',y)]:
        folds=[]
        for target in range(3):
            tr=mask&(motor!=target);te=mask&(motor==target);trials=[]
            for C in grid:
                scores=[]
                for val in range(3):
                    if val==target:continue
                    a=tr&(motor!=val);b=tr&(motor==val)
                    m=fit(X[a],labels[a],rec[a],C);s=summarize(labels[b],rec[b],predict(m,X[b]));scores.append([s['record_BA'],s['window_BA']])
                trials.append(dict(C=C,score=np.mean(scores,axis=0).tolist()))
            C=max(trials,key=lambda z:tuple(z['score']))['C'];model=fit(X[tr],labels[tr],rec[tr],C);p=predict(model,X[te]);s=summarize(labels[te],rec[te],p)
            for rr in s['records']:rr.update({k:manifest[rr['record']][k] for k in ['group','fault_type','motor_kw']})
            s.update(target=target,C=C,trials=trials,train_records=np.unique(rec[tr]).tolist(),test_records=np.unique(rec[te]).tolist());assert not set(s['train_records'])&set(s['test_records'])
            folds.append(s);np.savez(OUT/f'{name}_target{target}_predictions.npz',ids=np.flatnonzero(te),record=rec[te],y=labels[te],prob=p);joblib.dump(model,OUT/f'{name}_target{target}_model.joblib')
            print(name,target,s['window_BA'],s['record_BA'],flush=True)
        res[name]=dict(window_BA=float(np.mean([f['window_BA'] for f in folds])),record_BA=float(np.mean([f['record_BA'] for f in folds])),folds=folds)
        dump(OUT/'shortcut_results.json',res)
    rows=[r for f in res['E2']['folds'] for r in f['records']]
    res['E2']['detected']={g:sum(r['y']==1 and r['group']==g and r['score']>.5 for r in rows) for g in 'ABC'}
    res['E2']['healthy_false_alarms']=sum(r['y']==0 and r['score']>.5 for r in rows)
    dump(OUT/'shortcut_results.json',res)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,default=Path('data/processed/kaist_all45_lr.npz'))
    p.add_argument('--out',type=Path,default=Path('outputs/shortcut'))
    a=p.parse_args();FEATURES=a.features;OUT=a.out;OUT.mkdir(parents=True,exist_ok=True);shortcuts()
