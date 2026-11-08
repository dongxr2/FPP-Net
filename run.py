"""Portable command-line interface. All numerical jobs run sequentially."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
from pathlib import Path
from workflow import evaluate,train,prepare,baseline,explain,download,check

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('command',choices=['check','download','prepare','evaluate','train','baseline','explain'])
 p.add_argument('--dataset',choices=['kaist','ceitec','all'],default='all')
 p.add_argument('--data-dir',type=Path,default=Path('data/processed'))
 p.add_argument('--raw-dir',type=Path,default=Path('data/raw'))
 p.add_argument('--out',type=Path,default=Path('outputs'))
 p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
 p.add_argument('--fold',type=int,choices=[0,1,2])
 p.add_argument('--seed',type=int,choices=[20261014,20261015,20261016])
 p.add_argument('--epochs',type=int,default=120,help='120 is the paper protocol; other values are development runs')
 p.add_argument('--index',type=Path,default=Path('checkpoints/index.json'))
 p.add_argument('--method',choices=['lr','svm','rf','ridge','gb','lightgbm'],default='lr')
 a=p.parse_args();assert a.epochs>0
 globals()[a.command](a)
if __name__=='__main__':main()
