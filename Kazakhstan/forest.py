import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'packages'))
import cv2,numpy as np,json,joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from specialist import peaks

def context(f):
 # Neighboring noise estimates encode abrupt local changes in texture.
 ids=[3,9,10,24,30,45,66,73,80,87,94,101,108,118,120,122,124,126,128,130,132,134]
 a=f[:,:,ids];out=[f]
 for dx,dy in [(-4,0),(4,0),(0,-4),(0,4),(-8,0),(8,0),(0,-8),(0,8)]:
  out.append(np.roll(a,(dy,dx),(0,1)))
 return np.concatenate(out,axis=2)

def main():
 root=Path(__file__).parent;out=root/'forest';out.mkdir(exist_ok=True)
 data=joblib.load(root/'specialist/cache.joblib');rng=np.random.default_rng(2026)
 fs=[];xs=[];ys=[];gs=[]
 for i,(name,f,m,v,kind) in enumerate(data):
  f=context(f);fs.append(f)
  pos=np.flatnonzero(m.ravel());neg=np.flatnonzero((v&~m).ravel());ids=np.r_[pos,rng.choice(neg,min(2000,len(neg)),replace=False)]
  xs.append(f.reshape(-1,f.shape[-1])[ids]);ys.append(m.ravel()[ids]);gs.extend([i]*len(ids))
 X=np.concatenate(xs);Y=np.concatenate(ys);G=np.array(gs)
 print('train',X.shape,flush=True)
 def model():return ExtraTreesClassifier(n_estimators=80,max_depth=22,min_samples_leaf=5,max_features='sqrt',class_weight='balanced',n_jobs=3,random_state=2026)
 hits=np.zeros(4,int);records=[]
 for fold,(tr,te) in enumerate(GroupKFold(3).split(X,Y,G)):
  clf=model().fit(X[tr],Y[tr])
  for i in np.unique(G[te]):
   name,_,m,v,kind=data[i];f=fs[i];pr=clf.predict_proba(f.reshape(-1,f.shape[-1]))[:,1].reshape(f.shape[:2]);pp=peaks(pr,v)
   rec=dict(exhibit=name,predictions=[])
   for j,(x,y,s) in enumerate(pp):
    hit=bool(m[y//4,x//4]);hits[j]+=hit;rec['predictions'].append(dict(x=x,y=y,score=s,hit=hit))
   records.append(rec);np.savez_compressed(out/(name+'_oof.npz'),prob=pr.astype(np.float16),mask=m,valid=v)
  print('fold',fold,hits.tolist(),flush=True)
 (out/'validation.json').write_text(json.dumps(dict(hits=hits.tolist(),predictions=records),indent=2))
 clf=model().fit(X,Y);joblib.dump(clf,out/'model.joblib');print('saved',flush=True)

if __name__=='__main__':main()
