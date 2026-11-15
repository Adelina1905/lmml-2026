import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'packages'))
import cv2,numpy as np,json,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from specialist import peaks

def noise_features(noise,prob,conf):
 n=noise.astype(np.float32);fs=[]
 def add(a):fs.append(a[::4,::4])
 scale=np.std(n)+1e-5;n=n/scale
 for sigma in [0,1,3,8]:
  b=n if sigma==0 else n-cv2.GaussianBlur(n,(0,0),sigma)
  for s in [5,13,29,61]:
   mean=cv2.boxFilter(b,-1,(s,s));var=np.maximum(cv2.boxFilter(b*b,-1,(s,s))-mean*mean,0)
   add(mean);add(np.log1p(var))
   add(cv2.boxFilter(np.abs(b),-1,(s,s)))
  for dx,dy in [(1,0),(0,1),(1,1),(2,0),(0,2),(8,0),(0,8)]:
   b2=np.roll(b,(dy,dx),(0,1))
   for s in [13,29]:
    add(cv2.boxFilter(b*b2,-1,(s,s))/(cv2.boxFilter(b*b,-1,(s,s))+.01))
 for q in [prob,conf]:
  for s in [5,13,29,61]:add(cv2.boxFilter(q,-1,(s,s)))
 return np.stack(fs,2)

def main():
 root=Path(__file__).parent;out=root/'noiselearn';out.mkdir(exist_ok=True)
 base=joblib.load(root/'plateau/cache.joblib');rng=np.random.default_rng(2026);data=[]
 for name,f,m,v,k in base:
  z=np.load(root/'trufor_results/exhibits'/(name+'.npz'))
  e=noise_features(z['noise'],z['prob'].astype(np.float32),z['conf'].astype(np.float32))
  data.append((name,np.concatenate([f,e],2),m,v,k))
 xs=[];ys=[];gs=[]
 for i,(name,f,m,v,kind) in enumerate(data):
  pos=np.flatnonzero(m.ravel());neg=np.flatnonzero((v&~m).ravel());ids=np.r_[pos,rng.choice(neg,min(3000,len(neg)),replace=False)]
  xs.append(f.reshape(-1,f.shape[-1])[ids]);ys.append(m.ravel()[ids]);gs.extend([i]*len(ids))
 X=np.concatenate(xs);Y=np.concatenate(ys);G=np.array(gs);print('train',X.shape,flush=True)
 def model():return HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=23,learning_rate=.07,l2_regularization=8,random_state=2026)
 hits=np.zeros(4,int);records=[]
 for fold,(tr,te) in enumerate(GroupKFold(3).split(X,Y,G)):
  clf=model().fit(X[tr],Y[tr])
  for i in np.unique(G[te]):
   name,f,m,v,k=data[i];pr=clf.predict_proba(f.reshape(-1,f.shape[-1]))[:,1].reshape(f.shape[:2]);pp=peaks(pr,v);rec=dict(exhibit=name,predictions=[])
   for j,(x,y,s) in enumerate(pp):
    hit=bool(m[y//4,x//4]);hits[j]+=hit;rec['predictions'].append(dict(x=x,y=y,score=s,hit=hit))
   records.append(rec);np.savez_compressed(out/(name+'_oof.npz'),prob=pr.astype(np.float16),mask=m,valid=v)
  print('fold',fold,hits.tolist(),flush=True)
 (out/'validation.json').write_text(json.dumps(dict(hits=hits.tolist(),predictions=records),indent=2))
 clf=model().fit(X,Y);joblib.dump(clf,out/'model.joblib');print('saved',flush=True)

if __name__=='__main__':main()
