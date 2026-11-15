import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'packages'))
import cv2,numpy as np,json,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from specialist import peaks

def extra(im):
 a=im.astype(np.float32);fs=[]
 def add(v):fs.append(v[::4,::4])
 for dx,dy in [(1,0),(0,1),(1,1)]:
  dif=np.abs(a-np.roll(a,(dy,dx),(0,1))).max(2)
  for t in [.5,1.5,2.5,4.5]:
   eq=(dif<t).astype(np.float32)
   for s in [7,19,39]:add(cv2.boxFilter(eq,-1,(s,s)))
 gray=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
 residual=np.abs(gray-cv2.GaussianBlur(gray,(0,0),.7))
 unclipped=((a.min(2)>5)&(a.max(2)<250)).astype(np.float32)
 for t in [.1,.3,.7,1.5]:
  low=(residual<t).astype(np.float32)
  for s in [9,21,45]:
   add(cv2.boxFilter(low,-1,(s,s)))
   add(cv2.boxFilter(low*unclipped,-1,(s,s)))
 for s in [3,5,9]:
  minimum=cv2.erode(residual,np.ones((s,s),np.uint8))
  for w in [9,21,45]:add(cv2.boxFilter(minimum,-1,(w,w)))
 grad=np.maximum(np.abs(cv2.Sobel(gray,cv2.CV_32F,1,0)),np.abs(cv2.Sobel(gray,cv2.CV_32F,0,1)))
 for t in [3,8,20]:
  flat=cv2.erode((grad<t).astype(np.float32),np.ones((5,5),np.uint8))
  for w in [21,45]:
   count=cv2.boxFilter(flat,-1,(w,w));noise=cv2.boxFilter(flat*residual**2,-1,(w,w))/(count+.0001)
   add(count);add(np.log1p(noise))
 return np.stack(fs,2)

def main():
 root=Path(__file__).parent;out=root/'plateau';out.mkdir(exist_ok=True)
 base=joblib.load(root/'specialist/cache.joblib');rng=np.random.default_rng(2026)
 cache=out/'cache.joblib'
 if cache.exists():data=joblib.load(cache)
 else:
  data=[]
  for name,f,m,v,k in base:
   e=extra(cv2.imread(str(root/'task_kazakhstan/exhibits'/(name+'.jpg'))))
   data.append((name,np.concatenate([f,e],2),m,v,k))
  joblib.dump(data,cache)
 xs=[];ys=[];gs=[]
 for i,(name,f,m,v,kind) in enumerate(data):
  pos=np.flatnonzero(m.ravel());neg=np.flatnonzero((v&~m).ravel());ids=np.r_[pos,rng.choice(neg,min(3000,len(neg)),replace=False)]
  xs.append(f.reshape(-1,f.shape[-1])[ids]);ys.append(m.ravel()[ids]);gs.extend([i]*len(ids))
 X=np.concatenate(xs);Y=np.concatenate(ys);G=np.array(gs);print('train',X.shape,flush=True)
 def model():return HistGradientBoostingClassifier(max_iter=220,max_leaf_nodes=23,learning_rate=.07,l2_regularization=5,random_state=2026)
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
