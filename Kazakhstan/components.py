import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'packages'))
import cv2,numpy as np,json,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from specialist import peaks

def extra(im):
 g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
 residual=np.abs(g-cv2.GaussianBlur(g,(0,0),.7));envelope=cv2.boxFilter(residual,-1,(5,5))
 fs=[]
 for t in [.1,.2,.35,.5,.7,1,1.5,2]:
  m=(envelope<t).astype(np.uint8)
  m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
  n,lab,stats,centers=cv2.connectedComponentsWithStats(m)
  vals=np.zeros((n,6),np.float32)
  for k,(x,y,w,h,area) in enumerate(stats[1:],1):
   if w<4 or h<4:continue
   vals[k]=[np.log1p(w),np.log1p(h),area/(w*h),np.log1p(area),w/h,min(w,h)/max(w,h)]
  fs.extend([vals[lab[::4,::4],k] for k in range(6)])
 return np.stack(fs,2)

def main():
 root=Path(__file__).parent;out=root/'components';out.mkdir(exist_ok=True);base=joblib.load(root/'plateau/cache.joblib');rng=np.random.default_rng(2026);data=[]
 for name,f,m,v,k in base:
  e=extra(cv2.imread(str(root/'task_kazakhstan/exhibits'/(name+'.jpg'))));data.append((name,np.concatenate([f,e],2),m,v,k))
 xs=[];ys=[];gs=[]
 for i,(name,f,m,v,kind) in enumerate(data):
  pos=np.flatnonzero(m.ravel());neg=np.flatnonzero((v&~m).ravel());ids=np.r_[pos,rng.choice(neg,min(3000,len(neg)),replace=False)]
  xs.append(f.reshape(-1,f.shape[-1])[ids]);ys.append(m.ravel()[ids]);gs.extend([i]*len(ids))
 X=np.concatenate(xs);Y=np.concatenate(ys);G=np.array(gs);print('train',X.shape,flush=True)
 def model():return HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=23,learning_rate=.07,l2_regularization=5,random_state=2026)
 hits=np.zeros(4,int);records=[]
 for fold,(tr,te) in enumerate(GroupKFold(3).split(X,Y,G)):
  clf=model().fit(X[tr],Y[tr])
  for i in np.unique(G[te]):
   name,f,m,v,k=data[i];pr=clf.predict_proba(f.reshape(-1,f.shape[-1]))[:,1].reshape(f.shape[:2]);pp=peaks(pr,v);rec=dict(exhibit=name,predictions=[])
   for j,(x,y,s) in enumerate(pp):
    hit=bool(m[y//4,x//4]);hits[j]+=hit;rec['predictions'].append(dict(x=x,y=y,score=s,hit=hit))
   records.append(rec);np.savez_compressed(out/(name+'_oof.npz'),prob=pr.astype(np.float16),mask=m,valid=v)
  print('fold',fold,hits.tolist(),flush=True)
 (out/'validation.json').write_text(json.dumps(dict(hits=hits.tolist(),predictions=records),indent=2));clf=model().fit(X,Y);joblib.dump(clf,out/'model.joblib');print('saved',flush=True)

if __name__=='__main__':main()
