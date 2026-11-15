import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'packages'))
import cv2,numpy as np,json,joblib,argparse,csv
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from detect import features,valid_area,stamps_on,STRIDE

def peaks(prob,valid):
 smooth=cv2.boxFilter(prob.astype(np.float32),-1,(7,7))
 ans=[]
 for outer in [0,15,25,41]:
  score=smooth.copy()
  if outer:score-=cv2.boxFilter(prob.astype(np.float32),-1,(outer,outer))
  score[~valid]=-10;y,x=np.unravel_index(score.argmax(),score.shape)
  ans.append((int(x*4),int(y*4),float(score[y,x])))
 return ans

def main():
 root=Path(__file__).parent;out=root/'specialist';out.mkdir(exist_ok=True);cache=out/'cache.joblib'
 if cache.exists():data=joblib.load(cache)
 else:
  data=[]
  for p in sorted((root/'task_kazakhstan/exhibits').glob('*.jpg')):
   im=cv2.imread(str(p));m=cv2.imread(str(p.with_name(p.stem+'_mask.png')),0);f,_=features(im)
   x,y,w,h=cv2.boundingRect(m);kind=1 if w>h*1.1 else 2
   data.append((p.stem,f,(m>0)[::4,::4],valid_area(im),kind))
  joblib.dump(data,cache,compress=0)
 rng=np.random.default_rng(2026);xs=[];ys=[];gs=[]
 for i,(name,f,m,v,kind) in enumerate(data):
  pos=np.flatnonzero(m.ravel());neg=np.flatnonzero((v&~m).ravel());ids=np.r_[pos,rng.choice(neg,min(3000,len(neg)),replace=False)]
  xs.append(f.reshape(-1,f.shape[-1])[ids]);ys.append(m.ravel()[ids]*kind);gs.extend([i]*len(ids))
 X=np.concatenate(xs);Y=np.concatenate(ys);G=np.array(gs)
 print('train',X.shape,np.bincount(Y),flush=True)
 def model():return HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,learning_rate=.07,l2_regularization=8,random_state=2026)
 hits=np.zeros((3,4),int);records=[]
 for fold,(tr,te) in enumerate(GroupKFold(3).split(X,Y,G)):
  # Multiclass makes the two inconsistent alteration signatures distinct.
  clf=model().fit(X[tr],Y[tr])
  for i in np.unique(G[te]):
   name,f,m,v,kind=data[i];pr=clf.predict_proba(f.reshape(-1,f.shape[-1])).reshape(*f.shape[:2],3)
   records.append(dict(exhibit=name,kind=kind,predictions=[]))
   for k,prob in enumerate([pr[:,:,1],pr[:,:,2],1-pr[:,:,0]]):
    pp=peaks(prob,v)
    for j,(x,y,s) in enumerate(pp):
     hit=bool(m[y//4,x//4]);hits[k,j]+=hit
     records[-1]['predictions'].append(dict(type=k,contrast=j,x=x,y=y,score=s,hit=hit))
   np.savez_compressed(out/(name+'_oof.npz'),prob=pr.astype(np.float16),mask=m,valid=v)
  print('fold',fold,hits.tolist(),flush=True)
 (out/'validation.json').write_text(json.dumps(dict(hits=hits.tolist(),predictions=records),indent=2))
 clf=model().fit(X,Y);joblib.dump(clf,out/'model.joblib')
 print('saved',flush=True)

if __name__=='__main__':main()
