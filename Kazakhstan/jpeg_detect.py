"""Detect violations of the first JPEG quantization lattice on its recovered grid."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'packages'))
import cv2,numpy as np,json,csv,joblib,argparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from jpeg_probe import estimate,qtable,dct_blocks
from detect import stamps_on

def feat(im,quality,px,py):
 g=cv2.cvtColor(im.astype(np.float32),cv2.COLOR_BGR2YCrCb)[:,:,0]
 c=dct_blocks(g,px,py);h,w=c.shape[:2];q=qtable(quality)
 ratio=c/q;err=np.abs(ratio-np.rint(ratio));fs=[]
 def add(a):fs.append(a)
 def aggregates(a):
  for s in [1,3,5,9]:add(cv2.boxFilter(a.astype(np.float32),-1,(s,s)))
 for i in range(8):
  for j in range(8):
   add(err[:,:,i,j]);add(np.log1p(np.abs(ratio[:,:,i,j])))
 freq=np.add.outer(np.arange(8),np.arange(8))
 for sel in [(freq>0)&(freq<=3),(freq>3)&(freq<=7),freq>7,freq>0]:
  e=err[:,:,sel];r=ratio[:,:,sel];active=np.abs(r)>.75
  aggregates(e.mean(2));aggregates((e*active).sum(2)/(active.sum(2)+.001));aggregates(active.mean(2))
 aggregates(err[:,:,0,0]);aggregates(c[:,:,0,0]/8+128)
 # Mean lattice violation relative to nearby blocks suppresses stamp edges.
 violation=(err*q).reshape(h,w,-1)[:,:,1:].mean(2)
 for s in [1,3,5]:add(cv2.boxFilter(violation,-1,(s,s))-cv2.boxFilter(violation,-1,(11,11)))
 gray=g[py:py+8*h,px:px+8*w].reshape(h,8,w,8).transpose(0,2,1,3)
 for a in [gray.std((2,3)),(gray>252).mean((2,3)),(gray<3).mean((2,3))]:aggregates(a)
 add(np.full((h,w),quality,np.float32))
 return np.stack(fs,2).astype(np.float32)

def bestpoint(prob,valid,px,py,size=3):
 sm=cv2.boxFilter(prob.astype(np.float32),-1,(size,size));sm[~valid]=-1
 y,x=np.unravel_index(sm.argmax(),sm.shape);return int(x*8+px+4),int(y*8+py+4),float(sm[y,x])

def train(root,out):
 rank={d['exhibit']:d['rank'][0] for d in json.loads((root/'jpeg_probe.json').read_text())};data=[];xs=[];ys=[];gs=[]
 for i,p in enumerate(sorted((root/'task_kazakhstan/exhibits').glob('*.jpg'))):
  im=cv2.imread(str(p));_,q,px,py=rank[p.stem];f=feat(im,q,px,py);h,w=f.shape[:2]
  m=cv2.imread(str(p.with_name(p.stem+'_mask.png')),0)>0
  target=m[py+4:py+4+h*8:8,px+4:px+4+w*8:8];valid=np.zeros((h,w),bool)
  for x,y,bw,bh in stamps_on(im):
   yy,xx=np.mgrid[:h,:w];xx=xx*8+px+4;yy=yy*8+py+4
   valid|=(xx>x+18)&(xx<x+bw-18)&(yy>y+18)&(yy<y+bh-18)
  data.append((p.stem,f,target,valid,m,px,py))
  ids=np.flatnonzero(valid.ravel()|target.ravel());xs.append(f.reshape(-1,f.shape[-1])[ids]);ys.append(target.ravel()[ids]);gs.extend([i]*len(ids))
 X=np.concatenate(xs);Y=np.concatenate(ys);G=np.array(gs);print('train',X.shape,int(Y.sum()),flush=True)
 def model():return HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,learning_rate=.08,l2_regularization=3,random_state=2026)
 hits=np.zeros(3,int);records=[];aucs=[]
 for fold,(tr,te) in enumerate(GroupKFold(3).split(X,Y,G)):
  clf=model().fit(X[tr],Y[tr]);auc=roc_auc_score(Y[te],clf.predict_proba(X[te])[:,1]);aucs.append(float(auc))
  for i in np.unique(G[te]):
   name,f,tar,v,m,px,py=data[i];pr=clf.predict_proba(f.reshape(-1,f.shape[-1]))[:,1].reshape(f.shape[:2]);rec=dict(exhibit=name,predictions=[])
   for j,size in enumerate([1,2,3]):
    x,y,s=bestpoint(pr,v,px,py,size);hit=bool(m[y,x]);hits[j]+=hit;rec['predictions'].append(dict(x=x,y=y,score=s,hit=hit))
   records.append(rec);np.savez_compressed(out/(name+'_oof.npz'),prob=pr.astype(np.float16),target=tar,valid=v,px=px,py=py)
  print('fold',fold,'auc',auc,'hits',hits.tolist(),flush=True)
 (out/'validation.json').write_text(json.dumps(dict(hits=hits.tolist(),aucs=aucs,predictions=records),indent=2))
 clf=model().fit(X,Y);joblib.dump(clf,out/'model.joblib');print('saved',flush=True)

def infer(root,out):
 model=joblib.load(out/'model.joblib');(out/'maps').mkdir(exist_ok=True);rows=[];sheet=np.full((1200,8000,3),255,np.uint8);params=[]
 for ai,album in enumerate(sorted((root/'task_kazakhstan/albums').iterdir())):
  for p in sorted(album.glob('*.jpg')):
   im=cv2.imread(str(p));dest=out/'maps'/(album.name+'_'+p.stem+'.npz')
   if dest.exists():z=np.load(dest);prob=z['prob'].astype(np.float32);valid=z['valid'].astype(bool)
   else:
    prob=np.zeros((1200,1600),np.float32);valid=np.zeros(prob.shape,bool)
    for sx,sy,sw,sh in stamps_on(im):
     x0=max(0,sx-20);y0=max(0,sy-20);x1=min(1600,sx+sw+20);y1=min(1200,sy+sh+20);crop=im[y0:y1,x0:x1]
     g=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).astype(np.float32);r=estimate(g);score,q,px,py=r[0];params.append(dict(album=album.name,page=p.stem,stamp=[sx,sy,sw,sh],quality=q,phase=[(x0+px)%8,(y0+py)%8],lattice_score=score))
     f=feat(crop,q,px,py);h,w=f.shape[:2];pr=model.predict_proba(f.reshape(-1,f.shape[-1]))[:,1].reshape(h,w)
     up=np.repeat(np.repeat(pr,8,0),8,1);prob[y0+py:y0+py+h*8,x0+px:x0+px+w*8]=up
     valid[sy+18:sy+sh-18,sx+18:sx+sw-18]=True
    np.savez_compressed(dest,prob=prob[::4,::4].astype(np.float16),valid=valid[::4,::4])
   if prob.shape==(300,400):prob=cv2.resize(prob,(1600,1200));valid=cv2.resize(valid.astype(np.uint8),(1600,1200),interpolation=cv2.INTER_NEAREST)>0
   sm=cv2.boxFilter(prob,-1,(21,21));sm[~valid]=-1;y,x=np.unravel_index(sm.argmax(),sm.shape);s=float(sm[y,x]);rows.append([album.name,p.stem,int(x)-25,int(y)-25,50,50,s]);cv2.circle(sheet,(int(x)+ai*1600,int(y)),16,(0,0,0),-1)
   print(album.name,p.stem,int(x),int(y),round(s,3),flush=True)
  cv2.imwrite(str(out/'overlay.png'),cv2.resize(sheet,(2000,300)))
  with (out/'boxes.csv').open('w',newline='') as fh:
   wr=csv.writer(fh);wr.writerow(['album','page','x','y','w','h','score']);wr.writerows(rows)
  (out/'compression.json').write_text(json.dumps(params,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');args=ap.parse_args();root=Path(__file__).parent;out=root/'jpeg';out.mkdir(exist_ok=True);cv2.setNumThreads(2)
 if args.infer:infer(root,out)
 else:train(root,out)

if __name__=='__main__':main()
