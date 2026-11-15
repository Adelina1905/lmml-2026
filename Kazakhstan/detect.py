"""Learn local forensic features from exhibit masks, then localize one edit per page."""
import sys
from pathlib import Path
if (Path(__file__).parent/'packages').exists():
    sys.path.insert(0,str(Path(__file__).parent/'packages'))
import argparse, csv, json, time
import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import joblib

cv2.setNumThreads(2)
STRIDE=4

def features(im, spatial=False):
    """Content and multiscale high-frequency statistics; no page coordinates."""
    a=im.astype(np.float32)
    gray=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
    ch=cv2.cvtColor(a,cv2.COLOR_BGR2YCrCb)
    fs=[]; names=[]
    def add(a,n):
        fs.append(a[::STRIDE,::STRIDE]); names.append(n)
    bases=[]
    for k in range(3):
        v=ch[:,:,k]
        add(v,'color'+str(k))
        for sig in [0.7,1.4,2.8]:
            r=v-cv2.GaussianBlur(v,(0,0),sig)
            bases.append((np.abs(r),f'hp{k}_{sig}'))
    med=cv2.medianBlur(im,3).astype(np.float32)
    for k in range(3):bases.append((np.abs(a[:,:,k]-med[:,:,k]),f'med{k}'))
    lap=np.abs(cv2.Laplacian(gray,cv2.CV_32F))
    bases.append((lap,'lap'))
    for q in [75,90,95]:
        again=cv2.imdecode(cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,q])[1],1)
        bases.append((np.abs(a-again.astype(np.float32)).mean(2),f'ela{q}'))
    for b,n in bases:
        for s in [5,13,29,61]:
            mean=cv2.boxFilter(b,-1,(s,s))
            add(np.log1p(mean),f'{n}_mean{s}')
        for s in [13,29]:
            mean=cv2.boxFilter(b,-1,(s,s))
            sq=cv2.boxFilter(b*b,-1,(s,s))
            add(np.log1p(np.maximum(sq-mean*mean,0)),f'{n}_var{s}')
        inner=cv2.boxFilter(b,-1,(13,13)); outer=cv2.boxFilter(b,-1,(61,61))
        add((inner+.2)/(outer+.2),n+'_ratio')
    for s in [7,19,41]:
        mu=cv2.boxFilter(gray,-1,(s,s))
        add(np.log1p(np.maximum(cv2.boxFilter(gray*gray,-1,(s,s))-mu*mu,0)),f'gray_var{s}')
    # Estimate noise only where coarse structure is weak. Text edges otherwise
    # dominate ordinary residual averages and hide changes in the noise floor.
    coarse=np.abs(gray-cv2.GaussianBlur(gray,(0,0),2))
    residual=gray-cv2.GaussianBlur(gray,(0,0),.7)
    for threshold in [1,3,7]:
        flat=(coarse<threshold).astype(np.float32)
        for s in [13,29,61]:
            count=cv2.boxFilter(flat,-1,(s,s))
            var=cv2.boxFilter(residual*residual*flat,-1,(s,s))/(count+.001)
            add(np.log1p(var),f'conditional_noise_{threshold}_{s}')
            add(count,f'flat_fraction_{threshold}_{s}')
    for delta in [(1,0),(0,1),(1,1),(2,0),(0,2)]:
        dx,dy=delta
        shifted=np.roll(residual,(dy,dx),(0,1))
        for s in [13,29]:
            cov=cv2.boxFilter(residual*shifted,-1,(s,s))
            energy=cv2.boxFilter(residual*residual,-1,(s,s))
            add(cov/(energy+.1),f'residual_corr_{dx}_{dy}_{s}')
    if spatial:
        coords=np.zeros((*im.shape[:2],5),np.float32)
        for x,y,w,h in stamps_on(im):
            yy,xx=np.mgrid[0:h,0:w]
            coords[y:y+h,x:x+w,0]=xx/w
            coords[y:y+h,x:x+w,1]=yy/h
            coords[y:y+h,x:x+w,2]=np.minimum(xx,w-1-xx)/w
            coords[y:y+h,x:x+w,3]=np.minimum(yy,h-1-yy)/h
            coords[y:y+h,x:x+w,4]=w/h
        for k in range(5):add(coords[:,:,k],f'stamp_geometry_{k}')
    return np.stack(fs,axis=-1), names

def stamps_on(im):
    d=np.abs(im.astype(np.int16)-np.array([38,36,34])).max(2)
    m=cv2.morphologyEx((d>30).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    _,_,stats,_=cv2.connectedComponentsWithStats(m)
    return [(int(x),int(y),int(w),int(h)) for x,y,w,h,ar in stats[1:] if w>120 and h>120]

def valid_area(im):
    inside=np.zeros(im.shape[:2],np.uint8)
    for x,y,w,h in stamps_on(im):
        inside[y+18:y+h-18,x+18:x+w-18]=1
    return inside[::STRIDE,::STRIDE]>0

def peak(prob,valid):
    # Fixed local support selected without album labels or letter priors.
    score=cv2.boxFilter(prob.astype(np.float32),-1,(7,7),normalize=True)
    score[~valid]=-1
    y,x=np.unravel_index(score.argmax(),score.shape)
    return int(x*STRIDE),int(y*STRIDE),float(score[y,x])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,default=Path(__file__).parent/'task_kazakhstan')
    ap.add_argument('--out',type=Path,default=Path(__file__).parent/'results'); ap.add_argument('--validate',action='store_true'); ap.add_argument('--reuse',action='store_true')
    ap.add_argument('--spatial',action='store_true')
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(2026); xs=[]; ys=[]; groups=[]; exhibits=[]
    for i,p in enumerate(sorted((args.data/'exhibits').glob('*.jpg'))):
        im=cv2.imread(str(p)); m=cv2.imread(str(p.with_name(p.stem+'_mask.png')),0)>0
        f,names=features(im,args.spatial); target=m[::STRIDE,::STRIDE]; valid=valid_area(im)
        pos=np.flatnonzero(target.ravel()); neg=np.flatnonzero((valid&~target).ravel())
        ids=np.r_[rng.choice(pos,min(len(pos),500),replace=False),rng.choice(neg,min(len(neg),3000),replace=False)]
        xs.append(f.reshape(-1,f.shape[-1])[ids]); ys.append(target.ravel()[ids]); groups.extend([i]*len(ids))
        exhibits.append((p,im,f,target,valid,m))
    X=np.concatenate(xs); y=np.concatenate(ys); groups=np.array(groups)
    print('training',X.shape,'positives',int(y.sum()),flush=True)
    def model():return HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=23,l2_regularization=5,learning_rate=.08,random_state=2026)
    if args.validate:
        metrics=[]; valrows=[]
        for fold,(tr,te) in enumerate(GroupKFold(3).split(X,y,groups)):
            clf=model().fit(X[tr],y[tr]); auc=roc_auc_score(y[te],clf.predict_proba(X[te])[:,1]); hits=[]
            for i in np.unique(groups[te]):
                p,im,f,target,valid,m=exhibits[i]
                prob=clf.predict_proba(f.reshape(-1,f.shape[-1]))[:,1].reshape(f.shape[:2])
                x,cy,s=peak(prob,valid); hit=bool(m[cy,x]); hits.append(hit)
                valrows.append(dict(exhibit=p.stem,x=x,y=cy,score=s,hit=hit))
                heat=cv2.applyColorMap(np.uint8(prob*255),cv2.COLORMAP_TURBO); heat=cv2.resize(heat,(im.shape[1],im.shape[0]))
                cv2.circle(im,(x,cy),12,(0,0,255),2)
                cv2.imwrite(str(args.out/(p.stem+'_validation.jpg')),np.hstack([im,heat]))
            metrics.append(dict(fold=fold,auc=auc,hits=sum(hits),total=len(hits)))
            print('validation',metrics[-1],flush=True)
        (args.out/'validation.json').write_text(json.dumps(dict(folds=metrics,predictions=valrows),indent=2))
    if args.reuse and (args.out/'model.joblib').exists():clf=joblib.load(args.out/'model.joblib')
    else:
        clf=model().fit(X,y); joblib.dump(clf,args.out/'model.joblib')
    (args.out/'features.json').write_text(json.dumps(names,indent=2))
    rows=[]; sheet=np.full((1200,1600*5,3),255,np.uint8)
    (args.out/'pages').mkdir(exist_ok=True)
    (args.out/'maps').mkdir(exist_ok=True)
    for ai,album in enumerate(sorted((args.data/'albums').iterdir())):
        for p in sorted(album.glob('*.jpg')):
            im=cv2.imread(str(p)); valid=valid_area(im)
            map_path=args.out/'maps'/(album.name+'_'+p.stem+'.npz')
            if map_path.exists():
                prob=np.load(map_path)['prob'].astype(np.float32)
                x,y,s=peak(prob,valid);rows.append([album.name,p.stem,x-25,y-25,50,50,s])
                cv2.circle(sheet,(x+ai*1600,y),16,(0,0,0),-1)
                continue
            prob=np.zeros(valid.shape,np.float32)
            for sx,sy,sw,sh in stamps_on(im):
                # Align to the JPEG grid and retain context around the stamp.
                x0=max(0,(sx-32)//8*8); y0=max(0,(sy-32)//8*8)
                x1=min(im.shape[1],(sx+sw+39)//8*8); y1=min(im.shape[0],(sy+sh+39)//8*8)
                f,_=features(im[y0:y1,x0:x1],args.spatial)
                sub=f[::2,::2]
                pr=clf.predict_proba(sub.reshape(-1,f.shape[-1]))[:,1].reshape(sub.shape[:2])
                pr=cv2.resize(pr,(f.shape[1],f.shape[0]),interpolation=cv2.INTER_LINEAR)
                prob[y0//STRIDE:(y0//STRIDE+pr.shape[0]),x0//STRIDE:(x0//STRIDE+pr.shape[1])]=pr
            np.savez_compressed(map_path,prob=prob.astype(np.float16))
            x,y,s=peak(prob,valid); rows.append([album.name,p.stem,x-25,y-25,50,50,s])
            cv2.circle(sheet,(x+ai*1600,y),16,(0,0,0),-1)
            cv2.rectangle(im,(x-25,y-25),(x+25,y+25),(0,0,255),3)
            cv2.imwrite(str(args.out/'pages'/(album.name+'_'+p.name)),cv2.resize(im,(800,600)))
            print(album.name,p.stem,x,y,round(s,3),flush=True)
        cv2.imwrite(str(args.out/'overlay.png'),cv2.resize(sheet,(2000,300)))
    with (args.out/'boxes.csv').open('w',newline='') as fh:
        w=csv.writer(fh); w.writerow(['album','page','x','y','w','h','score']); w.writerows(rows)

if __name__=='__main__':main()
