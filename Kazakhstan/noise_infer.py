import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'packages'))
import cv2,numpy as np,csv,joblib
from detect import features,stamps_on,valid_area,peak
from plateau import extra
from noiselearn import noise_features

root=Path(__file__).parent;out=root/'noiselearn';(out/'maps').mkdir(exist_ok=True)
model=joblib.load(out/'model.joblib');sheet=np.full((1200,8000,3),255,np.uint8);rows=[]
cv2.setNumThreads(2)
for ai,album in enumerate(sorted((root/'task_kazakhstan/albums').iterdir())):
 for p in sorted(album.glob('*.jpg')):
  dest=out/'maps'/(album.name+'_'+p.stem+'.npz');im=cv2.imread(str(p));valid=valid_area(im)
  if dest.exists():prob=np.load(dest)['prob'].astype(np.float32)
  else:
   z=np.load(root/'trufor_results/maps'/(album.name+'_'+p.stem+'.npz'))
   noise=z['noise'].astype(np.float32);prior=cv2.resize(z['prob'].astype(np.float32),(1600,1200));conf=cv2.resize(z['conf'].astype(np.float32),(1600,1200))
   prob=np.zeros(valid.shape,np.float32)
   for x,y,w,h in stamps_on(im):
    x0=max(0,(x-32)//8*8);y0=max(0,(y-32)//8*8);x1=min(1600,(x+w+39)//8*8);y1=min(1200,(y+h+39)//8*8)
    crop=im[y0:y1,x0:x1];f,_=features(crop);e=extra(crop);n=noise_features(noise[y0:y1,x0:x1],prior[y0:y1,x0:x1],conf[y0:y1,x0:x1])
    a=np.concatenate([f,e,n],2)[::2,::2]
    pr=model.predict_proba(a.reshape(-1,a.shape[-1]))[:,1].reshape(a.shape[:2]);pr=cv2.resize(pr,(f.shape[1],f.shape[0]))
    prob[y0//4:y0//4+pr.shape[0],x0//4:x0//4+pr.shape[1]]=pr
   np.savez_compressed(dest,prob=prob.astype(np.float16))
  x,y,s=peak(prob,valid);rows.append([album.name,p.stem,x-25,y-25,50,50,s]);cv2.circle(sheet,(x+1600*ai,y),16,(0,0,0),-1)
  print(album.name,p.stem,x,y,round(s,3),flush=True)
 cv2.imwrite(str(out/'overlay.png'),cv2.resize(sheet,(2000,300)))
 with (out/'boxes.csv').open('w',newline='') as fh:
  wr=csv.writer(fh);wr.writerow(['album','page','x','y','w','h','score']);wr.writerows(rows)
