"""Diagnostic: fit letter templates jointly to all 26 per-page forensic maps."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'packages'))
import cv2,numpy as np,json,string
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import maximum_filter

def glyph(letter,font):
 a=np.zeros((400,400),np.uint8);cv2.putText(a,letter,(20,320),font,10,255,1,cv2.LINE_8)
 y,x=np.nonzero(a);p=np.stack([x,y],1).astype(float);p-=p.min(0);den=p.max(0);den[den==0]=1;p=p/den-.5
 # Cover the glyph with 26 separated representative points.
 selected=[0];d=((p-p[0])**2).sum(1)
 for _ in range(25):
  k=int(d.argmax());selected.append(k);d=np.minimum(d,((p-p[k])**2).sum(1))
 return p[selected]

def main():
 root=Path(__file__).parent;out=root/'joint';out.mkdir(exist_ok=True);records=[]
 transforms=np.array([(cx,cy,w,h) for cx in [175,200,225] for cy in [125,150,175] for w in [100,130,160,190,220,250] for h in [130,160,190,220,250]])
 all_costs=[]
 for ai in range(1,6):
  paths=sorted((root/'results_v2/maps').glob(f'album_{ai}_*.npz'));scores=[]
  for p in paths:
   q=np.load(p)['prob'].astype(np.float32);q=cv2.boxFilter(q,-1,(7,7))
   q=np.log((q+.015)/(1.015-q))
   q=maximum_filter(q,size=17,mode='nearest');scores.append(q)
  scores=np.stack(scores);best=[]
  album_cost=[]
  for font in [cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_PLAIN]:
   font_cost=[]
   for letter in string.ascii_uppercase:
    points=glyph(letter,font)
    pts=points[None,:,:]*transforms[:,None,2:]+transforms[:,None,:2]
    pts=np.rint(pts).astype(int);xs=np.clip(pts[:,:,0],0,399);ys=np.clip(pts[:,:,1],0,299)
    mats=scores[:,ys,xs].transpose(1,0,2)
    top=(-1e10,None,None);costs=[]
    for j,mat in enumerate(mats):
     r,c=linear_sum_assignment(mat,maximize=True);value=float(mat[r,c].sum())
     costs.append(value)
     if value>top[0]:top=(value,j,c)
    value,j,c=top
    best.append(dict(letter=letter,font=int(font),score=value,transform=transforms[j].tolist(),points=pts[j].tolist(),assignment=c.tolist()))
    font_cost.append(costs)
   album_cost.append(font_cost)
  all_costs.append(album_cost)
  best.sort(key=lambda d:d['score'],reverse=True)
  records.append(dict(album=ai,candidates=best))
  print('album',ai,[(b['letter'],b['font'],round(b['score'],2),b['transform']) for b in best[:10]],flush=True)
  (out/'ranking.json').write_text(json.dumps(records,indent=2))
 costs=np.array(all_costs);np.save(out/'costs.npy',costs)
 combined=costs.max(2).sum(0);order=np.argsort(combined.ravel())[::-1][:20];shared=[]
 for index in order:
  fi,ti=np.unravel_index(index,combined.shape);letters=costs[:,fi,:,ti].argmax(1)
  row=dict(word=''.join(string.ascii_uppercase[i] for i in letters),font=int(fi),transform=transforms[ti].tolist(),score=float(combined[fi,ti]))
  shared.append(row);print('SHARED',row,flush=True)
 (out/'shared.json').write_text(json.dumps(shared,indent=2))

if __name__=='__main__':main()
