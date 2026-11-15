import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'packages'))
import numpy as np,cv2,json

LUMA=np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]],np.float32)
D=(np.cos(np.pi*(np.arange(8)[None,:]+.5)*np.arange(8)[:,None]/8)*np.sqrt(2/8)).astype(np.float32)
D[0]/=np.sqrt(2)

def qtable(quality):
 scale=5000/quality if quality<50 else 200-2*quality
 return np.clip(np.floor((LUMA*scale+50)/100),1,255)

def dct_blocks(g,px=0,py=0):
 h,w=g.shape;h=(h-py)//8*8;w=(w-px)//8*8
 a=g[py:py+h,px:px+w]-128
 blocks=a.reshape(h//8,8,w//8,8).transpose(0,2,1,3)
 return D@blocks@D.T

def estimate(g,all_phases=True):
 rng=np.random.default_rng(2026);rank=[]
 qualities=list(range(55,97));qs=np.stack([qtable(q) for q in qualities]).reshape(len(qualities),64)
 inds=np.array([i*8+j for i in range(5) for j in range(5) if i+j>0 and i+j<=6])
 for py in range(8 if all_phases else 1):
  for px in range(8 if all_phases else 1):
   c=dct_blocks(g,px,py).reshape(-1,64)
   if len(c)>600:c=c[rng.choice(len(c),600,replace=False)]
   c=c[:,inds];q=qs[:,inds]
   ratio=c[None,:,:]/q[:,None,:];valid=np.abs(ratio)>1.5
   # Discard near-zero coefficients, which do not identify a quantization grid.
   score=(np.cos(2*np.pi*ratio)*valid).sum((1,2))/np.maximum(valid.sum((1,2)),1)
   for k in np.argsort(score)[-3:]:rank.append((float(score[k]),int(qualities[k]),px,py))
 return sorted(rank,reverse=True)

def main():
 root=Path(__file__).parent;records=[]
 for p in sorted((root/'task_kazakhstan/exhibits').glob('*.jpg')):
  im=cv2.imread(str(p));g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
  rank=estimate(g);print(p.stem,rank[:5],flush=True);records.append(dict(exhibit=p.stem,rank=rank[:10]))
 (root/'jpeg_probe.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':main()
