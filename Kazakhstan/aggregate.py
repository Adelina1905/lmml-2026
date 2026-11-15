import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'packages'))
import numpy as np,cv2
from detect import valid_area
root=Path(__file__).parent
sheet=np.zeros((300*4,400*5,3),np.uint8)
for ai in range(5):
 pages=sorted((root/'results_v2/maps').glob(f'album_{ai+1}_*.npz'));sums=[np.zeros((300,400),np.float32) for _ in range(4)];base=sums[0].copy()
 for p in pages:
  pr=np.load(p)['prob'].astype(np.float32)
  page=root/'task_kazakhstan/albums'/f'album_{ai+1}'/(p.stem.split(f'album_{ai+1}_')[1]+'.jpg')
  valid=valid_area(cv2.imread(str(page)))
  pr=cv2.boxFilter(pr,-1,(7,7));pr[~valid]=0
  base+=valid/valid.sum()
  for i,beta in enumerate([1,2,4,8]):
   q=pr**beta;q/=q.sum()+1e-9;sums[i]+=q
 for i,total in enumerate(sums):
  total=cv2.GaussianBlur(total,(0,0),7)
  # Reduce the common central placement bias shared by all stamps.
  bg=cv2.GaussianBlur(base.astype(np.float32),(0,0),15)
  ratio=total/(bg+.00001)
  np.save(root/f'aggregate_{ai+1}_{i}.npy',ratio)
  a=(np.clip(ratio/np.percentile(ratio,99.5),0,1)*255).astype(np.uint8)
  vis=cv2.applyColorMap(a,cv2.COLORMAP_TURBO)
  cv2.putText(vis,f'Album {ai+1} power {[1,2,4,8][i]}',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
  sheet[i*300:(i+1)*300,ai*400:(ai+1)*400]=vis
 print(ai+1,len(pages),flush=True)
cv2.imwrite(str(root/'aggregate.jpg'),sheet)
