import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'packages'))
import cv2,numpy as np
from detect import valid_area
root=Path(__file__).parent
lut=cv2.applyColorMap(np.arange(256,dtype=np.uint8).reshape(-1,1),cv2.COLORMAP_TURBO).reshape(256,3).astype(float)
hits=np.zeros((3,5),int)
for p in sorted((root/'results_v2').glob('*validation.jpg')):
 im=cv2.imread(str(p));im,heat=np.split(im,2,axis=1);heat=heat[::4,::4].astype(float)
 prob=np.argmin(np.sum((heat[:,:,None,:]-lut[None,None,:,:])**2,axis=3),axis=2).astype(np.float32)/255
 mask=cv2.imread(str(root/'task_kazakhstan/exhibits'/(p.name.replace('_validation.jpg','_mask.png'))),0)>0
 valid=valid_area(im)
 for i,s in enumerate([3,5,7]):
  mean=cv2.boxFilter(prob,-1,(s,s))
  for j,outer in enumerate([0,15,25,41,61]):
   score=mean.copy()
   if outer:score-=cv2.boxFilter(prob,-1,(outer,outer))
   score[~valid]=-10;y,x=np.unravel_index(score.argmax(),score.shape)
   hits[i,j]+=mask[y*4,x*4]
print(hits)
