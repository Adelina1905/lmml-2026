import sys
sys.path.insert(0,str(__import__('pathlib').Path(__file__).parent/'packages'))
from pathlib import Path
import cv2, numpy as np
from PIL import Image,ImageDraw

root=Path(__file__).parent
panels=[]
for p in sorted((root/'task_kazakhstan/exhibits').glob('*.jpg')):
    im=cv2.imread(str(p)); m=cv2.imread(str(p.with_name(p.stem+'_mask.png')),0)
    x,y,w,h=cv2.boundingRect(m)
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype('float32')
    r=np.abs(g-cv2.GaussianBlur(g,(0,0),1))
    r=cv2.boxFilter(r,-1,(7,7))
    inside=m>0; outside=cv2.dilate(m,np.ones((71,71),np.uint8))>0; outside &= ~inside
    print(p.stem,(x,y,w,h),'residual',np.round([r[inside].mean(),r[outside].mean()],2),flush=True)
    cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
    vis=np.hstack([cv2.resize(im,(300,270)),cv2.resize(cv2.applyColorMap(np.uint8(np.clip(r*15,0,255)),cv2.COLORMAP_TURBO),(300,270))])
    panels.append(Image.fromarray(cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)))
out=Image.new('RGB',(1800,270*10))
for i,p in enumerate(panels):out.paste(p,((i%3)*600,(i//3)*270))
out.save(root/'exhibit_forensics.jpg')
