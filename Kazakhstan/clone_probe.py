exec(open(__file__.replace('clone_probe.py','probe.py')).read().split('root=')[0])
root=Path(__file__).parent
for p in sorted((root/'task_kazakhstan/exhibits').glob('*.jpg')):
 im=cv2.imread(str(p)); m=cv2.imread(str(p.with_name(p.stem+'_mask.png')),0)
 x,y,w,h=cv2.boundingRect(m)
 roi=im[y+5:y+h-5,x+5:x+w-5]
 match=cv2.matchTemplate(im,roi,cv2.TM_CCOEFF_NORMED)
 match[max(0,y-h):y+h,max(0,x-w):x+w]=-1
 _,v,_,loc=cv2.minMaxLoc(match)
 print(p.stem,round(v,4),loc,flush=True)
