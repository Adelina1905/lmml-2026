exec(open(__file__.replace('noise_probe.py','probe.py')).read().split('root=')[0])
root=Path(__file__).parent
for p in sorted((root/'task_kazakhstan/exhibits').glob('*.jpg')):
 im=cv2.imread(str(p)); m=cv2.imread(str(p.with_name(p.stem+'_mask.png')),0)
 g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype('float32')
 r=g-cv2.GaussianBlur(g,(0,0),.7)
 edge=np.abs(g-cv2.GaussianBlur(g,(0,0),2))
 ins=cv2.erode(m,np.ones((7,7),np.uint8))>0
 out=cv2.dilate(m,np.ones((81,81),np.uint8))>0; out &= ~(cv2.dilate(m,np.ones((15,15),np.uint8))>0)
 print(p.stem,[(int((a&(edge<t)).sum()),np.round(r[a&(edge<t)].std(),3)) for a in [ins,out] for t in [1,3,10]],flush=True)
