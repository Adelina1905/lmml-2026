"""Evaluate the official pretrained TruFor checkpoint on exhibits and album crops."""
import sys,types,argparse,time,json,csv
from pathlib import Path
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'packages'))
if (ROOT/'gpu_packages').exists():sys.path.insert(0,str(ROOT/'gpu_packages'))
sys.path.insert(0,str(ROOT/'TruFor/TruFor_train_test'))
import numpy as np,cv2,torch
from torch import nn
torch.set_num_threads(4);cv2.setNumThreads(1)

# The upstream inference graph only needs these three timm utilities.
# DropPath is exactly the identity in evaluation mode; initialization is
# overwritten by checkpoint parameters. This avoids an unrelated vision SDK.
class EvalDropPath(nn.Module):
 def __init__(self,drop_prob=0.):super().__init__();self.drop_prob=drop_prob
 def forward(self,x):
  if self.training and self.drop_prob:raise RuntimeError('Inference-only compatibility layer')
  return x
layers=types.ModuleType('timm.models.layers');layers.DropPath=EvalDropPath
layers.to_2tuple=lambda x:x if isinstance(x,tuple) else (x,x)
layers.trunc_normal_=nn.init.trunc_normal_
sys.modules['timm']=types.ModuleType('timm');sys.modules['timm.models']=types.ModuleType('timm.models');sys.modules['timm.models.layers']=layers
from lib.config import config
from lib.models.cmx.builder_np_conf import EncoderDecoder
from detect import stamps_on,valid_area

def predict(model,im,device):
 a=np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1))
 x=torch.from_numpy(a).unsqueeze(0).to(device=device,dtype=torch.float32)/256.
 with torch.inference_mode():
  pred,conf,det,npp=model(x,save_np=True)
  prob=pred.softmax(1)[0,1].cpu().numpy()
  conf=conf.sigmoid()[0,0].cpu().numpy()
  noise=npp[0,0].cpu().numpy()
 return prob,conf,noise

def choose(prob,conf,valid):
 ans=[]
 for a in [prob,prob*conf]:
  sm=cv2.boxFilter(a.astype(np.float32),-1,(25,25));sm[~valid]=-1
  y,x=np.unravel_index(sm.argmax(),sm.shape);ans.append((int(x),int(y),float(sm[y,x])))
 return ans

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pages',action='store_true');ap.add_argument('--cpu',action='store_true');ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
 device='cuda' if torch.cuda.is_available() and not args.cpu else 'cpu';print('device',device,flush=True)
 cfg=config.clone();cfg.defrost();cfg.merge_from_file(str(ROOT/'TruFor/TruFor_train_test/lib/config/trufor_ph3.yaml'));cfg.MODEL.PRETRAINED=None
 model=EncoderDecoder(cfg);checkpoint=torch.load(ROOT/'trufor_weights/weights/trufor.pth.tar',map_location='cpu',weights_only=False)
 model.load_state_dict(checkpoint['state_dict']);del checkpoint;model.eval().to(device);print('loaded',flush=True)
 out=ROOT/'trufor_results';out.mkdir(exist_ok=True);(out/'exhibits').mkdir(exist_ok=True);(out/'maps').mkdir(exist_ok=True)
 metrics=[];t=time.time()
 for i,p in enumerate(sorted((ROOT/'task_kazakhstan/exhibits').glob('*.jpg'))):
  if args.limit and i>=args.limit:break
  dest=out/'exhibits'/(p.stem+'.npz');im=cv2.imread(str(p));mask=cv2.imread(str(p.with_name(p.stem+'_mask.png')),0)>0
  if dest.exists() and 'noise' in np.load(dest):z=np.load(dest);prob=z['prob'].astype(np.float32);conf=z['conf'].astype(np.float32)
  else:
   prob,conf,noise=predict(model,im,device);np.savez_compressed(dest,prob=prob.astype(np.float16),conf=conf.astype(np.float16),noise=noise.astype(np.float16))
  valid=cv2.resize(valid_area(im).astype(np.uint8),(im.shape[1],im.shape[0]),interpolation=cv2.INTER_NEAREST)>0
  pp=choose(prob,conf,valid);metrics.append(dict(exhibit=p.stem,predictions=[dict(x=x,y=y,score=s,hit=bool(mask[y,x])) for x,y,s in pp]))
  x,y,_=pp[0];cv2.circle(im,(x,y),12,(0,0,255),2)
  cv2.imwrite(str(out/'exhibits'/(p.stem+'.jpg')),np.hstack([im,cv2.applyColorMap(np.uint8(prob*255),cv2.COLORMAP_TURBO)]))
  print(p.stem,metrics[-1]['predictions'],'seconds',round(time.time()-t,1),flush=True)
 (out/'validation.json').write_text(json.dumps(metrics,indent=2))
 if not args.pages:return
 rows=[];sheet=np.full((1200,8000,3),255,np.uint8)
 for ai,album in enumerate(sorted((ROOT/'task_kazakhstan/albums').iterdir())):
  for p in sorted(album.glob('*.jpg')):
   dest=out/'maps'/(album.name+'_'+p.stem+'.npz');im=cv2.imread(str(p))
   if dest.exists():z=np.load(dest);prob=cv2.resize(z['prob'].astype(np.float32),(1600,1200));conf=cv2.resize(z['conf'].astype(np.float32),(1600,1200))
   else:
    prob=np.zeros((1200,1600),np.float32);conf=prob.copy();noise=prob.copy()
    for x,y,w,h in stamps_on(im):
     x0=max(0,x-40);y0=max(0,y-40);x1=min(1600,x+w+40);y1=min(1200,y+h+40)
     pr,co,no=predict(model,im[y0:y1,x0:x1],device)
     prob[y0:y1,x0:x1]=pr;conf[y0:y1,x0:x1]=co;noise[y0:y1,x0:x1]=no
    np.savez_compressed(dest,prob=prob[::4,::4].astype(np.float16),conf=conf[::4,::4].astype(np.float16),noise=noise.astype(np.float16))
   valid=cv2.resize(valid_area(im).astype(np.uint8),(1600,1200),interpolation=cv2.INTER_NEAREST)>0
   x,y,s=choose(prob,conf,valid)[0];rows.append([album.name,p.stem,x-25,y-25,50,50,s])
   cv2.circle(sheet,(x+ai*1600,y),16,(0,0,0),-1);print(album.name,p.stem,x,y,round(s,3),flush=True)
  cv2.imwrite(str(out/'overlay.png'),cv2.resize(sheet,(2000,300)))
  with (out/'boxes.csv').open('w',newline='') as fh:
   wr=csv.writer(fh);wr.writerow(['album','page','x','y','w','h','score']);wr.writerows(rows)

if __name__=='__main__':main()
