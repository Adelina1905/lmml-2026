import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'packages'))
import cv2,numpy as np,torch,time,json,argparse,csv
from torch import nn
from torch.nn import functional as F
from detect import stamps_on
torch.set_num_threads(4); cv2.setNumThreads(1)
torch.manual_seed(2026); np.random.seed(2026)

def inputs(im):
 a=im.astype(np.float32)
 return np.concatenate([a/127.5-1,(a-cv2.GaussianBlur(a,(0,0),.7))/8,(a-cv2.GaussianBlur(a,(0,0),2))/16],2).transpose(2,0,1)

class Net(nn.Module):
 def __init__(self):
  super().__init__()
  def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.ReLU())
  self.a=block(9,12); self.b=block(12,24); self.c=block(24,48); self.d=block(48,64)
  self.e=block(64+48,48); self.f=block(48+24,24); self.g=block(24+12,12); self.out=nn.Conv2d(12,1,1)
 def forward(self,x):
  a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2));d=self.d(F.avg_pool2d(c,2))
  e=self.e(torch.cat([F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
  f=self.f(torch.cat([F.interpolate(e,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
  return self.out(self.g(torch.cat([F.interpolate(f,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1)))

def predict(net,im):
 with torch.no_grad():
  a=inputs(im); h,w=im.shape[:2]
  a=np.pad(a,((0,0),(0,(-h)%8),(0,(-w)%8)),mode='reflect')
  return torch.sigmoid(net(torch.from_numpy(a[None])))[0,0,:h,:w].numpy()

def train(data,steps,out,tag):
 net=Net(); opt=torch.optim.Adam(net.parameters(),lr=.001); rng=np.random.default_rng(2026)
 t=time.time()
 for step in range(steps):
  xx=[];yy=[]
  for j in range(8):
   a,m,box=data[int(rng.integers(len(data)))];h,w=m.shape
   if j<5:
    x,y,bw,bh=box;cx=rng.integers(x,x+bw);cy=rng.integers(y,y+bh)
   else:cx=rng.integers(50,w-50);cy=rng.integers(50,h-50)
   x=int(np.clip(cx-48,0,w-96)); y=int(np.clip(cy-48,0,h-96))
   ai=a[:,y:y+96,x:x+96];mi=m[None,y:y+96,x:x+96]
   k=int(rng.integers(4));ai=np.rot90(ai,k,(1,2));mi=np.rot90(mi,k,(1,2))
   if rng.random()<.5:ai=ai[:,:,::-1];mi=mi[:,:,::-1]
   xx.append(ai.copy());yy.append(mi.copy())
  x=torch.from_numpy(np.stack(xx));y=torch.from_numpy(np.stack(yy))
  opt.zero_grad();logits=net(x);loss=F.binary_cross_entropy_with_logits(logits,y,pos_weight=torch.tensor(2.))
  loss.backward();opt.step()
  if step%100==0:print(tag,step,round(loss.item(),4),round(time.time()-t,1),flush=True)
 torch.save(net.state_dict(),out/(tag+'.pt'));return net.eval()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--out',type=Path,default=Path(__file__).parent/'cnn');ap.add_argument('--reuse',action='store_true');ap.add_argument('--validation-only',action='store_true');args=ap.parse_args()
 args.out.mkdir(parents=True,exist_ok=True);data=[];ims=[];paths=[]
 root=Path(__file__).parent/'task_kazakhstan'
 for p in sorted((root/'exhibits').glob('*.jpg')):
  im=cv2.imread(str(p));m=cv2.imread(str(p.with_name(p.stem+'_mask.png')),0)
  data.append((inputs(im),m.astype(np.float32)/255,cv2.boundingRect(m)));ims.append(im);paths.append(p)
 if not args.reuse:
  valids=[0,5,10,15,20,25];net=train([d for i,d in enumerate(data) if i not in valids],args.steps,args.out,'validation')
  metrics=[]
  for i in valids:
   im=ims[i];prob=predict(net,im);score=cv2.boxFilter(prob,-1,(21,21));inside=np.zeros(prob.shape,np.uint8)
   for x,y,w,h in stamps_on(im):inside[y+20:y+h-20,x+20:x+w-20]=1
   score[inside==0]=-1;y,x=np.unravel_index(score.argmax(),score.shape);hit=bool(data[i][1][y,x])
   metrics.append(dict(exhibit=paths[i].stem,x=int(x),y=int(y),hit=hit,score=float(score[y,x])))
   cv2.circle(im,(x,y),12,(0,0,255),2)
   cv2.imwrite(str(args.out/(paths[i].stem+'.jpg')),np.hstack([im,cv2.applyColorMap(np.uint8(prob*255),cv2.COLORMAP_TURBO)]))
  (args.out/'validation.json').write_text(json.dumps(metrics,indent=2));print('VALIDATION',metrics,flush=True)
  if args.validation_only:return
  net=train(data,args.steps,args.out,'model')
 else:net=Net();net.load_state_dict(torch.load(args.out/'model.pt',weights_only=True));net.eval()
 rows=[];sheet=np.full((1200,8000,3),255,np.uint8);(args.out/'maps').mkdir(exist_ok=True)
 for ai,album in enumerate(sorted((root/'albums').iterdir())):
  for p in sorted(album.glob('*.jpg')):
   im=cv2.imread(str(p));prob=np.zeros(im.shape[:2],np.float32);inside=np.zeros(prob.shape,np.uint8)
   for x,y,w,h in stamps_on(im):
    x0=max(0,x-24);y0=max(0,y-24);x1=min(1600,x+w+24);y1=min(1200,y+h+24)
    prob[y0:y1,x0:x1]=predict(net,im[y0:y1,x0:x1]);inside[y+20:y+h-20,x+20:x+w-20]=1
   score=cv2.boxFilter(prob,-1,(25,25));score[inside==0]=-1;y,x=np.unravel_index(score.argmax(),score.shape)
   rows.append([album.name,p.stem,int(x)-25,int(y)-25,50,50,float(score[y,x])])
   np.savez_compressed(args.out/'maps'/(album.name+'_'+p.stem+'.npz'),prob=prob[::4,::4].astype(np.float16))
   cv2.circle(sheet,(int(x)+ai*1600,int(y)),16,(0,0,0),-1)
   print(album.name,p.stem,int(x),int(y),round(float(score[y,x]),3),flush=True)
  cv2.imwrite(str(args.out/'overlay.png'),cv2.resize(sheet,(2000,300)))
  with (args.out/'boxes.csv').open('w',newline='') as fh:
   writer=csv.writer(fh);writer.writerow(['album','page','x','y','w','h','score']);writer.writerows(rows)

if __name__=='__main__':main()
