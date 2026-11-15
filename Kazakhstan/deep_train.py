from trufor_runner import *
from torch.nn import functional as F
import joblib

class Head(nn.Module):
 def __init__(self,channels):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(channels,48,1),nn.BatchNorm2d(48),nn.ReLU(),nn.Conv2d(48,48,3,padding=1),nn.BatchNorm2d(48),nn.ReLU(),nn.Conv2d(48,24,3,padding=1),nn.ReLU(),nn.Conv2d(24,1,1))
 def forward(self,x):return self.net(x)

def train(data,idx,steps=1200):
 rng=np.random.default_rng(2026);torch.manual_seed(2026)
 # Use only the training exhibits to normalize feature channels.
 mean=np.mean([data[i][0].mean((1,2)) for i in idx],0)
 std=np.sqrt(np.mean([(data[i][0]-mean[:,None,None])**2 for i in []],0)) if False else np.sqrt(np.mean([data[i][0].var((1,2)) for i in idx],0)+.01)
 normalized={i:((data[i][0]-mean[:,None,None])/std[:,None,None]).astype(np.float32) for i in idx}
 net=Head(len(mean)).cuda();opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.01)
 t=time.time()
 for step in range(steps):
  xx=[];yy=[]
  for j in range(6):
   i=int(rng.choice(idx));a=normalized[i];m=data[i][1];h,w=m.shape
   if j<4:
    ys,xs=np.nonzero(m);k=int(rng.integers(len(ys)));cy=ys[k];cx=xs[k]
   else:cy=int(rng.integers(12,h-12));cx=int(rng.integers(12,w-12))
   x=int(np.clip(cx-16,0,w-32));y=int(np.clip(cy-16,0,h-32));ai=a[:,y:y+32,x:x+32];mi=m[None,y:y+32,x:x+32]
   k=int(rng.integers(4));ai=np.rot90(ai,k,(1,2));mi=np.rot90(mi,k,(1,2))
   xx.append(ai.copy());yy.append(mi.copy())
  x=torch.from_numpy(np.stack(xx)).cuda();y=torch.from_numpy(np.stack(yy)).cuda()
  opt.zero_grad();logit=net(x);loss=F.binary_cross_entropy_with_logits(logit,y,pos_weight=torch.tensor(2.,device='cuda'))
  loss.backward();opt.step()
  if step%300==0:print('step',step,round(loss.item(),4),round(time.time()-t,1),flush=True)
 return net.eval(),mean,std

def main():
 out=ROOT/'deep';data=[];paths=sorted((ROOT/'task_kazakhstan/exhibits').glob('*.jpg'))
 for p in paths:
  f=np.load(out/(p.stem+'.npz'))['features'].astype(np.float32)
  m=cv2.imread(str(p.with_name(p.stem+'_mask.png')),0)[::4,::4].astype(np.float32)/255
  data.append((f,m))
 records=[]
 for fold in range(3):
  tr=[i for i in range(30) if i%3!=fold];te=[i for i in range(30) if i%3==fold]
  net,mean,std=train(data,tr);hits=0
  for i in te:
   f,m=data[i];a=(f-mean[:,None,None])/std[:,None,None]
   with torch.inference_mode():pr=net(torch.from_numpy(a[None]).cuda()).sigmoid()[0,0].cpu().numpy()
   im=cv2.imread(str(paths[i]));valid=valid_area(im);score=cv2.boxFilter(pr,-1,(7,7));score[~valid]=-1
   y,x=np.unravel_index(score.argmax(),score.shape);hit=bool(m[y,x]);hits+=hit
   records.append(dict(exhibit=paths[i].stem,x=int(x*4),y=int(y*4),hit=hit,score=float(score[y,x])))
   np.savez_compressed(out/(paths[i].stem+'_oof.npz'),prob=pr.astype(np.float16),mask=m,valid=valid)
  print('FOLD',fold,hits,flush=True)
  del net;torch.cuda.empty_cache()
 (out/'validation.json').write_text(json.dumps(records,indent=2))
 net,mean,std=train(data,list(range(30)))
 torch.save(dict(state=net.state_dict(),mean=mean,std=std),out/'head.pt')

if __name__=='__main__':main()
