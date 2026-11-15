from trufor_runner import *
from torch.nn import functional as F

def inp(im,n):
 a=im.astype(np.float32);n=n.astype(np.float32)/.14
 return np.concatenate([n[:,:,None],(n-cv2.GaussianBlur(n,(0,0),1))[:,:,None],(n-cv2.GaussianBlur(n,(0,0),3))[:,:,None],(a-cv2.GaussianBlur(a,(0,0),.7))/8,(a-cv2.GaussianBlur(a,(0,0),2))/16],2).transpose(2,0,1)

class Net(nn.Module):
 def __init__(self):
  super().__init__()
  def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU())
  self.a=block(9,16);self.b=block(16,32);self.c=block(32,64);self.d=block(64,96);self.e=block(96+64,64);self.f=block(64+32,32);self.g=block(32+16,16);self.out=nn.Conv2d(16,1,1)
 def forward(self,x):
  a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2));d=self.d(F.avg_pool2d(c,2))
  e=self.e(torch.cat([F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1));f=self.f(torch.cat([F.interpolate(e,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
  return self.out(self.g(torch.cat([F.interpolate(f,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1)))

def main():
 out=ROOT/'residual_net';out.mkdir(exist_ok=True);rng=np.random.default_rng(2026);torch.manual_seed(2026);data=[];paths=sorted((ROOT/'task_kazakhstan/exhibits').glob('*.jpg'))
 for p in paths:
  im=cv2.imread(str(p));m=cv2.imread(str(p.with_name(p.stem+'_mask.png')),0).astype(np.float32)/255;n=np.load(ROOT/'trufor_results/exhibits'/(p.stem+'.npz'))['noise'];a=inp(im,n)
  valid=cv2.resize(valid_area(im).astype(np.uint8),(im.shape[1],im.shape[0]),interpolation=cv2.INTER_NEAREST)>0
  pos=np.stack(np.nonzero(m),1);neg=np.stack(np.nonzero(valid&(m==0)),1)
  data.append((a,m,pos,neg,valid))
 tr=[i for i in range(30) if i%3!=0];te=[i for i in range(30) if i%3==0]
 net=Net().cuda();opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.01);t=time.time()
 for step in range(1600):
  xx=[];yy=[]
  for j in range(8):
   i=int(rng.choice(tr));a,m,pos,neg,v=data[i];h,w=m.shape;pool=pos if j<5 else neg;cy,cx=pool[int(rng.integers(len(pool)))];x=int(np.clip(cx-64,0,w-128));y=int(np.clip(cy-64,0,h-128));ai=a[:,y:y+128,x:x+128];mi=m[None,y:y+128,x:x+128]
   k=int(rng.integers(4));ai=np.rot90(ai,k,(1,2));mi=np.rot90(mi,k,(1,2));xx.append(ai.copy());yy.append(mi.copy())
  x=torch.from_numpy(np.stack(xx)).cuda();y=torch.from_numpy(np.stack(yy)).cuda();opt.zero_grad();logit=net(x);loss=F.binary_cross_entropy_with_logits(logit,y,pos_weight=torch.tensor(2.,device='cuda'));loss.backward();opt.step()
  if step%400==0:print('step',step,round(loss.item(),4),round(time.time()-t,1),flush=True)
 net.eval();records=[]
 for i in te:
  a,m,_,_,valid=data[i];h,w=m.shape;a=np.pad(a,((0,0),(0,(-h)%8),(0,(-w)%8)),mode='reflect')
  with torch.inference_mode():pr=net(torch.from_numpy(a[None]).cuda()).sigmoid()[0,0,:h,:w].cpu().numpy()
  score=cv2.boxFilter(pr,-1,(25,25));score[~valid]=-1;y,x=np.unravel_index(score.argmax(),score.shape)
  records.append(dict(exhibit=paths[i].stem,x=int(x),y=int(y),hit=bool(m[y,x]),score=float(score[y,x])))
  np.savez_compressed(out/(paths[i].stem+'_oof.npz'),prob=pr.astype(np.float16),mask=m,valid=valid)
 print('RESULT',sum(r['hit'] for r in records),records,flush=True);(out/'validation.json').write_text(json.dumps(records,indent=2));torch.save(net.state_dict(),out/'validation.pt')

if __name__=='__main__':main()
