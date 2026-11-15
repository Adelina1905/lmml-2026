from trufor_runner import *
from torch.nn import functional as F
import joblib

def main():
 out=ROOT/'deep';out.mkdir(exist_ok=True)
 cfg=config.clone();cfg.defrost();cfg.merge_from_file(str(ROOT/'TruFor/TruFor_train_test/lib/config/trufor_ph3.yaml'));cfg.MODEL.PRETRAINED=None
 model=EncoderDecoder(cfg);checkpoint=torch.load(ROOT/'trufor_weights/weights/trufor.pth.tar',map_location='cpu',weights_only=False)
 model.load_state_dict(checkpoint['state_dict']);del checkpoint;model.eval().cuda()
 captured=[]
 def hook(m,i,o):captured.extend(o)
 handle=model.backbone.register_forward_hook(hook)
 for p in sorted((ROOT/'task_kazakhstan/exhibits').glob('*.jpg')):
  dest=out/(p.stem+'.npz')
  if dest.exists():continue
  im=cv2.imread(str(p));h,w=im.shape[:2];captured.clear()
  with torch.inference_mode():
   predict(model,im,'cuda');shape=((h+3)//4,(w+3)//4)
   f=torch.cat([F.interpolate(v,size=shape,mode='bilinear',align_corners=False) for v in captured],1)[0].cpu().numpy().astype(np.float16)
  np.savez_compressed(dest,features=f)
  print(p.stem,f.shape,flush=True)
 handle.remove()

if __name__=='__main__':main()
