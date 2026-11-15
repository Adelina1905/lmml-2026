from pathlib import Path
import shutil,json

ROOT=Path(__file__).parent
DEST=ROOT.parent/'lmml-2026/task_kazakhstan'
DEST.mkdir(exist_ok=True)
(DEST/'models').mkdir(exist_ok=True)

utils=(ROOT/'jpeg_probe.py').read_text()
utils=utils[:utils.index('\ndef main():')]
utils += '''\n\ndef stamps_on(im):
    """Locate stamp rectangles against the dark stock-book background."""
    d=np.abs(im.astype(np.int16)-np.array([38,36,34])).max(2)
    m=cv2.morphologyEx((d>30).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    _,_,stats,_=cv2.connectedComponentsWithStats(m)
    return [(int(x),int(y),int(w),int(h)) for x,y,w,h,area in stats[1:] if w>120 and h>120]
'''
utils=utils.replace("sys.path.insert(0,str(Path(__file__).parent/'packages'))\n",'')
(DEST/'jpeg_utils.py').write_text(utils)
code=(ROOT/'jpeg_detect.py').read_text()
code=code.replace("sys.path.insert(0,str(Path(__file__).parent/'packages'))\n",'')
code=code.replace('from jpeg_probe import estimate,qtable,dct_blocks\nfrom detect import stamps_on','from jpeg_utils import estimate,qtable,dct_blocks,stamps_on')
code=code.replace('def train(root,out):','def train(data,out):')
code=code.replace("rank={d['exhibit']:d['rank'][0] for d in json.loads((root/'jpeg_probe.json').read_text())};data=[];xs=[];ys=[];gs=[]", "rank={}\n for p in sorted((data/'exhibits').glob('*.jpg')):\n  g=cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2GRAY).astype(np.float32)\n  rank[p.stem]=estimate(g)[0]\n (out/'training_compression.json').write_text(json.dumps(rank,indent=2))\n dataset=[];xs=[];ys=[];gs=[]")
code=code.replace("(root/'task_kazakhstan/exhibits')","(data/'exhibits')")
code=code.replace('data.append((p.stem,f,target,valid,m,px,py))','dataset.append((p.stem,f,target,valid,m,px,py))')
code=code.replace('=data[i];pr=', '=dataset[i];pr=')
code=code.replace('def infer(root,out):\n model=joblib.load(out/\'model.joblib\');','def infer(data,out,model_path):\n model=joblib.load(model_path);')
code=code.replace("(root/'task_kazakhstan/albums')","(data/'albums')")
code=code[:code.index('\ndef main():')]+'''
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--data',type=Path,required=True,help='Dataset root containing albums/ and exhibits/')
 ap.add_argument('--output',type=Path,default=Path(__file__).parent/'outputs')
 ap.add_argument('--model',type=Path,default=Path(__file__).parent/'models/detector.joblib')
 ap.add_argument('--train',action='store_true',help='Run grouped validation and train a new model in the output directory')
 args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2)
 if args.train:train(args.data,args.output)
 else:infer(args.data,args.output,args.model)

if __name__=='__main__':main()
'''
(DEST/'solve.py').write_text(code)
shutil.copy2(ROOT/'jpeg/model.joblib',DEST/'models/detector.joblib')
(DEST/'requirements.txt').write_text('numpy==2.5.4\nopencv-python-headless==5.0.0.93\nscikit-learn==1.9.1\njoblib==1.6.0\n')
shutil.copy2(ROOT/'task_kazakhstan/baseline_solution.py',DEST/'baseline_solution.py')
print(DEST)
