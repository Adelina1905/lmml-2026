import sys,json,csv,hashlib
from pathlib import Path
ROOT=Path(__file__).parent;DEST=ROOT.parent/'lmml-2026/task_kazakhstan'
sys.path.insert(0,str(ROOT/'packages'));sys.path.insert(0,str(DEST))
import cv2,numpy as np,joblib
from jpeg_utils import D

original=ROOT/'jpeg';reproduced=ROOT/'reproduce_outputs'
assert (original/'boxes.csv').read_bytes()==(reproduced/'boxes.csv').read_bytes(),'Box reproduction differs'
assert (original/'compression.json').read_bytes()==(reproduced/'compression.json').read_bytes(),'Compression estimates differ'
assert (original/'overlay.png').read_bytes()==(reproduced/'overlay.png').read_bytes(),'Overlay reproduction differs'
assert json.loads((original/'validation.json').read_text())==json.loads((ROOT/'reproduce_train/validation.json').read_text()),'Training validation differs'
rows=list(csv.DictReader((DEST/'outputs/boxes.csv').open()));assert len(rows)==130
assert all(sum(r['album']==f'album_{a}' for r in rows)==26 for a in range(1,6))
assert len(json.loads((original/'compression.json').read_text()))==387
for r in rows:
 x,y,w,h=[int(r[k]) for k in ['x','y','w','h']];assert 0<=x<x+w<=1600 and 0<=y<y+h<=1200
 z=np.load(DEST/'outputs/maps'/(r['album']+'_'+r['page']+'.npz'));assert z['prob'].shape==(300,400)
 im=cv2.imread(str(DEST/'outputs/pages'/(r['album']+'_'+r['page']+'.jpg')));assert im.shape[:2]==(600,800)
model=joblib.load(DEST/'models/detector.joblib');assert model.n_features_in_==200
block=np.random.default_rng(2026).normal(size=(8,8)).astype(np.float32)
error=float(np.max(np.abs(D@block@D.T-cv2.dct(block))));assert error<1e-5
manifest=json.loads((DEST/'input_manifest.json').read_text())
for entry in manifest['files']:
 p=ROOT/'task_kazakhstan'/entry['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']
assert len(manifest['files'])==192
summary=dict(flag='SIGMOID_SARSA',pages=130,detected_stamp_regions=387,input_files=192,features=200,exhibit_hits=25,exhibits=30,training_reproduction='identical validation results',inference_reproduction='byte-identical boxes, compression estimates and raw overlay',dct_max_error=error,annotated_page_dimensions=[800,600],maps_dimensions=[400,300])
(DEST/'outputs/verification.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

