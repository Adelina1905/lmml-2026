import sys,json,shutil,hashlib,csv
from pathlib import Path
ROOT=Path(__file__).parent;DEST=ROOT.parent/'lmml-2026/task_kazakhstan'
sys.path.insert(0,str(ROOT/'packages'))
import numpy as np,cv2

# Remove the prototype cache branch: every inference run reconstructs the maps
# from the JPEG inputs, preserving original phase and compression metadata.
p=DEST/'solve.py';code=p.read_text();lines=code.splitlines();start=next(i for i,s in enumerate(lines) if s=='   if True:');end=next(i for i in range(start+1,len(lines)) if lines[i].startswith('   if prob.shape=='))
lines=lines[:start]+[s[1:] for s in lines[start+1:end]]+lines[end+1:]
p.write_text('\n'.join(lines)+'\n')

out=DEST/'outputs';shutil.copytree(ROOT/'jpeg',out,dirs_exist_ok=True)
shutil.copy2(ROOT/'jpeg_probe.json',out/'training_compression_rankings.json')
for name in ['jpeg_train.log','jpeg_infer.log']:
 shutil.copy2(ROOT/name,out/name)
(DEST/'FLAG.txt').write_text('SIGMOID_SARSA\n')
manifest=[]
for f in sorted((ROOT/'task_kazakhstan').rglob('*')):
 if f.is_file():manifest.append(dict(path=f.relative_to(ROOT/'task_kazakhstan').as_posix(),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
(DEST/'input_manifest.json').write_text(json.dumps(dict(archive_sha256='be71aeb42bdf45e9f08e413ebb4173db60f7af909b104ed7cb7de4b9f5fc9f54',files=manifest),indent=2))

rows=list(csv.DictReader((out/'boxes.csv').open()));assert len(rows)==130
canvas=np.full((380,2000,3),255,np.uint8)
for ai,letter in enumerate('SARSA'):
 cv2.putText(canvas,f'Album {ai+1}: {letter}',(ai*400+110,36),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2)
 for row in rows:
  if row['album']!=f'album_{ai+1}':continue
  x=(float(row['x'])+25)/4;y=(float(row['y'])+25)/4
  color=(0,0,0) if float(row['score'])>=.5 else (0,80,220)
  cv2.circle(canvas,(round(ai*400+x),round(55+y)),4,color,-1)
 cv2.rectangle(canvas,(ai*400,50),(ai*400+399,355),(210,210,210),1)
cv2.putText(canvas,'Black: score >= 0.5. Orange: lower confidence. Coordinates selected automatically.',(20,375),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,30),1)
cv2.imwrite(str(out/'letters.png'),canvas)
(out/'pages').mkdir(exist_ok=True)
for row in rows:
 im=cv2.imread(str(ROOT/'task_kazakhstan/albums'/row['album']/(row['page']+'.jpg')))
 x,y,w,h=[int(row[k]) for k in ['x','y','w','h']]
 cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),3)
 label=f"{row['album']}/{row['page']} score={float(row['score']):.3f}"
 cv2.putText(im,label,(20,40),cv2.FONT_HERSHEY_SIMPLEX,.85,(255,255,255),2)
 cv2.imwrite(str(out/'pages'/(row['album']+'_'+row['page']+'.jpg')),cv2.resize(im,(800,600)),[cv2.IMWRITE_JPEG_QUALITY,90])
scores=np.array([float(r['score']) for r in rows]);stats=dict(pages=len(rows),albums={a:sum(r['album']==a for r in rows) for a in sorted(set(r['album'] for r in rows))},lower_confidence_pages=[dict(album=r['album'],page=r['page'],score=float(r['score'])) for r in rows if float(r['score'])<.5],flag='SIGMOID_SARSA',letters=list('SARSA'))
(out/'summary.json').write_text(json.dumps(stats,indent=2))
print(json.dumps(stats,indent=2))
