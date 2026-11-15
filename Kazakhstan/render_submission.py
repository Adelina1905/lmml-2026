"""Rebuild presentation images from saved automatic boxes; no detection occurs here."""
import sys,csv,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'packages'))
import cv2,numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--word',default='');args=ap.parse_args()
 rows=list(csv.DictReader((args.output/'boxes.csv').open()));canvas=np.full((380,2000,3),255,np.uint8)
 for ai in range(5):
  label=f'Album {ai+1}'+(f': {args.word[ai]}' if len(args.word)==5 else '')
  cv2.putText(canvas,label,(ai*400+110,36),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2)
  for row in rows:
   if row['album']!=f'album_{ai+1}':continue
   x=(float(row['x'])+25)/4;y=(float(row['y'])+25)/4;color=(0,0,0) if float(row['score'])>=.5 else (0,80,220)
   cv2.circle(canvas,(round(ai*400+x),round(55+y)),4,color,-1)
  cv2.rectangle(canvas,(ai*400,50),(ai*400+399,355),(210,210,210),1)
 cv2.putText(canvas,'Black: score >= 0.5. Orange: lower confidence. Coordinates selected automatically.',(20,375),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,30),1)
 cv2.imwrite(str(args.output/'letters.png'),canvas);(args.output/'pages').mkdir(exist_ok=True)
 for row in rows:
  im=cv2.imread(str(args.data/'albums'/row['album']/(row['page']+'.jpg')));x,y,w,h=[int(row[k]) for k in ['x','y','w','h']]
  cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),3)
  label=f"{row['album']}/{row['page']} score={float(row['score']):.3f}"
  cv2.putText(im,label,(20,40),cv2.FONT_HERSHEY_SIMPLEX,.85,(255,255,255),2)
  cv2.imwrite(str(args.output/'pages'/(row['album']+'_'+row['page']+'.jpg')),cv2.resize(im,(800,600)),[cv2.IMWRITE_JPEG_QUALITY,90])

if __name__=='__main__':main()
