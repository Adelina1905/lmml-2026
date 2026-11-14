import argparse, csv, hashlib, json, subprocess, sys, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parent
MODEL_URL = 'https://raw.githubusercontent.com/ICAERUS-EU/UC3_Livestock_Monitoring/main/models/sheep_detection/yolo8m_sheep_v4/yolov8m_all_finetune.pt'
MODEL_SHA256 = 'a292c078c27641ccd88e89635059c56ab8b7bef0611326f374e585fed4894600'

def ensure_model():
    destination = ROOT / 'models' / 'icaerus_yolov8m_v4' / 'yolov8m_all_finetune.pt'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if(not destination.exists()):
        temporary = destination.with_suffix('.pt.part')
        print('Downloading the pretrained ICAERUS sheep model (52 MB).', flush=True)
        request = urllib.request.Request(MODEL_URL, headers={'User-Agent': 'lmml-mongolia-solution'})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as handle:
            while(chunk := response.read(1024 * 1024)):
                handle.write(chunk)
        if(hashlib.sha256(temporary.read_bytes()).hexdigest() != MODEL_SHA256):
            raise ValueError('Downloaded model checksum differs from the model used for this solution.')
        temporary.replace(destination)
    if(hashlib.sha256(destination.read_bytes()).hexdigest() != MODEL_SHA256):
        raise ValueError('Local model checksum differs from the model used for this solution.')

def run(script, arguments=()):
    subprocess.run([sys.executable, str(ROOT / script), *arguments], check=True, cwd=ROOT)

def main():
    parser = argparse.ArgumentParser(description='Reproduce sheep counting and the survey grid, without training a model.')
    parser.add_argument('--evaluate', action='store_true', help='Recreate the labelled-data evaluation and confidence calibration.')
    parser.add_argument('--fresh', action='store_true', help='Recompute detections rather than reuse cached model outputs.')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--word', help='Record a manually read word, e.g. RECALL. This is not automatic OCR.')
    args = parser.parse_args()
    if(args.batch < 1):
        parser.error('--batch must be positive.')
    word = args.word.strip().upper() if(args.word) else None
    if(word and (not word.isascii() or not word.isalpha())):
        parser.error('--word must contain only letters A-Z.')
    ensure_model()
    common = ['--batch', str(args.batch)] + (['--fresh'] if(args.fresh) else [])
    if(args.evaluate or not (ROOT / 'evaluation' / 'imgsz_640' / 'metrics.json').exists()):
        run('evaluate_model.py', common)
    run('predict_survey.py', common)
    run('decode_map.py')
    counts = list(csv.DictReader((ROOT / 'counts.csv').open(newline='', encoding='utf-8-sig')))
    plots = list(csv.DictReader((ROOT / 'task_mongolia' / 'survey_plots.csv').open(newline='', encoding='utf-8-sig')))
    if(len(counts) != 333 or len({row['image'] for row in counts}) != 333 or {row['image'] for row in counts} != {row['image'] for row in plots} or any(int(row['count']) < 0 for row in counts)):
        raise ValueError('The counts file does not cover all 333 survey photos exactly once.')
    geometry = json.loads((ROOT / 'survey_results' / 'grid_geometry.json').read_text())
    if(geometry['grid_rows'] != 9 or geometry['grid_columns'] != 37 or geometry['occupied_cells'] != 333):
        raise ValueError('Unexpected survey grid geometry.')
    print('Verified counts.csv: 333 unique images, all present in the survey log.', flush=True)
    if(word):
        flag = 'SIGMOID_' + word
        (ROOT / 'flag.txt').write_text(flag + '\n', encoding='utf-8')
        result = dict(word=word, flag=flag, word_source='Manual reading of the noisy survey count grid; supplied with --word.', model_training=False, counts_source='Unedited ICAERUS v4 predictions at confidence 0.425, image size 640, NMS IoU 0.7.', counts_sha256=hashlib.sha256((ROOT / 'counts.csv').read_bytes()).hexdigest(), platform_verified=False, **geometry)
        (ROOT / 'survey_results' / 'solution_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(flag, flush=True)
    else:
        print('Read survey_results/decoded_grid.png and record the word with --word WORD.', flush=True)

if(__name__ == '__main__'):
    main()
