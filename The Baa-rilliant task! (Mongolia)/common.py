import csv, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def load_runtime():
    (ROOT / '.yolo' / 'Ultralytics').mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.yolo'))
    import torch, ultralytics
    from ultralytics import YOLO, settings
    settings.update({'sync': False, 'datasets_dir': str(ROOT / 'task_mongolia'), 'weights_dir': str(ROOT / 'models'), 'runs_dir': str(ROOT / 'runs')})
    torch.set_num_threads(4)
    return torch, ultralytics, YOLO

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
