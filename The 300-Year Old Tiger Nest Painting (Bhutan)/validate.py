import json
import subprocess
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from skimage.metrics import structural_similarity
sys.dont_write_bytecode = True
from main import edge_costs, recover_order

root = Path(__file__).resolve().parent
started = time.perf_counter()
process = subprocess.run([sys.executable, str(root / 'main.py')], cwd=root, capture_output=True, text=True, timeout=20, check=True)
elapsed = time.perf_counter() - started
expected_order = json.loads((root / 'validation' / 'reference_order.json').read_text())
actual_order = json.loads((root / 'reconstructed_order.json').read_text())
input_files = {p.name for p in (root / 'input').glob('*.png')}
assert(len(set(expected_order)) == len(expected_order) and input_files == set(expected_order))
expected_image = np.concatenate([cv2.imread(str(root / 'input' / name)) for name in expected_order], axis=1)
actual_image = cv2.imread(str(root / 'reconstructed_image.png'))
report = {'training': {'strips': len(expected_order), 'order_accuracy': float(np.mean(np.array(actual_order) == np.array(expected_order))), 'pixel_exact': bool(np.array_equal(actual_image, expected_image)), 'ssim': float(structural_similarity(expected_image, actual_image, channel_axis=2, data_range=255)), 'wall_seconds_including_imports': elapsed}, 'reshredded': []}
rng = np.random.default_rng(2026)
for(width, offset) in [(6, 3), (8, 4), (12, 5), (16, 7)]:
    count = (expected_image.shape[1] - offset) // width
    strips = np.stack([expected_image[:, offset + i * width:offset + (i + 1) * width] for i in range(count)])
    permutation = rng.permutation(count)
    started = time.perf_counter()
    result = recover_order(edge_costs(strips[permutation]))
    recovered = permutation[result]
    report['reshredded'].append({'width': width, 'offset': offset, 'strips': count, 'order_accuracy': float(np.mean(recovered == np.arange(count))), 'seconds': time.perf_counter() - started})
(root / 'validation' / 'results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(process.stdout.strip())
print(json.dumps(report, indent=2))
assert(report['training']['pixel_exact'] and report['training']['order_accuracy'] == 1 and report['training']['ssim'] == 1)
assert(all(test['order_accuracy'] == 1 for test in report['reshredded']))
