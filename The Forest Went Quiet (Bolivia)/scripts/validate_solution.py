import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
import scipy
import torch
ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ROOT / 'materials'
RESULTS = ROOT / 'results'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MATERIALS))
from audio_io import decode_wav_mono, validate
from challenge_model import CLASS_NAMES, load_model, predict_probabilities
from main import transform

def run_checks():
    torch.set_num_threads(2)
    RESULTS.mkdir(exist_ok=True)
    with (MATERIALS / 'recordings.csv').open(encoding='utf-8', newline='') as stream: rows = list(csv.DictReader(stream))
    audio = np.stack([decode_wav_mono(MATERIALS / row['path'])[0] for row in rows])
    for(row) in rows: assert hashlib.sha256((MATERIALS / row['path']).read_bytes()).hexdigest() == row['sha256']
    model = load_model()
    modified = transform(audio)
    edit = validate(audio, modified)
    before, after = predict_probabilities(model, audio), predict_probabilities(model, modified)
    report = {'environment': {'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__}, 'public_recordings': {'count': len(audio), 'rain_predictions': int(sum(after.argmax(1) == 3)), 'max_edit_percent': float(edit.max()), 'max_output_amplitude': float(abs(modified).max()), 'by_class': {name: {'count': sum(r['label'] == name for r in rows), 'rain_predictions': sum(int(after[i].argmax() == 3) for i, r in enumerate(rows) if r['label'] == name)} for name in CLASS_NAMES}}}
    with (RESULTS / 'local_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['clip_id', 'label', 'classification_status', 'before', 'after', 'rain_probability', 'edit_percent'])
        for(i, row) in enumerate(rows): writer.writerow([row['clip_id'], row['label'], row['classification_status'], CLASS_NAMES[int(before[i].argmax())], CLASS_NAMES[int(after[i].argmax())], float(after[i, 3]), float(edit[i])])
    refs = [i for i, row in enumerate(rows) if row['classification_status'] == 'correct']
    rng = np.random.default_rng(20261115)
    augmented, labels = [], []
    for(index) in refs:
        for(_) in range(24):
            clip = np.roll(audio[index], int(rng.integers(0, 32000))).astype(float)
            gain = min(float(10 ** rng.uniform(-.6, .3)), .94 / max(float(abs(clip).max()), 1e-12))
            augmented.append((clip * gain).astype(np.float32))
            labels.append(rows[index]['label'])
    augmented = np.stack(augmented)
    changed = np.concatenate([transform(augmented[i:i + 64]) for i in range(0, len(augmented), 64)])
    augmented_edit = validate(augmented, changed)
    success = predict_probabilities(model, changed).argmax(1) == 3
    report['shifted_and_rescaled_references'] = {'count': len(augmented), 'rain_rate': float(success.mean()), 'max_edit_percent': float(augmented_edit.max()), 'by_class': {name: float(success[np.array(labels) == name].mean()) for name in CLASS_NAMES}, 'note': 'Derived from public recordings; these are not hidden evaluation data.'}
    with np.load(MATERIALS / 'train-input.npz', allow_pickle=False) as data:
        transformed = transform(data['audio'])
        train_edit = validate(data['audio'], transformed)
        report['train_example'] = {'count': len(transformed), 'rain_predictions': int(sum(predict_probabilities(model, transformed).argmax(1) == 3)), 'max_edit_percent': float(train_edit.max())}
    temporary = tempfile.TemporaryDirectory(prefix='validation-', dir=ROOT)
    runs = Path(temporary.name)
    report['execution'] = []
    for(batch_size) in [1, 64]:
        batch = np.stack([audio[i % len(audio)] for i in range(batch_size)])
        ids = np.array([f'validation-{i:02d}' for i in range(batch_size)], dtype='<U64')
        np.savez(runs / 'test-input.npz', audio=batch, sample_rate=np.array(16000, dtype=np.int32), clip_ids=ids)
        begin = time.perf_counter()
        subprocess.run([sys.executable, str(ROOT / 'main.py')], cwd=runs, check=True, timeout=30)
        elapsed = time.perf_counter() - begin
        with np.load(runs / 'test-output.npz', allow_pickle=False) as output:
            assert set(output.files) == {'audio', 'sample_rate', 'clip_ids'}
            assert output['sample_rate'].shape == () and output['sample_rate'].dtype == np.int32 and output['sample_rate'] == 16000
            assert np.array_equal(output['clip_ids'], ids) and output['clip_ids'].dtype == ids.dtype
            validate(batch, output['audio'])
            assert np.array_equal(output['audio'], transform(batch))
        size = (runs / 'test-output.npz').stat().st_size
        assert size <= 32 * 1024 ** 2
        report['execution'].append({'batch_size': batch_size, 'seconds': elapsed, 'bytes': size, 'format_valid': True, 'deterministic': True})
    temporary.cleanup()
    stress = np.zeros((4, 32000), dtype=np.float32)
    stress[1, ::2], stress[1, 1::2] = 1, -1
    stress[2] = np.linspace(-1, 1, 32000, dtype=np.float32)
    stress[3, 16000] = 1
    validate(stress, transform(stress))
    report['edge_cases'] = 'Zero energy, full scale alternating samples, full scale ramp, and single impulse satisfy audio limits.'
    (RESULTS / 'validation_results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))

if(__name__ == '__main__'): run_checks()
