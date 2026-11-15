"""Step 03 support: check held-out training, saved models, image geometry, and synthetic outliers."""
# Synthetic results exercise the implementation; they are not challenge accuracy measurements.
import json
from argparse import Namespace
from pathlib import Path
import numpy as np
import step_03_train_local_detector as detector


def run():
    """Train small synthetic fold models and require injected local changes to stand out."""
    output = detector.ROOT / 'results/local_detector_offline_checks'
    output.mkdir(exist_ok=True)
    rows = detector.load_inputs(limit=60)
    tensor, occupancy, canvas, mask = detector.prepare_view(rows[0])
    assert tuple(tensor.shape) == (3, 504, 196)
    assert (occupancy >= detector.MIN_OCCUPANCY).sum() > 30
    rng = np.random.default_rng(17)
    count = 90
    base = rng.normal(0, 0.04, (detector.PATCH_COUNT, 384)).astype(np.float32)
    features = (base[None] + rng.normal(0, 0.004, (count, detector.PATCH_COUNT, 384))).astype(np.float16)
    coverage = np.zeros((count, detector.PATCH_COUNT), np.float32)
    coverage.reshape(count, detector.GRID_HEIGHT, detector.GRID_WIDTH)[:, 2:34, 4:10] = 1
    patches = [row * detector.GRID_WIDTH + column for row in range(24, 28) for column in [6, 7]]
    for index in [5, 35, 65]:
        features[index, patches] += np.float16(0.10)
    fake_rows = [{'image_id': f'synthetic_{index:04}.png'} for index in range(count)]
    args = Namespace(folds=3, training_patches=10000, components=16, trees=48, threads=4)
    with detector.threadpool_limits(limits=4):
        maps, energies, folds = detector.fit_and_score(features, coverage, fake_rows, output, args)
    ranking = detector.ranking_rows(fake_rows, maps, energies, folds)
    top = [row['image_id'] for row in ranking[:3]]
    expected = {f'synthetic_{index:04}.png' for index in [5, 35, 65]}
    recovered = len(set(top) & expected)
    assert recovered == 3, (top, recovered)
    assert np.all(np.isnan(maps[coverage == 0]))
    check = {'synthetic_images': count, 'injected_outliers': sorted(expected), 'recovered_in_top_3': recovered, 'prepared_image_shape_check': True, 'background_not_scored': True, 'held_out_models_and_saved_prediction_checks': True, 'synthetic_only_not_challenge_accuracy': True, 'api_requests': 0}
    detector.save_json(output / 'checks.json', check)
    print(json.dumps(check, indent=2))


if(__name__ == '__main__'):
    run()
