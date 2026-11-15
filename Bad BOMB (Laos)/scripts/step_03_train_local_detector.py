"""Step 03: fit local patch anomaly detectors and rank the unlabeled shell images."""
# DINOv2 extracts frozen visual features. PCA and Isolation Forest are fitted on this dataset.
# Each image is scored by a fold model that excludes that image from all detector fitting.
import argparse
import csv
import hashlib
import html
import io
import json
import math
import sys
import time
from pathlib import Path
import cv2
import joblib
import numpy as np
import sklearn
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
BACKBONE = 'dinov2_vits14'
BACKBONE_REVISION = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
PATCH = 14
WIDTH, HEIGHT = 196, 504
GRID_HEIGHT, GRID_WIDTH = HEIGHT // PATCH, WIDTH // PATCH
PATCH_COUNT = GRID_HEIGHT * GRID_WIDTH
MIN_OCCUPANCY = 0.65
SEED = 20261115


def sha256(data):
    """Fingerprint inputs and settings so incompatible caches cannot be mixed."""
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    """Replace a readable progress or metadata file atomically."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def progress(output, phase, completed, total, started, note=''):
    """Show counts, elapsed time, and a phase-specific ETA in the terminal and status file."""
    elapsed = time.perf_counter() - started
    eta = elapsed * (total - completed) / completed if(completed) else None
    state = {'phase': phase, 'completed_in_phase': completed, 'total_in_phase': total, 'elapsed_seconds': round(elapsed, 2), 'eta_phase_seconds': round(eta, 2) if(eta is not None) else None, 'message': note, 'all_labels_unverified': True}
    save_json(output / 'status.json', state)
    eta_text = f'{eta:.0f}s' if(eta is not None) else 'estimating'
    line = f'{phase}: {completed}/{total} | elapsed {elapsed:.0f}s | phase ETA {eta_text} | {note}'
    print(line, flush=True)
    with (output / 'progress.log').open('a', encoding='utf-8') as handle:
        handle.write(line + '\n')


def load_inputs(limit=0):
    """Join the unique audit and prepared-image manifests without guessing a test split."""
    with (ROOT / 'results/unique_manifest.csv').open(newline='', encoding='utf-8') as handle:
        audited = {row['image_id']: row for row in csv.DictReader(handle)}
    with (ROOT / 'processed/manifest.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if(len(rows) != len(audited) or {row['image_id'] for row in rows} != set(audited)):
        raise ValueError('Prepared and unique image manifests do not match.')
    rows.sort(key=lambda row: row['image_id'])
    if(limit):
        rows = [rows[index] for index in np.linspace(0, len(rows) - 1, min(limit, len(rows)), dtype=int)]
    for row in rows:
        if(Path(row['image_id']).name != row['image_id']):
            raise ValueError('Invalid image identifier.')
        row['audited_source_sha256'] = audited[row['image_id']]['sha256']
        row['crop_sha256'] = sha256((ROOT / 'processed' / row['crop_path']).read_bytes())
        row['mask_sha256'] = sha256((ROOT / 'processed' / row['mask_path']).read_bytes())
    if(len({row['audited_source_sha256'] for row in rows}) != len(rows)):
        raise ValueError('Exact duplicate images must not enter detector training.')
    return rows


def prepare_view(row):
    """Preserve the crop proportions and pad to a fixed grid of 14-pixel patches."""
    crop_path = ROOT / 'processed' / row['crop_path']
    mask_path = ROOT / 'processed' / row['mask_path']
    with Image.open(crop_path) as source:
        crop = source.convert('RGB')
    with Image.open(mask_path) as source:
        bounds = tuple(int(row[key]) for key in ['crop_left', 'crop_top', 'crop_right', 'crop_bottom'])
        mask = source.convert('L').crop(bounds)
    if(crop.size != mask.size):
        raise ValueError('Prepared crop and mask dimensions differ.')
    scale = min((WIDTH - 14) / crop.width, (HEIGHT - 14) / crop.height)
    size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
    offset = ((WIDTH - size[0]) // 2, (HEIGHT - size[1]) // 2)
    canvas = Image.new('RGB', (WIDTH, HEIGHT), (127, 127, 127))
    canvas.paste(crop.resize(size, Image.Resampling.LANCZOS), offset)
    mask_canvas = Image.new('L', (WIDTH, HEIGHT), 0)
    mask_canvas.paste(mask.resize(size, Image.Resampling.NEAREST), offset)
    occupancy = np.asarray(mask_canvas, dtype=np.float32).reshape(GRID_HEIGHT, PATCH, GRID_WIDTH, PATCH).mean(axis=(1, 3)).reshape(-1) / 255
    pixels = np.asarray(canvas, dtype=np.float32) / 255
    pixels = (pixels - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy()), occupancy, canvas, mask_canvas


def extract_features(rows, output, args):
    """Cache frozen DINOv2 patch descriptors; this stage does not train neural weights."""
    cache = output / 'feature_cache'
    cache.mkdir(exist_ok=True)
    settings = {'backbone': BACKBONE, 'revision': BACKBONE_REVISION, 'width': WIDTH, 'height': HEIGHT, 'patch': PATCH, 'normalization': 'ImageNet RGB; L2-normalized final-layer patch tokens; fp16 cache', 'preprocessing_version': 1}
    signature = sha256(json.dumps(settings, sort_keys=True).encode())
    features, occupancies, pending = [None] * len(rows), [None] * len(rows), []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        fingerprint = sha256((signature + row['crop_sha256'] + row['mask_sha256']).encode())
        path = cache / (Path(row['image_id']).stem + '.npz')
        if(path.exists()):
            with np.load(path, allow_pickle=False) as saved:
                if(str(saved['fingerprint']) == fingerprint):
                    features[index], occupancies[index] = saved['features'], saved['occupancy']
                    continue
        pending.append((index, row, path, fingerprint))
    progress(output, 'feature cache', len(rows) - len(pending), len(rows), started, f'{len(pending)} images need extraction')
    model = None
    if(pending):
        if(args.device == 'cuda' and not torch.cuda.is_available()):
            raise RuntimeError('CUDA is unavailable. Check the GPU installation or explicitly choose --device cpu.')
        torch.hub.set_dir(str(ROOT / 'models/torch_hub'))
        model = torch.hub.load('facebookresearch/dinov2:' + BACKBONE_REVISION, BACKBONE, pretrained=True, trust_repo=True, skip_validation=True)
        model = model.eval().to(args.device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        mean_parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f'Frozen encoder: {BACKBONE}; {mean_parameters:,} parameters; device={args.device}', flush=True)
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        prepared = [prepare_view(row) for _, row, _, _ in batch]
        tensors = torch.stack([item[0] for item in prepared]).to(args.device)
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=args.device == 'cuda', dtype=torch.float16):
            tokens = model.forward_features(tensors)['x_norm_patchtokens']
        normalized = torch.nn.functional.normalize(tokens.float(), dim=-1).cpu().numpy().astype(np.float16)
        if(normalized.shape[1:] != (PATCH_COUNT, 384)):
            raise ValueError('Unexpected backbone feature dimensions.')
        for offset, (index, row, path, fingerprint) in enumerate(batch):
            features[index], occupancies[index] = normalized[offset], prepared[offset][1]
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as handle:
                np.savez(handle, features=features[index], occupancy=occupancies[index], fingerprint=np.array(fingerprint))
            temporary.replace(path)
        progress(output, 'feature extraction', len(rows) - len(pending) + min(start + len(batch), len(pending)), len(rows), started, 'frozen model; no API calls')
    del model
    if(args.device == 'cuda'):
        torch.cuda.empty_cache()
    matrix, occupancy = np.stack(features), np.stack(occupancies)
    if(not np.isfinite(matrix).all() or np.any((occupancy >= MIN_OCCUPANCY).sum(axis=1) < 30)):
        raise ValueError('Invalid features or insufficient foreground patch coverage.')
    save_json(output / 'feature_settings.json', {**settings, 'signature': signature, 'images': len(rows), 'feature_shape': list(matrix.shape), 'foreground_patch_count': int((occupancy >= MIN_OCCUPANCY).sum()), 'extraction_seconds': round(time.perf_counter() - started, 3)})
    return matrix, occupancy


def fit_spatial_statistics(features, valid, training):
    """Learn robust location-specific centers and scales from training images only."""
    centers, scales = np.zeros((PATCH_COUNT, features.shape[-1]), np.float32), np.ones((PATCH_COUNT, features.shape[-1]), np.float32)
    supported = np.zeros(PATCH_COUNT, bool)
    for patch in range(PATCH_COUNT):
        selection = training[valid[training, patch]]
        if(len(selection) < 30):
            continue
        values = features[selection, patch].astype(np.float32)
        centers[patch] = np.median(values, axis=0)
        scales[patch] = np.maximum(np.median(np.abs(values - centers[patch]), axis=0) * 1.4826, 0.005)
        supported[patch] = True
    return centers, scales, supported


def residuals(features, image_indices, patch_indices, centers, scales):
    """Express each patch relative to typical appearance at the same aligned position."""
    return np.clip((features[image_indices, patch_indices].astype(np.float32) - centers[patch_indices]) / scales[patch_indices], -12, 12)


def descriptors(values, patches, occupancy, projection):
    """Retain learned PCA coordinates, residual energy, spatial location, and mask coverage."""
    coordinates = np.column_stack((patches % GRID_WIDTH / (GRID_WIDTH - 1), patches // GRID_WIDTH / (GRID_HEIGHT - 1)))
    return np.column_stack((projection.transform(values), np.sqrt(np.mean(values * values, axis=1)), coordinates, occupancy)).astype(np.float32)


def robust_calibration(values, minimum_scale):
    """Learn a robust score center and scale from the detector training sample."""
    center = float(np.median(values))
    scale = max(float(np.median(np.abs(values - center))) * 1.4826, minimum_scale)
    return center, scale


def score_images(features, occupancy, indices, pack):
    """Apply a fitted fold model without fitting on any of the scored images."""
    valid = (occupancy[indices] >= MIN_OCCUPANCY) & pack['supported'][None, :]
    local_images, patches = np.nonzero(valid)
    values = residuals(features, indices[local_images], patches, pack['centers'], pack['scales'])
    vectors = descriptors(values, patches, occupancy[indices[local_images], patches], pack['projection'])
    # Preserve off-projection deviations: a forest can miss a rare direction among many coordinates.
    forest_scores = -pack['forest'].score_samples(vectors)
    residual_energy = np.sqrt(np.mean(values * values, axis=1))
    forest_center, forest_scale = pack['forest_calibration']
    energy_center, energy_scale = pack['energy_calibration']
    forest_z = np.maximum(0, (forest_scores - forest_center) / forest_scale)
    energy_z = np.maximum(0, (np.log1p(residual_energy) - energy_center) / energy_scale)
    scores = np.full((len(indices), PATCH_COUNT), np.nan, np.float32)
    scores[local_images, patches] = (forest_z + energy_z) * 0.5
    energy = np.full_like(scores, np.nan)
    energy[local_images, patches] = residual_energy
    return scores, energy


def fit_and_score(features, occupancy, rows, output, args):
    """Fit PCA and Isolation Forest on separate image folds, save models, and score held-out images."""
    valid = occupancy >= MIN_OCCUPANCY
    maps, energies = np.full(valid.shape, np.nan, np.float32), np.full(valid.shape, np.nan, np.float32)
    fold_ids, checks = np.full(len(rows), -1, np.int16), []
    models = output / 'models'
    models.mkdir(exist_ok=True)
    started = time.perf_counter()
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for fold, (training, held_out) in enumerate(splitter.split(rows)):
        generator = np.random.default_rng(SEED + fold)
        calibration = generator.choice(training, max(12, len(training) // 10), replace=False)
        fitting = np.setdiff1d(training, calibration)
        progress(output, 'detector training', fold, args.folds, started, f'fold {fold + 1}: {len(fitting)} fitting; {len(calibration)} calibration; {len(held_out)} held out')
        if(set(training) & set(held_out)):
            raise ValueError('Training and held-out image folds overlap.')
        centers, scales, supported = fit_spatial_statistics(features, valid, fitting)
        local_images, patches = np.nonzero(valid[fitting] & supported[None, :])
        sample = generator.choice(len(patches), min(args.training_patches, len(patches)), replace=False)
        sample_images, sample_patches = fitting[local_images[sample]], patches[sample]
        values = residuals(features, sample_images, sample_patches, centers, scales)
        projection = PCA(n_components=args.components, svd_solver='randomized', whiten=True, random_state=SEED + fold)
        projection.fit(values)
        vectors = descriptors(values, sample_patches, occupancy[sample_images, sample_patches], projection)
        forest = IsolationForest(n_estimators=args.trees, max_samples=512, contamination='auto', n_jobs=args.threads, random_state=SEED + fold)
        forest.fit(vectors)
        pack = {'centers': centers, 'scales': scales, 'supported': supported, 'projection': projection, 'forest': forest, 'training_indices': fitting, 'calibration_indices': calibration, 'held_out_indices': held_out, 'training_image_ids': [rows[index]['image_id'] for index in fitting], 'calibration_image_ids': [rows[index]['image_id'] for index in calibration], 'held_out_image_ids': [rows[index]['image_id'] for index in held_out], 'feature_backbone': BACKBONE, 'feature_revision': BACKBONE_REVISION, 'foreground_threshold': MIN_OCCUPANCY, 'grid_size': (GRID_HEIGHT, GRID_WIDTH), 'schema_version': 2}
        # Calibrate on separate images; in-sample forest scores can make held-out scores look inflated.
        calibration_images, calibration_patches = np.nonzero(valid[calibration] & supported[None, :])
        calibration_sample = generator.choice(len(calibration_patches), min(20000, len(calibration_patches)), replace=False)
        calibration_images, calibration_patches = calibration[calibration_images[calibration_sample]], calibration_patches[calibration_sample]
        calibration_values = residuals(features, calibration_images, calibration_patches, centers, scales)
        calibration_vectors = descriptors(calibration_values, calibration_patches, occupancy[calibration_images, calibration_patches], projection)
        pack['forest_calibration'] = robust_calibration(-forest.score_samples(calibration_vectors), 0.01)
        pack['energy_calibration'] = robust_calibration(np.log1p(np.sqrt(np.mean(calibration_values * calibration_values, axis=1))), 0.02)
        path = models / f'fold_{fold + 1}.joblib'
        joblib.dump(pack, path, compress=3)
        for start in range(0, len(held_out), 32):
            selection = held_out[start:start + 32]
            maps[selection], energies[selection] = score_images(features, occupancy, selection, pack)
        fold_ids[held_out] = fold
        # Check the saved artifact, not just the in-memory estimator.
        restored = joblib.load(path)
        repeated, _ = score_images(features, occupancy, held_out[:1], restored)
        np.testing.assert_allclose(repeated[0], maps[held_out[0]], atol=1e-7, equal_nan=True)
        checks.append({'fold': fold + 1, 'fitting_images': len(fitting), 'calibration_images': len(calibration), 'held_out_images': len(held_out), 'sampled_training_patches': len(sample), 'supported_grid_positions': int(supported.sum()), 'learned_pca_variance_fraction': float(projection.explained_variance_ratio_.sum()), 'training_held_out_overlap': 0, 'calibration_held_out_overlap': len(set(calibration) & set(held_out)), 'fitting_calibration_overlap': len(set(fitting) & set(calibration)), 'saved_model_prediction_check': True})
        progress(output, 'detector training', fold + 1, args.folds, started, f'saved fold {fold + 1}; held-out predictions checked')
    if(np.any(fold_ids < 0) or np.any(np.isfinite(maps).sum(axis=1) < 30)):
        raise ValueError('Some images lack valid held-out anomaly scores.')
    np.savez(output / 'patch_scores.npz', scores=maps, residual_energy=energies, occupancy=occupancy, fold_ids=fold_ids, image_ids=np.array([row['image_id'] for row in rows]))
    save_json(output / 'training_checks.json', {'checks': checks, 'training_seconds': round(time.perf_counter() - started, 3), 'labels_used': 0, 'encoder_weights_updated': False, 'pca_and_isolation_forest_fitted': True, 'every_image_scored_by_a_model_that_excluded_it': True, 'not_an_accuracy_test': True})
    return maps, energies, fold_ids


def ranking_rows(rows, maps, energies, fold_ids):
    """Rank images by the mean anomaly score of their most unusual one percent of shell patches."""
    ranking = []
    for index, row in enumerate(rows):
        valid = np.isfinite(maps[index])
        values = maps[index, valid]
        count = max(1, math.ceil(len(values) * 0.01))
        peak = int(np.nanargmax(maps[index]))
        record = {'image_id': row['image_id'], 'anomaly_score': float(np.sort(values)[-count:].mean()), 'maximum_patch_score': float(values.max()), 'top_5_percent_score': float(np.sort(values)[-max(1, math.ceil(len(values) * 0.05)):].mean()), 'residual_energy_top_1_percent': float(np.sort(energies[index, valid])[-count:].mean()), 'foreground_scored_patches': int(valid.sum()), 'peak_grid_x': peak % GRID_WIDTH, 'peak_grid_y': peak // GRID_WIDTH, 'held_out_fold': int(fold_ids[index]) + 1, 'ground_truth_status': 'unverified'}
        ranking.append(record)
    ranking.sort(key=lambda row: (-row['anomaly_score'], row['image_id']))
    return [{'rank': index + 1, **row} for index, row in enumerate(ranking)]


def save_csv(path, rows):
    """Write a stable, human-readable result table."""
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_visuals(rows, ranking, maps, output):
    """Render original/crop/heatmap comparisons using a shared color scale."""
    folder = output / 'heatmaps'
    folder.mkdir(exist_ok=True)
    index_by_id = {row['image_id']: index for index, row in enumerate(rows)}
    metadata = {row['image_id']: row for row in rows}
    low, high = np.nanpercentile(maps, [2, 99])
    high = max(high, low + 1e-6)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    previews = {}
    for record in ranking[:60]:
        name = record['image_id']
        _, _, canvas, mask = prepare_view(metadata[name])
        values = np.nan_to_num(maps[index_by_id[name]].reshape(GRID_HEIGHT, GRID_WIDTH), nan=float(low))
        values = cv2.resize(values, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        intensity = np.clip((values - low) / (high - low), 0, 1)
        colors = cv2.cvtColor(cv2.applyColorMap((intensity * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        alpha = (np.asarray(mask, dtype=np.float32) / 255 * 0.50)[:, :, None]
        overlay = Image.fromarray(np.uint8(np.asarray(canvas) * (1 - alpha) + colors * alpha))
        overlay.save(folder / name)
        with Image.open(ROOT / metadata[name]['source_path']) as source:
            original = ImageOps.contain(source.convert('RGB'), (220, 252))
        tile = Image.new('RGB', (428, 292), '#f2f2f2')
        tile.paste(original, (0, 30))
        tile.paste(canvas.resize((98, 252), Image.Resampling.LANCZOS), (226, 30))
        tile.paste(overlay.resize((98, 252), Image.Resampling.LANCZOS), (328, 30))
        draw = ImageDraw.Draw(tile)
        draw.text((6, 6), f'#{record["rank"]} {name}  score={record["anomaly_score"]:.4f}', font=font, fill='#111111')
        previews[name] = tile
    for start, label in [(0, 'top_28'), (28, 'ranks_29_to_56')]:
        selected = ranking[start:start + 28]
        if(not selected):
            continue
        sheet = Image.new('RGB', (428 * 4, 42 + math.ceil(len(selected) / 4) * 292), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 12), 'PROVISIONAL: original | prepared view | anomaly overlay. Red = higher score; no verified labels.', font=font, fill='#111111')
        for offset, record in enumerate(selected):
            sheet.paste(previews[record['image_id']], ((offset % 4) * 428, 42 + (offset // 4) * 292))
        sheet.save(output / (label + '.jpg'), quality=94)
    cards = []
    for record in ranking[:60]:
        name = html.escape(record['image_id'])
        cards.append(f'<article><h3>#{record["rank"]} {name}</h3><p>Score {record["anomaly_score"]:.4f}; held-out fold {record["held_out_fold"]}</p><div><a href="../../images/{name}"><img src="../../images/{name}" alt="Original {name}"></a><img class="heat" src="heatmaps/{name}" alt="Anomaly overlay {name}"></div></article>')
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Bad BOMB: local detector review</title><style>body{font:16px system-ui;margin:24px;background:#f3f5f7;color:#17212b}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}article{background:white;padding:16px;border-radius:10px}article div{display:flex;align-items:center;gap:12px}img{width:230px;max-width:100%}.heat{width:98px}p{line-height:1.5}</style><h1>Local detector: provisional candidates</h1><p>DINOv2 frozen patch features + fitted PCA and Isolation Forest. Each image was excluded from the detector that scored it. Scores are relative outlier ratings, not probabilities or verified defect labels. Red marks more unusual shell patches. Mask boundaries and lighting can still cause false alarms.</p><p><a href="ranking.csv">All image rankings</a> · <a href="provisional_top28.txt">Provisional 28 filenames</a> · <a href="summary.json">Run details</a></p><main>' + ''.join(cards) + '</main></html>'
    (output / 'review.html').write_text(document, encoding='utf-8')
    return {'heatmap_scale_low': float(low), 'heatmap_scale_high': float(high), 'heatmaps_rendered': min(60, len(ranking))}




def run(args):
    """Extract features, train the dataset-specific detector, and export review artifacts."""
    output = (ROOT / 'results' / args.run_name).resolve()
    if(output.parent != (ROOT / 'results').resolve()):
        raise ValueError('Run name must name one folder directly inside results/.')
    if(args.status):
        print((output / 'status.json').read_text(encoding='utf-8') if((output / 'status.json').exists()) else 'No saved local run yet.')
        return
    if(args.folds < 2 or args.batch_size < 1 or args.components < 2 or args.trees < 16):
        raise ValueError('Invalid detector or extraction settings.')
    output.mkdir(exist_ok=True)
    started = time.perf_counter()
    rows = load_inputs(args.limit)
    if(len(rows) < max(90, args.folds * 30)):
        raise ValueError('This baseline requires at least 90 unique images for fitting, calibration, and held-out scoring.')
    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    configuration = {**vars(args), 'backbone': BACKBONE, 'backbone_revision': BACKBONE_REVISION, 'seed': SEED, 'canvas': [WIDTH, HEIGHT], 'grid': [GRID_WIDTH, GRID_HEIGHT], 'min_occupancy': MIN_OCCUPANCY, 'input_fingerprint': sha256(json.dumps(rows, sort_keys=True).encode()), 'image_ids': [row['image_id'] for row in rows], 'method': 'Custom spatially normalized DINOv2 patch features + cross-fitted PCA/Isolation Forest and residual energy; equal-weight positive robust score deviations calibrated on separate images inside each training fold. Not an official AnomalyDINO implementation.', 'script_sha256': sha256(Path(__file__).read_bytes())}
    save_json(output / 'configuration.json', configuration)
    with threadpool_limits(limits=args.threads):
        features, occupancy = extract_features(rows, output, args)
        maps, energies, fold_ids = fit_and_score(features, occupancy, rows, output, args)
    ranking = ranking_rows(rows, maps, energies, fold_ids)
    save_csv(output / 'ranking.csv', ranking)
    selected = [row['image_id'] for row in ranking[:28]]
    if(len(selected) != 28 or len(set(selected)) != 28):
        raise ValueError('Provisional selection must contain exactly 28 unique identifiers.')
    (output / 'provisional_top28.txt').write_text('\n'.join(selected) + '\n', encoding='utf-8')
    visual_settings = make_visuals(rows, ranking, maps, output)
    weights = ROOT / 'models/torch_hub/checkpoints/dinov2_vits14_pretrain.pth'
    summary = {'phase': 'complete', 'images_ranked': len(rows), 'actual_training': 'Learned spatial median/MAD statistics, PCA projections, and Isolation Forest trees in each held-out fold.', 'encoder_weights_updated': False, 'training_labels_used': 0, 'folds': args.folds, 'trees_per_fold': args.trees, 'pca_components_per_fold': args.components, 'every_image_held_out_from_its_detector': True, 'anomaly_score_is_probability': False, 'accuracy_available': False, 'all_labels_unverified': True, 'provisional_selection_count': len(selected), 'elapsed_seconds': round(time.perf_counter() - started, 3), 'software': {'torch': torch.__version__, 'sklearn': sklearn.__version__, 'numpy': np.__version__, 'Pillow': Image.__version__}, 'device': torch.cuda.get_device_name() if(args.device == 'cuda') else 'CPU', 'encoder_weight_sha256': sha256(weights.read_bytes()) if(weights.exists()) else None, 'api_calls': 0, 'api_cost_usd': 0, 'input_limitations': '1225 unique supplied images versus 1125 in brief; train-output.txt missing. No submission or evaluator score.', 'top_28': selected, **visual_settings}
    save_json(output / 'summary.json', summary)
    progress(output, 'complete', len(rows), len(rows), started, 'saved trained fold models, ranking, provisional 28, and heatmaps; accuracy unknown')
    print(json.dumps(summary, indent=2), flush=True)


if(__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Train a local unsupervised detector on frozen image patch features and score each image using a held-out detector.')
    parser.add_argument('--run-name', default='local_detector_v1')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--components', type=int, default=32)
    parser.add_argument('--trees', type=int, default=256)
    parser.add_argument('--training-patches', type=int, default=60000)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--limit', type=int, default=0, help='Use evenly spaced images for a small diagnostic run; zero uses all unique images.')
    parser.add_argument('--status', action='store_true', help='Read saved status without training or downloading model files.')
    options = parser.parse_args()
    try:
        run(options)
    except Exception as error:
        print(f'Local detector stopped: {error}', file=sys.stderr, flush=True)
        raise
