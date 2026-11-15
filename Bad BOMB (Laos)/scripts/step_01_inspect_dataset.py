"""Step 01: audit the input images and identify duplicate copies."""
# Input: images/ and the original ZIP. Output: results/ manifests, metadata, and contact sheets.
# Run this before step 02. Uses local files only; it does not predict defects.

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def file_hash(path):
    """Hash file bytes to detect exact copies and later input changes."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def contact_sheet(paths, output, columns=8, thumbnail=128):
    """Build a labeled image grid for quick visual inspection."""
    rows = math.ceil(len(paths) / columns)
    canvas = Image.new('RGB', (columns * thumbnail, rows * (thumbnail + 24)), '#eeeeee')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 13)
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            tile = source.convert('RGB')
            tile.thumbnail((thumbnail, thumbnail), Image.Resampling.LANCZOS)
        x, y = (index % columns) * thumbnail, (index // columns) * (thumbnail + 24)
        canvas.paste(tile, (x + (thumbnail - tile.width) // 2, y))
        draw.text((x + 4, y + thumbnail + 4), path.name, fill='#111111', font=font)
    canvas.save(output, quality=91)

def inspect(root):
    """Audit every image, choose canonical copies, and save inspection artifacts."""
    data = root / 'images'
    results = root / 'results'
    sheets = results / 'contact_sheets'
    sheets.mkdir(parents=True, exist_ok=True)
    paths = sorted(data.rglob('*.png'))
    if(not paths):
        raise ValueError(f'No PNG files in {data}')
    records, errors, hashes, pixel_hashes, signatures = [], [], defaultdict(list), defaultdict(list), []
    for index, path in enumerate(paths):
        try:
            with Image.open(path) as source:
                source.load()
                pixels = np.asarray(source.convert('RGB'))
                gray = source.convert('L')
                small = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
                signature = int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), 'big')
                width, height, mode = source.width, source.height, source.mode
                alpha_min, alpha_max = source.getchannel('A').getextrema() if('A' in source.getbands()) else (255, 255)
            sha = file_hash(path)
            pixel_sha = hashlib.sha256(f'{width}x{height}:'.encode() + pixels.tobytes()).hexdigest()
            luminance = np.asarray(gray, dtype=np.float32)
            center = luminance[height // 10:9 * height // 10, 2 * width // 5:3 * width // 5]
            records.append({'image_id': path.name, 'relative_path': path.relative_to(root).as_posix(), 'width': width, 'height': height, 'mode': mode, 'alpha_min': alpha_min, 'alpha_max': alpha_max, 'bytes': path.stat().st_size, 'sha256': sha, 'pixel_sha256': pixel_sha, 'mean_luminance': round(float(luminance.mean()), 3), 'center_mean_luminance': round(float(center.mean()), 3), 'contrast_std': round(float(luminance.std()), 3), 'dhash': f'{signature:016x}'})
            hashes[sha].append(path.name)
            pixel_hashes[pixel_sha].append(path.name)
            signatures.append((path.name, signature))
        except Exception as error:
            errors.append({'file': path.name, 'error': str(error)})
        if((index + 1) % 200 == 0):
            print(f'Inspected {index + 1}/{len(paths)}', flush=True)
    with (results / 'manifest.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]) if(records) else ['image_id'])
        writer.writeheader()
        writer.writerows(records)
    # Keep one canonical image per exact pixel hash; retain duplicate evidence in the audit.
    canonical = {}
    for row in sorted(records, key=lambda row: ('(' in row['image_id'], row['image_id'])):
        canonical.setdefault(row['pixel_sha256'], row)
    with (results / 'unique_manifest.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]) if(records) else ['image_id'])
        writer.writeheader()
        writer.writerows(sorted(canonical.values(), key=lambda row: row['image_id']))
    # Similar-looking hashes are review hints, not proof that two images are duplicates.
    near_duplicates = []
    for index, (name, signature) in enumerate(signatures):
        for other_name, other_signature in signatures[index + 1:]:
            distance = (signature ^ other_signature).bit_count()
            if(distance <= 3):
                near_duplicates.append({'image_a': name, 'image_b': other_name, 'dhash_distance': distance})
    (results / 'near_duplicate_candidates.json').write_text(json.dumps(near_duplicates, indent=2), encoding='utf-8')
    for start in range(0, len(paths), 64):
        contact_sheet(paths[start:start + 64], sheets / f'page_{start // 64 + 1:02}.jpg')
    sample_indices = sorted(set(round(index * (len(paths) - 1) / min(63, len(paths) - 1)) for index in range(min(64, len(paths))))) if(len(paths) > 1) else [0]
    contact_sheet([paths[index] for index in sample_indices], sheets / 'overview.jpg', thumbnail=160)
    numbers = sorted(int(match.group(1)) for path in paths if((match := re.fullmatch(r'image_(\d+)\.png', path.name))))
    number_set = set(numbers)
    archives = sorted(root.glob('*.zip'))
    summary = {'image_count': len(paths), 'brief_expected_count': 1125, 'count_difference': len(paths) - 1125, 'readable_count': len(records), 'errors': errors, 'dimensions': dict(Counter(f'{row["width"]}x{row["height"]}' for row in records)), 'modes': dict(Counter(row['mode'] for row in records)), 'total_image_bytes': sum(row['bytes'] for row in records), 'filename_min': paths[0].name, 'filename_max': paths[-1].name, 'numeric_id_min': min(numbers) if(numbers) else None, 'numeric_id_max': max(numbers) if(numbers) else None, 'missing_numeric_ids': [value for value in range(min(numbers), max(numbers) + 1) if(value not in number_set)] if(numbers) else [], 'nonmatching_filenames': [path.name for path in paths if(not re.fullmatch(r'image_(\d+)\.png', path.name))], 'exact_file_duplicate_groups': [group for group in hashes.values() if(len(group) > 1)], 'exact_pixel_duplicate_groups': [group for group in pixel_hashes.values() if(len(group) > 1)], 'near_duplicate_candidate_pairs': len(near_duplicates), 'near_duplicate_note': 'dHash distance <= 3 is only a screening heuristic; similar shell layouts can collide without being duplicates.', 'format_reference_files': [path.relative_to(root).as_posix() for path in root.rglob('train-output.txt')], 'archives': [{'filename': path.name, 'bytes': path.stat().st_size, 'sha256': file_hash(path)} for path in archives], 'contact_sheet_pages': math.ceil(len(paths) / 64), 'software': {'numpy': np.__version__, 'Pillow': Image.__version__}}
    summary.update({'unique_pixel_images': len(canonical), 'redundant_duplicate_files': len(records) - len(canonical), 'alpha_extrema_counts': dict(Counter(f'{row["alpha_min"]}:{row["alpha_max"]}' for row in records))})
    (results / 'input_metadata.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
# Command-line entry point; importing this module does not start a run.
if(__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Audit the shell images and generate labeled contact sheets; no predictions are made.')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    inspect(parser.parse_args().root.resolve())
