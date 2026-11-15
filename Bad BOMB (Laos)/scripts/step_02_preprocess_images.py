"""Step 02: isolate, straighten, and crop each unique shell image."""
# Input: step 01 unique manifest and images/. Output: processed/ views, masks, crops, and QA.
# Keep the original views: segmentation boundaries are approximate. Uses no API.

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

def read_rgb(path):
    """Decode an image from its path into RGB pixels."""
    pixels = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if(pixels is None):
        raise ValueError(f'Cannot decode {path}')
    return cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)

def write_image(path, pixels):
    """Encode and save pixels without relying on ASCII-only file paths."""
    data = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR) if(pixels.ndim == 3) else pixels
    parameters = [cv2.IMWRITE_JPEG_QUALITY, 95] if(path.suffix == '.jpg') else [cv2.IMWRITE_PNG_COMPRESSION, 3]
    success, encoded = cv2.imencode(path.suffix, data, parameters)
    if(not success):
        raise ValueError(f'Cannot encode {path}')
    path.write_bytes(encoded.tobytes())

def largest_contour(mask):
    """Select the main connected object outline."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if(not contours):
        raise ValueError('No foreground contour')
    return max(contours, key=cv2.contourArea)

def green_seeds(rgb):
    """Find reliable green shell pixels away from the side borders."""
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    signed = rgb.astype(np.int16)
    green = ((signed[:, :, 1] - signed[:, :, 0] > 9) & (signed[:, :, 1] - signed[:, :, 2] > 12) & (hsv[:, :, 0] >= 36) & (hsv[:, :, 0] <= 90) & (hsv[:, :, 1] >= 65)).astype(np.uint8) * 255
    green[:, :width // 5] = 0
    green[:, 4 * width // 5:] = 0
    return green

def trim_frame_spurs(rgb, mask):
    """Remove stray mask regions attached to the top or bottom frame."""
    seed_rows = np.where(green_seeds(rgb) > 0)[0]
    if(not len(seed_rows)):
        return mask
    result = mask.copy()
    foreground_rows = np.where(result > 0)[0]
    if(foreground_rows.min() < 3 and seed_rows.min() >= 5):
        result[:seed_rows.min() - 2] = 0
    if(foreground_rows.max() >= mask.shape[0] - 3 and seed_rows.max() <= mask.shape[0] - 6):
        result[seed_rows.max() + 3:] = 0
    return result

def segment(rgb, iterations):
    """Use green seeds and GrabCut to estimate the complete shell mask."""
    height, width = rgb.shape[:2]
    green = green_seeds(rgb)
    closed = cv2.morphologyEx(green, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 31)))
    proposal = np.zeros(green.shape, dtype=np.uint8)
    cv2.drawContours(proposal, [largest_contour(closed)], -1, 255, cv2.FILLED)
    x, y, box_width, box_height = cv2.boundingRect(proposal)
    if(box_height < height * 0.5 or np.count_nonzero(proposal) < 10000):
        raise ValueError('Foreground proposal is implausibly small')
    # Seed GrabCut with reliable green pixels while allowing non-green band details inside the object.
    labels = np.full(green.shape, cv2.GC_BGD, dtype=np.uint8)
    labels[max(0, y - 18):min(height, y + box_height + 18), max(0, x - 18):min(width, x + box_width + 18)] = cv2.GC_PR_BGD
    labels[cv2.dilate(proposal, np.ones((5, 5), np.uint8)) > 0] = cv2.GC_PR_FGD
    core = (green > 0) & (cv2.erode(proposal, np.ones((13, 13), np.uint8)) > 0)
    if(np.count_nonzero(core) < 1000):
        raise ValueError('Insufficient reliable foreground seeds')
    labels[core] = cv2.GC_FGD
    cv2.grabCut(rgb, labels, None, np.zeros((1, 65)), np.zeros((1, 65)), iterations, cv2.GC_INIT_WITH_MASK)
    selected = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    mask = np.zeros(green.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour(selected)], -1, 255, cv2.FILLED)
    return trim_frame_spurs(rgb, mask)

def axis_tilt(mask):
    """Measure the mask long-axis angle relative to vertical."""
    ys, xs = np.nonzero(mask)
    covariance = np.cov(np.column_stack((xs, ys)), rowvar=False)
    _, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, -1]
    if(axis[1] < 0):
        axis = -axis
    return math.degrees(math.atan2(float(axis[0]), float(axis[1])))

def align(rgb, mask):
    """Straighten and center the shell, retaining both context and cropped views."""
    height, width = mask.shape
    tilt = axis_tilt(mask)
    applied_rotation = -tilt if(abs(tilt) > 0.15) else 0.0
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), applied_rotation, 1.0)
    contour = largest_contour(mask).astype(np.float64)
    transformed = cv2.transform(contour, transform).reshape(-1, 2)
    low, high = transformed.min(axis=0), transformed.max(axis=0)
    transform[:, 2] += np.array([(width - 1) / 2, (height - 1) / 2]) - (low + high) / 2
    if(applied_rotation == 0):
        transform[:, 2] = np.rint(transform[:, 2])
    mapped = cv2.transform(contour, transform).reshape(-1, 2)
    if(np.any(mapped.min(axis=0) < 3) or np.any(mapped.max(axis=0) > np.array([width - 4, height - 4]))):
        raise ValueError('Alignment would clip the silhouette')
    aligned = cv2.warpAffine(rgb, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
    aligned_mask = cv2.warpAffine(mask, transform, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Retain a narrow rim so uncertain mask edges do not erase visible shell details.
    support = cv2.dilate(aligned_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    foreground = np.where(support[:, :, None], aligned, np.uint8(127)).astype(np.uint8)
    x, y, box_width, box_height = cv2.boundingRect(aligned_mask)
    left, top, right, bottom = max(0, x - 18), max(0, y - 18), min(width, x + box_width + 18), min(height, y + box_height + 18)
    metrics = {'source_tilt_degrees': round(tilt, 6), 'applied_rotation_degrees': round(applied_rotation, 6), 'aligned_tilt_degrees': round(axis_tilt(aligned_mask), 6), 'mask_area': int(np.count_nonzero(mask)), 'aligned_mask_area': int(np.count_nonzero(aligned_mask)), 'aligned_width': box_width, 'aligned_height': box_height, 'crop_left': left, 'crop_top': top, 'crop_right': right, 'crop_bottom': bottom, 'transform': json.dumps(transform.tolist(), separators=(',', ':'))}
    return aligned, aligned_mask, foreground, foreground[top:bottom, left:right], metrics

def create_preview(root, output, rows, target, tile=192):
    """Show originals, mask outlines, aligned context, and isolated shells together."""
    headings = ['Original', 'Detected outline', 'Aligned with context', 'Isolated shell']
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    canvas = Image.new('RGB', (tile * 4, 28 + len(rows) * (tile + 24)), '#eeeeee')
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(headings):
        draw.text((column * tile + 5, 5), label, fill='#111111', font=font)
    for index, row in enumerate(rows):
        original = read_rgb(root / row['source_path'])
        source_mask = cv2.imdecode(np.fromfile(output / row['source_mask_path'], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        overlay = original.copy()
        cv2.drawContours(overlay, [largest_contour(source_mask)], -1, (255, 0, 255), 2)
        panels = [Image.fromarray(original), Image.fromarray(overlay), Image.open(output / row['aligned_path']).convert('RGB'), Image.open(output / row['foreground_path']).convert('RGB')]
        for column, panel in enumerate(panels):
            panel = ImageOps.contain(panel, (tile, tile))
            canvas.paste(panel, (column * tile, 28 + index * (tile + 24)))
            draw.text((column * tile + 4, 28 + index * (tile + 24) + tile + 3), row['image_id'], fill='#111111', font=font)
    canvas.save(target, quality=94)

def run(args):
    """Prepare each unique image and save geometry checks, manifests, and previews."""
    root = args.root.resolve()
    output = (args.output or root / 'processed').resolve()
    output.mkdir(parents=True, exist_ok=True)
    for directory in ['aligned', 'source_masks', 'masks', 'foreground', 'crops', 'qa']:
        (output / directory).mkdir(exist_ok=True)
    with (root / 'results/unique_manifest.csv').open(newline='', encoding='utf-8') as handle:
        inputs = list(csv.DictReader(handle))
    if(args.limit):
        inputs = [inputs[index] for index in np.linspace(0, len(inputs) - 1, min(args.limit, len(inputs)), dtype=int)]
    cv2.setNumThreads(args.threads)
    records, failures = [], []
    start = time.perf_counter()
    for index, row in enumerate(inputs):
        try:
            cv2.setRNGSeed(args.seed)
            source = root / row['relative_path']
            if(hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']):
                raise ValueError('Source file differs from audited input')
            rgb = read_rgb(source)
            cached_mask_path = output / 'source_masks' / row['image_id']
            mask = cv2.imdecode(np.fromfile(cached_mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE) if(args.reuse_masks and cached_mask_path.exists()) else segment(rgb, args.iterations)
            mask = trim_frame_spurs(rgb, mask)
            aligned, aligned_mask, foreground, crop, metrics = align(rgb, mask)
            basename = Path(row['image_id']).stem
            aligned_path, mask_path, foreground_path, crop_path = f'aligned/{basename}.jpg', f'masks/{basename}.png', f'foreground/{basename}.png', f'crops/{basename}.png'
            source_mask_path = f'source_masks/{basename}.png'
            for relative, pixels in [(aligned_path, aligned), (source_mask_path, mask), (mask_path, aligned_mask), (foreground_path, foreground), (crop_path, crop)]:
                write_image(output / relative, pixels)
            warnings = []
            if(not 22000 <= metrics['mask_area'] <= 65000):
                warnings.append('unusual_mask_area')
            if(not 380 <= metrics['aligned_height'] <= 490):
                warnings.append('unusual_height')
            if(not 70 <= metrics['aligned_width'] <= 170):
                warnings.append('unusual_width')
            if(abs(metrics['source_tilt_degrees']) > 15):
                warnings.append('large_rotation')
            if(abs(metrics['aligned_tilt_degrees']) > 0.15):
                warnings.append('residual_rotation')
            source_y, source_x = np.nonzero(mask)
            if(min(source_x.min(), source_y.min(), rgb.shape[1] - 1 - source_x.max(), rgb.shape[0] - 1 - source_y.max()) < 3):
                warnings.append('source_border_contact')
            if(abs(metrics['aligned_mask_area'] / metrics['mask_area'] - 1) > 0.01):
                warnings.append('mask_area_change')
            support = cv2.dilate(aligned_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
            if(not np.array_equal(read_rgb(output / foreground_path)[support], aligned[support])):
                raise ValueError('Saved foreground pixels differ from the aligned source')
            records.append({'image_id': row['image_id'], 'source_path': row['relative_path'], 'aligned_path': aligned_path, 'source_mask_path': source_mask_path, 'mask_path': mask_path, 'foreground_path': foreground_path, 'crop_path': crop_path, 'warnings': ';'.join(warnings), **metrics})
        except Exception as error:
            failures.append({'image_id': row['image_id'], 'error': str(error)})
        if((index + 1) % 100 == 0 or index + 1 == len(inputs)):
            print(f'Processed {index + 1}/{len(inputs)}; failures={len(failures)}; elapsed={time.perf_counter() - start:.1f}s', flush=True)
    fields = list(records[0]) if(records) else ['image_id']
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    if(records):
        median_area = np.median([row['mask_area'] for row in records])
        ranked_review = sorted(records, key=lambda row: (bool(row['warnings']), abs(row['source_tilt_degrees']), abs(row['mask_area'] - median_area)), reverse=True)
        review = list({row['image_id']: row for row in ranked_review[:12] + sorted(records, key=lambda row: row['mask_area'])[:6] + sorted(records, key=lambda row: row['mask_area'])[-6:] + [records[index] for index in np.linspace(0, len(records) - 1, min(24, len(records)), dtype=int)]}.values())
        for index in range(0, len(review), 6):
            create_preview(root, output, review[index:index + 6], output / f'qa/page_{index // 6 + 1:02}.jpg')
        preferred = [records[0], records[min(1, len(records) - 1)], max(records, key=lambda row: abs(row['source_tilt_degrees'])), records[len(records) // 2]]
        create_preview(root, output, preferred, root / 'results/preprocessing_preview.jpg')
    summary = {'input_count': len(inputs), 'processed_count': len(records), 'failure_count': len(failures), 'failures': failures, 'warning_count': sum(bool(row['warnings']) for row in records), 'warning_types': dict(Counter(warning for row in records for warning in row['warnings'].split(';') if(warning))), 'parameters': {'iterations': args.iterations, 'threads': args.threads, 'seed': args.seed, 'foreground_margin_pixels': 3, 'crop_margin_pixels': 18, 'alignment_deadband_degrees': 0.15, 'reuse_masks': args.reuse_masks, 'resize': False, 'brightness_normalization': False, 'background_rgb': [127, 127, 127]}, 'software': {'opencv': cv2.__version__, 'numpy': np.__version__}, 'elapsed_seconds': round(time.perf_counter() - start, 3), 'qa_note': 'Automated geometry and saved-pixel checks plus selected visual previews; segmentation accuracy needs visual review and has no labeled ground truth.'}
    for key in ['mask_area', 'aligned_width', 'aligned_height', 'source_tilt_degrees', 'aligned_tilt_degrees']:
        values = np.array([row[key] for row in records])
        if(len(values)):
            summary[key] = {'min': float(values.min()), 'median': float(np.median(values)), 'max': float(values.max())}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (root / 'results/preprocessing_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    if(failures):
        raise SystemExit(1)
# Command-line entry point; importing this module does not start a run.
if(__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Isolate and align unique shell images with classical color segmentation and GrabCut; no pretrained model or API is used.')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reuse-masks', action='store_true', help='Reuse cached source-coordinate masks, apply current border cleanup, and regenerate aligned outputs. Use only with compatible segmentation settings.')
    run(parser.parse_args())
