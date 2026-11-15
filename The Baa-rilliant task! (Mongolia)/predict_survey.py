import argparse, csv, hashlib, json, time
from pathlib import Path
from common import ROOT, np, plt, write_csv, load_runtime

def main():
    parser = argparse.ArgumentParser(description='Count survey sheep using the evaluated ICAERUS model and draw their geographic map.')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--fresh', action='store_true', help='Run inference again instead of using saved detections.')
    args = parser.parse_args()
    if(args.batch < 1):
        parser.error('--batch must be positive.')
    torch, ultralytics, YOLO = load_runtime()
    output = ROOT / 'survey_results'
    output.mkdir(exist_ok=True)
    data = ROOT / 'task_mongolia'
    rows = list(csv.DictReader((data / 'survey_plots.csv').open(newline='', encoding='utf-8-sig')))
    assert(len(rows) == 333 and len({row['image'] for row in rows}) == 333)
    assert(all((data / 'survey' / row['image']).is_file() for row in rows))
    evaluation = json.loads((ROOT / 'evaluation' / 'imgsz_640' / 'metrics.json').read_text())
    threshold = evaluation['selected_confidence']
    model_path = ROOT / 'models' / 'icaerus_yolov8m_v4' / 'yolov8m_all_finetune.pt'
    fingerprint = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert(fingerprint == evaluation['model_sha256'])
    cache = output / 'detections.json'
    detections = json.loads(cache.read_text()) if(cache.exists() and not args.fresh) else {}
    pending = [str(data / 'survey' / row['image']) for row in rows if(row['image'] not in detections)]
    device = '0' if(torch.cuda.is_available()) else 'cpu'
    started = time.perf_counter()
    if(pending):
        model = YOLO(str(model_path))
        sheep_classes = [int(key) for key, name in model.names.items() if('sheep' in name.lower())]
        assert(sheep_classes)
        for start in range(0, len(pending), args.batch):
            results = model.predict(source=pending[start:start + args.batch], imgsz=640, conf=0.01, iou=0.7, classes=sheep_classes, max_det=1000, device=device, half=False, verbose=False, save=False)
            for result in results:
                detections[Path(result.path).name] = dict(boxes=result.boxes.xyxy.cpu().tolist(), confidence=result.boxes.conf.cpu().tolist())
            if(start % 40 == 0 or start + args.batch >= len(pending)):
                cache.write_text(json.dumps(detections), encoding='utf-8')
                print(f'Counted {len(detections)}/333 survey photos in {time.perf_counter() - started:.1f}s', flush=True)
    assert(set(detections) == {row['image'] for row in rows})
    counts = {row['image']: sum(value >= threshold for value in detections[row['image']]['confidence']) for row in rows}
    write_csv(ROOT / 'counts.csv', [dict(image=row['image'], count=counts[row['image']]) for row in rows])
    write_csv(output / 'counts_with_coordinates.csv', [dict(image=row['image'], x_m=row['x_m'], y_m=row['y_m'], count=counts[row['image']]) for row in rows])
    xy = np.array([[float(row['x_m']), float(row['y_m'])] for row in rows])
    values = np.array([counts[row['image']] for row in rows])
    fig, ax = plt.subplots(figsize=(12, 8))
    dots = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap='viridis', s=70, edgecolors='white', linewidths=0.3)
    fig.colorbar(dots, ax=ax, label='Predicted sheep count')
    ax.set(xlabel='x (metres)', ylabel='y (metres)', title='Drone survey: sheep counts by plot', aspect='equal')
    fig.tight_layout()
    fig.savefig(output / 'survey_map.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, cutoff in zip(axes.flat, [10, 20, 30, 40, 60, 80]):
        selected = values >= cutoff
        ax.scatter(xy[~selected, 0], xy[~selected, 1], c='#e2e6e9', s=18)
        ax.scatter(xy[selected, 0], xy[selected, 1], c='#153e52', s=34)
        ax.set(title=f'Count >= {cutoff} ({selected.sum()} plots)', aspect='equal')
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(output / 'survey_thresholds.png', dpi=150)
    plt.close(fig)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(xy.T))
    direction = eigenvectors[:, -1]
    if(direction[0] < 0):
        direction = -direction
    rotation = np.column_stack([direction, [-direction[1], direction[0]]])
    aligned = (xy - xy.mean(axis=0)) @ rotation
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    dots = axes[0].scatter(aligned[:, 0], aligned[:, 1], c=values, cmap='viridis', s=65)
    fig.colorbar(dots, ax=axes[0], label='Predicted count', shrink=0.8)
    axes[0].set_title('Survey rotated to its principal geographic axis')
    for ax, cutoff in zip(axes[1:], [20, 40, 60]):
        selected = values >= cutoff
        ax.scatter(aligned[~selected, 0], aligned[~selected, 1], c='#e2e6e9', s=20)
        ax.scatter(aligned[selected, 0], aligned[selected, 1], c='#153e52', s=50)
        ax.set_title(f'Predicted count >= {cutoff}')
    for ax in axes:
        ax.set_aspect('equal')
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(output / 'survey_aligned.png', dpi=150)
    plt.close(fig)
    summary = dict(images=len(rows), confidence=threshold, imgsz=640, nms_iou=0.7, model_sha256=fingerprint, torch=torch.__version__, ultralytics=ultralytics.__version__, device=device, count_min=int(values.min()), count_max=int(values.max()), count_total=int(values.sum()), count_median=float(np.median(values)), status='Unedited automatic model estimates; the word is read manually from the geographic count grid.')
    (output / 'prediction_metadata.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)

if(__name__ == '__main__'):
    main()
