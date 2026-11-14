import argparse, csv, hashlib, json, random, time
from pathlib import Path
from common import ROOT, np, plt, write_csv, load_runtime

def metrics(rows, predictions):
    actual = np.array([int(row['count']) for row in rows])
    predicted = np.array([predictions[row['image']] for row in rows])
    errors = predicted - actual
    return dict(images=len(rows), mae=float(np.mean(np.abs(errors))), rmse=float(np.sqrt(np.mean(errors ** 2))), bias=float(np.mean(errors)), exact=int(np.sum(errors == 0)), within_2=int(np.sum(np.abs(errors) <= 2)), within_5=int(np.sum(np.abs(errors) <= 5)), true_total=int(actual.sum()), predicted_total=int(predicted.sum()))

def main():
    parser = argparse.ArgumentParser(description='Evaluate ICAERUS sheep counting on the supplied labelled images, without training.')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--fresh', action='store_true', help='Run inference again instead of using saved detections.')
    args = parser.parse_args()
    if(args.batch < 1 or args.imgsz < 32):
        parser.error('--batch must be positive and --imgsz must be at least 32.')
    torch, ultralytics, YOLO = load_runtime()
    output = ROOT / 'evaluation' / f'imgsz_{args.imgsz}'
    output.mkdir(parents=True, exist_ok=True)
    model_path = ROOT / 'models' / 'icaerus_yolov8m_v4' / 'yolov8m_all_finetune.pt'
    data = ROOT / 'task_mongolia'
    rows = list(csv.DictReader((data / 'train_counts.csv').open(newline='', encoding='utf-8-sig')))
    assert(len(rows) == 280 and len({row['image'] for row in rows}) == 280)
    shuffled = rows.copy()
    random.Random(2026).shuffle(shuffled)
    calibration, holdout = shuffled[:210], shuffled[210:]
    split = {row['image']: 'calibration' for row in calibration} | {row['image']: 'holdout' for row in holdout}
    cache = output / 'detections.json'
    device = '0' if(torch.cuda.is_available()) else 'cpu'
    if(cache.exists() and not args.fresh):
        detections = json.loads(cache.read_text())
    else:
        model = YOLO(str(model_path))
        sheep_classes = [int(key) for key, value in model.names.items() if('sheep' in value.lower())]
        if(not sheep_classes):
            raise ValueError(f'Model has no sheep class: {model.names}')
        print(json.dumps(dict(device=device, torch=torch.__version__, ultralytics=ultralytics.__version__, classes=model.names, imgsz=args.imgsz)), flush=True)
        sources = [str(data / 'train' / row['image']) for row in rows]
        detections = {}
        started = time.perf_counter()
        for start in range(0, len(sources), args.batch):
            results = model.predict(source=sources[start:start + args.batch], imgsz=args.imgsz, conf=0.01, iou=0.7, classes=sheep_classes, max_det=1000, device=device, half=False, verbose=False, save=False)
            for result in results:
                detections[Path(result.path).name] = dict(boxes=result.boxes.xyxy.cpu().tolist(), confidence=result.boxes.conf.cpu().tolist())
            if(start % 40 == 0 or start + args.batch >= len(sources)):
                print(f'Predicted {min(start + args.batch, len(sources))}/{len(sources)} images in {time.perf_counter() - started:.1f}s', flush=True)
        cache.write_text(json.dumps(detections), encoding='utf-8')
    assert(set(detections) == {row['image'] for row in rows})
    thresholds = [round(float(value), 3) for value in np.arange(0.05, 0.801, 0.025)]
    predictions = {threshold: {name: sum(conf >= threshold for conf in item['confidence']) for name, item in detections.items()} for threshold in thresholds}
    sweep = [dict(confidence=threshold, **metrics(calibration, predictions[threshold])) for threshold in thresholds]
    selected = min(sweep, key=lambda row: (row['mae'], row['rmse'], abs(row['confidence'] - 0.25)))['confidence']
    report = dict(model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(), torch=torch.__version__, ultralytics=ultralytics.__version__, device=device, imgsz=args.imgsz, nms_iou=0.7, seed=2026, calibration_images=210, holdout_images=70, selected_confidence=selected, no_training_on_task=True, note='Confidence selected using only calibration images. Holdout is separate from threshold selection; overlap with the external model training data has not been audited.', default_all=metrics(rows, predictions[0.25]), default_holdout=metrics(holdout, predictions[0.25]), calibrated_all=metrics(rows, predictions[selected]), calibrated_holdout=metrics(holdout, predictions[selected]))
    report['count_groups'] = {label: metrics([row for row in rows if(low <= int(row['count']) < high)], predictions[selected]) for label, low, high in [('0-19', 0, 20), ('20-59', 20, 60), ('60+', 60, 10000)]}
    report['visually_flagged_labels'] = [{'image': 'train_007.jpg', 'supplied_count': 0, 'observation': 'A sheep is visible, but no sheep points are supplied.'}, {'image': 'train_107.jpg', 'supplied_count': 0, 'observation': 'A flock is visible, but no sheep points are supplied.'}]
    report['note'] += ' All reported errors use the original supplied counts, including visually flagged label issues. No labels have been changed.'
    comparisons = [dict(image=row['image'], split=split[row['image']], true_count=int(row['count']), default_count=predictions[0.25][row['image']], calibrated_count=predictions[selected][row['image']], error=predictions[selected][row['image']] - int(row['count'])) for row in rows]
    write_csv(output / 'counts_comparison.csv', comparisons)
    write_csv(output / 'confidence_calibration.csv', sweep)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    actual = [row['true_count'] for row in comparisons]
    predicted = [row['calibrated_count'] for row in comparisons]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual, predicted, c=['#ec7547' if(row['split'] == 'holdout') else '#266b8c' for row in comparisons], alpha=0.7, s=25)
    maximum = max(actual + predicted) + 5
    ax.plot([0, maximum], [0, maximum], color='black', linestyle='--', linewidth=1)
    ax.set(xlabel='Supplied sheep count (includes flagged label issues)', ylabel='Predicted sheep count', title=f'ICAERUS v4 | {args.imgsz}px | confidence {selected}\nOrange: held-out photos; blue: threshold calibration')
    fig.tight_layout()
    fig.savefig(output / 'count_accuracy.png', dpi=150)
    plt.close(fig)
    truth = {}
    for point in csv.DictReader((data / 'train_points.csv').open(newline='', encoding='utf-8-sig')):
        truth.setdefault(point['image'], []).append([float(point['x']), float(point['y'])])
    ordered = sorted(comparisons, key=lambda row: abs(row['error']), reverse=True)
    selected_rows = ordered[:3] + sorted(comparisons, key=lambda row: abs(row['true_count'] - 20))[:1] + sorted(comparisons, key=lambda row: abs(row['true_count'] - 60))[:1] + [next(row for row in comparisons if(row['true_count'] == 0))]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, row in zip(axes.flat, selected_rows):
        name = row['image']
        ax.imshow(plt.imread(data / 'train' / name))
        dots = np.array(truth.get(name, []))
        if(len(dots)):
            ax.scatter(dots[:, 0], dots[:, 1], s=35, facecolors='none', edgecolors='#00e676', linewidths=1)
        boxes = [box for box, confidence in zip(detections[name]['boxes'], detections[name]['confidence']) if(confidence >= selected)]
        if(boxes):
            centres = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in boxes])
            ax.scatter(centres[:, 0], centres[:, 1], s=22, marker='+', c='#ff3dc8', linewidths=1)
        ax.set_title(f"{name}: supplied {row['true_count']}, predicted {row['calibrated_count']} ({row['error']:+d})")
        ax.axis('off')
    fig.suptitle('Green circles: supplied sheep labels | Pink crosses: model detections', fontsize=14)
    fig.tight_layout()
    fig.savefig(output / 'prediction_review.jpg', dpi=130)
    plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)

if(__name__ == '__main__'):
    main()
