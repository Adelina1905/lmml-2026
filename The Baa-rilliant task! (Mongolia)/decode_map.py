import csv, json
from common import ROOT, np, plt, write_csv

def group_axis(values):
    ordered = np.sort(values)
    labels = np.r_[0, np.cumsum(np.diff(ordered) > 25)]
    centres = np.array([ordered[labels == index].mean() for index in range(labels.max() + 1)])
    return np.argmin(np.abs(values[:, None] - centres[None, :]), axis=1), centres

def main():
    output = ROOT / 'survey_results'
    rows = list(csv.DictReader((output / 'counts_with_coordinates.csv').open()))
    xy = np.array([[float(row['x_m']), float(row['y_m'])] for row in rows])
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(xy.T))
    direction = eigenvectors[:, -1]
    if(direction[0] < 0):
        direction = -direction
    aligned = (xy - xy.mean(axis=0)) @ np.column_stack([direction, [-direction[1], direction[0]]])
    columns, x_centres = group_axis(aligned[:, 0])
    row_indices, y_centres = group_axis(aligned[:, 1])
    height, width = len(y_centres), len(x_centres)
    grid = np.full((height, width), np.nan)
    indexed = []
    for row, column, index in zip(rows, columns, row_indices):
        top_row = height - 1 - index
        assert(np.isnan(grid[top_row, column]))
        grid[top_row, column] = int(row['count'])
        indexed.append(dict(image=row['image'], grid_row=int(top_row), grid_column=int(column), count=int(row['count'])))
    assert(np.isfinite(grid).all())
    write_csv(output / 'grid_counts.csv', indexed)
    np.savetxt(output / 'count_matrix.csv', grid, delimiter=',', fmt='%d')
    fig, axes = plt.subplots(3, 1, figsize=(16, 9))
    axes[0].imshow(grid, cmap='gray_r', vmin=15, vmax=55, interpolation='nearest')
    axes[0].set_title('Predicted count: contrast limited to 15-55 sheep')
    for ax, cutoff in zip(axes[1:], [35, 40]):
        ax.imshow(grid >= cutoff, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f'Predicted count >= {cutoff}')
    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(output / 'decoded_grid.png', dpi=150)
    plt.close(fig)
    detections = json.loads((output / 'detections.json').read_text())
    fig, axes = plt.subplots(4, 3, figsize=(18, 10))
    for axes_row, confidence in zip(axes, [0.1, 0.25, 0.425, 0.6]):
        alternate = np.zeros_like(grid)
        for row in indexed:
            alternate[row['grid_row'], row['grid_column']] = sum(value >= confidence for value in detections[row['image']]['confidence'])
        for ax, cutoff in zip(axes_row, [35, 45, 55]):
            ax.imshow(alternate >= cutoff, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(f'Confidence {confidence}; count >= {cutoff}')
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(output / 'decoding_sensitivity.png', dpi=160)
    plt.close(fig)
    geometry = dict(grid_rows=height, grid_columns=width, angle_degrees=float(np.degrees(np.arctan2(direction[1], direction[0]))), median_column_spacing_m=float(np.median(np.diff(x_centres))), median_row_spacing_m=float(np.median(np.diff(y_centres))), occupied_cells=int(np.isfinite(grid).sum()))
    (output / 'grid_geometry.json').write_text(json.dumps(geometry, indent=2), encoding='utf-8')
    print(json.dumps(geometry))

if(__name__ == '__main__'):
    main()
