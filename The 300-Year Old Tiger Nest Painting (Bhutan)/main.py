import json
import time
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def edge_costs(strips):
    pixels = strips.astype(np.float32)
    if(pixels.ndim == 3): pixels = pixels[..., None]
    count = len(pixels)
    left, right = pixels[:, :, 0].reshape(count, -1), pixels[:, :, -1].reshape(count, -1)
    if(pixels.shape[2] > 1):
        left_gradient = (pixels[:, :, 1] - pixels[:, :, 0]).reshape(count, -1)
        right_gradient = (pixels[:, :, -1] - pixels[:, :, -2]).reshape(count, -1)
        costs = cdist(right + 0.5 * right_gradient, left, 'sqeuclidean') + cdist(right, left - 0.5 * left_gradient, 'sqeuclidean')
    else:
        costs = cdist(right, left, 'sqeuclidean')
    np.fill_diagonal(costs, np.inf)
    return costs

def recover_order(costs):
    count = len(costs)
    if(count == 1): return [0]
    # A dummy strip turns a left-to-right path into a cycle with free endpoints.
    augmented = np.zeros((count + 1, count + 1), dtype=np.float64)
    augmented[:count, :count] = costs
    rows, columns = linear_sum_assignment(augmented)
    successor = np.empty(count + 1, dtype=np.intp)
    successor[rows] = columns
    remaining, cycles = set(range(count + 1)), []
    while(remaining):
        cycle, current = [], min(remaining)
        while(current in remaining):
            cycle.append(current)
            remaining.remove(current)
            current = int(successor[current])
        cycles.append(np.array(cycle, dtype=np.intp))
    # Merge any disconnected assignment cycles by the least costly edge exchange.
    while(len(cycles) > 1):
        best = None
        for(a, first) in enumerate(cycles):
            for(b, second) in enumerate(cycles[:a]):
                delta = augmented[np.ix_(first, successor[second])] + augmented[np.ix_(second, successor[first])].T - augmented[first, successor[first]][:, None] - augmented[second, successor[second]][None, :]
                i, j = np.unravel_index(np.argmin(delta), delta.shape)
                candidate = (float(delta[i, j]), a, b, int(first[i]), int(second[j]))
                if(best is None or candidate[0] < best[0]): best = candidate
        _, a, b, x, y = best
        successor[x], successor[y] = successor[y], successor[x]
        cycles[b] = np.concatenate((cycles[b], cycles[a]))
        cycles.pop(a)
    order, current = [], int(successor[count])
    while(current != count):
        order.append(current)
        current = int(successor[current])
    if(len(order) != count or len(set(order)) != count): raise RuntimeError('The reconstruction does not contain each strip exactly once.')
    return order

def main():
    started = time.perf_counter()
    root = Path(__file__).resolve().parent
    folder = root / 'input'
    if(not folder.is_dir()): folder = Path.cwd() / 'input'
    files = sorted(p for p in folder.iterdir() if(p.is_file() and p.suffix.lower() == '.png'))
    if(not files): raise FileNotFoundError('Place the PNG strips in the input folder.')
    images = [cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_UNCHANGED) for p in files]
    if(any(image is None for image in images)): raise ValueError('An input strip could not be decoded.')
    if(any(image.shape != images[0].shape for image in images)): raise ValueError('All strips must have the same dimensions and channels.')
    order = recover_order(edge_costs(np.stack(images)))
    reconstructed = np.concatenate([images[i] for i in order], axis=1)
    success, encoded = cv2.imencode('.png', reconstructed)
    if(not success): raise RuntimeError('The reconstructed image could not be encoded.')
    encoded.tofile(root / 'reconstructed_image.png')
    (root / 'reconstructed_order.json').write_text(json.dumps([files[i].name for i in order], indent=2) + '\n', encoding='utf-8')
    print(f'Reconstructed {len(files)} strips in {time.perf_counter() - started:.3f} seconds.')

if(__name__ == '__main__'): main()
