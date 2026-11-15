import csv
import sys

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import percentile_filter
from scipy.signal import stft

OUTPUT_FILE = "detection.csv"

FRAME = 8192            # long window -> fine frequency resolution down to 20 Hz
HOP = 512               # ~23 ms between frames
N_BANDS = 24            # log-spaced bands from LOWEST_HZ to Nyquist
LOWEST_HZ = 20
FLOOR_WINDOW_S = 0.5    # rolling window for the noise-floor estimate
FLOOR_PERCENTILE = 10   # "quiet moments" = the 10th percentile inside that window
SEARCH_STEP_S = 0.05    # time resolution of the boundary search
MIN_LEN_S = 1.0         # shortest splice considered
MAX_LEN_S = 20.0        # longest splice considered


def log(message):
    print(message, file=sys.stderr)


def load_audio(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def band_levels(x, sr):
    f, t, Z = stft(x, fs=sr, nperseg=FRAME, noverlap=FRAME - HOP, boundary=None, padded=False)
    db = 10 * np.log10(np.abs(Z) ** 2 + 1e-6)
    edges = np.geomspace(LOWEST_HZ, sr / 2, N_BANDS + 1)
    levels = np.array([db[(f >= lo) & (f < hi)].mean(axis=0) for lo, hi in zip(edges[:-1], edges[1:])])
    return t, levels


def noise_floor(levels, frame_s):
    width = max(1, int(FLOOR_WINDOW_S / frame_s))
    return percentile_filter(levels, FLOOR_PERCENTILE, size=(1, width), mode="nearest")


def deviation_from_take(floor):

    median = np.median(floor, axis=1, keepdims=True)
    spread = np.median(np.abs(floor - median), axis=1, keepdims=True) * 1.4826 + 0.5
    return np.clip((floor - median) / spread, -10, 10)


def best_segment(Z, step_s):
    n = Z.shape[1]
    cumsum = np.concatenate([np.zeros((Z.shape[0], 1)), np.cumsum(Z, axis=1)], axis=1)
    total = cumsum[:, -1]
    min_len = max(1, int(MIN_LEN_S / step_s))
    max_len = int(MAX_LEN_S / step_s)

    best_score, best_i, best_j = -1.0, 0, min_len
    for i in range(n - min_len + 1):
        j = np.arange(i + min_len, min(n, i + max_len) + 1)
        n_in = j - i
        sum_in = cumsum[:, j] - cumsum[:, [i]]
        mean_in = sum_in / n_in
        mean_out = (total[:, None] - sum_in) / (n - n_in)
        score = ((mean_in - mean_out) ** 2).sum(axis=0) * n_in * (n - n_in) / n
        k = int(np.argmax(score))
        if score[k] > best_score:
            best_score, best_i, best_j = float(score[k]), i, int(j[k])
    return best_i, best_j, best_score


def detect_splice(x, sr):
    t, levels = band_levels(x, sr)
    frame_s = HOP / sr
    Z = deviation_from_take(noise_floor(levels, frame_s))

    step = max(1, int(SEARCH_STEP_S / frame_s))        # search on a coarser grid for speed
    Zc, tc = Z[:, ::step], t[::step]
    i, j, score = best_segment(Zc, step * frame_s)
    start, end = float(tc[i]), float(tc[min(j, len(tc) - 1)])

    inside = np.abs(Zc[:, i:j]).mean()
    outside = np.abs(np.concatenate([Zc[:, :i], Zc[:, j:]], axis=1)).mean()
    log(f"score {score:.0f}   mean |deviation| inside {inside:.2f} vs outside {outside:.2f}")
    return start, end


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "test-input.wav"
    x, sr = load_audio(input_file)
    log(f"Loaded {input_file}: {sr} Hz, {len(x) / sr:.2f} s")

    start, end = detect_splice(x, sr)
    log(f"Splice: {start:.3f} s -> {end:.3f} s ({end - start:.2f} s long)")

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_s", "end_s"])
        writer.writerow([f"{start:.3f}", f"{end:.3f}"])


if __name__ == "__main__":
    main()
