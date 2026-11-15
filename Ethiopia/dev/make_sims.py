"""Build a test bench of simulated splices from train-input.wav.

1. Remove the real splice (14.96-20.965 s) by covering it with real music from the same file.
2. For each anomaly type, insert a 4-8 s piece taken from elsewhere in the song and altered in
   one way (EQ, reverb, pitch, tempo, noise floor, ...), at a random position.
3. Save every case as sims/<name>.wav and the true boundaries in sims/truth.csv.

Run from the Ethiopia folder:  python dev/make_sims.py
"""
import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, resample, sosfilt

HERE = Path(__file__).parent
TRAIN = HERE.parent / "train-input.wav"
OUT = HERE / "sims"
TRUE_SPLICE = (14.96, 20.965)
CASES_PER_TYPE = 3
rng = np.random.default_rng(2026)


def lowpass(x, sr, hz, order=2):
    return sosfilt(butter(order, hz, "lp", fs=sr, output="sos"), x)


def highpass(x, sr, hz, order=2):
    return sosfilt(butter(order, hz, "hp", fs=sr, output="sos"), x)


def pitch_shift(x, sr, semitones, n):
    """Resample-based shift: changes pitch and tempo together, like a different take."""
    r = 2 ** (semitones / 12)
    return resample(x[: int(n * r) + 10], int((int(n * r) + 10) / r))[:n]


def time_stretch(x, sr, factor, n):
    """Different tempo, same pitch: overlap-add of 50 ms grains."""
    grain, hop_out = int(0.05 * sr), int(0.025 * sr)
    hop_in = int(hop_out * factor)
    win = np.hanning(grain)
    out = np.zeros(n + grain)
    norm = np.zeros(n + grain)
    i = 0
    while i * hop_out + grain < n + grain and i * hop_in + grain < len(x):
        out[i * hop_out:i * hop_out + grain] += x[i * hop_in:i * hop_in + grain] * win
        norm[i * hop_out:i * hop_out + grain] += win
        i += 1
    return (out / np.maximum(norm, 1e-3))[:n]


def reverb(x, sr):
    n = int(0.4 * sr)
    ir = np.exp(-np.arange(n) / (0.1 * sr)) * rng.standard_normal(n) * 0.2
    ir[0] = 1.0
    return np.convolve(x, ir)[: len(x)]


# Each anomaly gets (source chunk, sr, n) and returns the altered chunk of length n.
ANOMALIES = {
    "eq_less_treble": lambda c, sr, n: c[:n] - 0.5 * highpass(c[:n], sr, 2000),
    "eq_brighter":    lambda c, sr, n: c[:n] + 1.0 * highpass(c[:n], sr, 2500),
    "lowpass":        lambda c, sr, n: lowpass(c[:n], sr, 3000, 4),
    "less_bass":      lambda c, sr, n: c[:n] - 0.7 * lowpass(c[:n], sr, 300),
    "no_bass":        lambda c, sr, n: highpass(c[:n], sr, 400, 4),
    "reverb":         lambda c, sr, n: reverb(c[:n], sr),
    "noise_floor":    lambda c, sr, n: c[:n] + rng.standard_normal(n) * 0.02 * c[:n].std(),
    "pitch_up":       lambda c, sr, n: pitch_shift(c, sr, rng.choice([1, 2, 3, 5]), n),
    "pitch_down":     lambda c, sr, n: pitch_shift(c, sr, -rng.choice([1, 2, 3, 5]), n),
    "faster":         lambda c, sr, n: time_stretch(c, sr, 1.25, n),
    "slower":         lambda c, sr, n: time_stretch(c, sr, 0.8, n),
    "dc_shift":       lambda c, sr, n: c[:n] + 0.05 * c[:n].std(),
}


def main():
    sr, x = wavfile.read(TRAIN)
    x = x.astype(np.float64)
    a, b = (int(s * sr) for s in TRUE_SPLICE)
    base = x.copy()
    base[a:b] = x[25 * sr:25 * sr + (b - a)]
    duration = len(base) / sr

    OUT.mkdir(exist_ok=True)
    # Paths are stored relative to the Ethiopia folder (evaluate.py is run from there).
    rows = [("train", "real", TRAIN.name, *TRUE_SPLICE)]
    for kind, alter in ANOMALIES.items():
        for k in range(CASES_PER_TYPE):
            length = rng.uniform(4, 8)
            pos = rng.uniform(2, duration - length - 2)
            src = rng.uniform(2, duration - 2 * length - 2)
            n = int(length * sr)
            chunk = base[int(src * sr):int(src * sr) + 2 * n]          # extra room for pitch/tempo
            piece = alter(chunk, sr, n)
            target = base[int(pos * sr):int(pos * sr) + n]
            piece = piece * target.std() / (piece.std() + 1e-9)     # no jump in volume
            y = base.copy()
            y[int(pos * sr):int(pos * sr) + n] = piece
            name = f"{kind}-{k}"
            path = OUT / f"{name}.wav"
            wavfile.write(path, sr, np.clip(y, -32768, 32767).astype(np.int16))
            rows.append((name, kind, path.relative_to(HERE.parent).as_posix(), round(pos, 3), round(pos + length, 3)))

    with open(OUT / "truth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "kind", "path", "start_s", "end_s"])
        w.writerows(rows)
    print(f"wrote {len(rows) - 1} simulated cases + train to {OUT}")


if __name__ == "__main__":
    main()
