"""
The Azmari's Borrowed Melody - splice localisation.

Usage:  python main.py test-1-input.wav
Writes: detection.csv  (header start_s,end_s and one row, seconds)

Idea
----
A splice inserts audio that was recorded somewhere else (other room, other
microphone, other take). Even if the seam is inaudible and the volume is
matched, the inserted part carries a different "fingerprint": a different
background noise floor (e.g. low-frequency rumble), a different spectral
balance, and melodic material that does not repeat anywhere else in the take.

We turn the recording into a sequence of feature vectors (one every 50 ms)
and search every possible interval [a, b). For each interval we measure how
much of each feature's variance is explained by the split "inside vs outside"
(R^2 of a two-group model). The interval with the highest mean R^2 over all
features is the one that is most different from the rest of the file:
that is the splice.
"""
import sys
import csv
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.ndimage import uniform_filter1d, percentile_filter


def load_mono(path):
    sr, d = wavfile.read(path)
    x = d.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return sr, x


def quiet_mask(P, f, hop_s, drop_db=15.0):
    """Frames that are much quieter than the music (pauses, digital silence,
    silent padding, quiet lead-in). The forger matched the volume, so the
    splice is never quiet; these frames must not create contrast."""
    lin = P[(f > 100) & (f < 5000)].mean(axis=0)
    k = max(1, int(round(0.25 / hop_s)))
    lin_s = uniform_filter1d(lin, k)
    db = 10 * np.log10(lin_s + 1e-20)
    q = db < np.median(db) - drop_db
    d = max(1, int(round(0.3 / hop_s)))           # also drop the fade edges
    q = np.convolve(q.astype(float), np.ones(2 * d + 1), "same") > 0
    if q.mean() > 0.5:                             # odd file: do not mask
        q[:] = False
    return q


def frame_features(x, sr, hop_s):
    hop = max(1, int(round(hop_s * sr)))
    n = 4096 if sr > 30000 else 2048
    n = max(n, 2 * hop)
    f, t, S = signal.stft(x, sr, nperseg=n, noverlap=n - hop,
                          boundary=None, padded=False)
    P = np.abs(S) ** 2
    P = P + max(1e-10, 1e-9 * P.mean())           # no -inf dB on digital silence
    quiet = quiet_mask(P, f, hop_s)

    # 1) energy in 40 log-spaced frequency bands (15 Hz ... 11 kHz)
    edges = np.geomspace(15, min(11000, sr / 2 * 0.98), 41)
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < hi)
        if m.sum() == 0:
            k = np.argmin(np.abs(f - (lo + hi) / 2))
            bands.append(10 * np.log10(P[k]))
        else:
            bands.append(10 * np.log10(P[m].mean(axis=0)))
    B = np.array(bands)

    # quiet frames get the typical (median) music values -> neutral
    def neutral(M):
        if quiet.any():
            M = M.copy()
            M[:, quiet] = np.median(M[:, ~quiet], axis=1, keepdims=True)
        return M
    B = neutral(B)

    # 2) background floor per band: quietest 10 % of frames within 1 s
    w = max(3, int(round(1.0 / hop_s)))
    Fl = percentile_filter(B, 10, size=(1, w))

    # 3) spectral shape: band energies relative to the frame's overall level
    lvl = 10 * np.log10(P[(f > 100) & (f < 5000)].mean(axis=0))
    Bn = neutral(B - lvl)

    # 4) repetition: does this spectral frame occur elsewhere in the take?
    m = (f > 100) & (f < 4000)
    C = np.log10(P[m])
    C = uniform_filter1d(C, 5, axis=1)
    C = C - C.mean(axis=0)
    C /= np.linalg.norm(C, axis=0) + 1e-9
    SS = C.T @ C
    idx = np.arange(SS.shape[0])
    ex = int(round(3.0 / hop_s))
    SS[np.abs(idx[:, None] - idx[None, :]) < ex] = -1  # ignore own neighbourhood
    SS[:, quiet] = -1                                   # silence is not a match
    rep = np.sort(SS, axis=1)[:, -5:].mean(axis=1)
    rep = neutral(rep[None, :])

    X = np.vstack([B, Fl, Bn, rep])
    return t, X


def best_segment(X, minlen, maxlen, margin):
    """Exhaustive search of [a, b) maximising mean R^2 over features."""
    k, N = X.shape
    X = X - X.mean(axis=1, keepdims=True)
    tss = (X ** 2).sum(axis=1) + 1e-12
    cs = np.concatenate([np.zeros((k, 1)), np.cumsum(X, axis=1)], axis=1)
    best, best_ab = -np.inf, (margin, margin + minlen)
    for a in range(margin, N - margin - minlen + 1):
        bs = np.arange(a + minlen, min(a + maxlen, N - margin) + 1)
        if len(bs) == 0:
            continue
        s_in = cs[:, bs] - cs[:, [a]]
        n_in = (bs - a)[None, :]
        n_out = N - n_in
        between = s_in ** 2 * (1.0 / n_in + 1.0 / n_out)
        score = (between / tss[:, None]).mean(axis=0)
        j = int(np.argmax(score))
        if score[j] > best:
            best, best_ab = score[j], (a, int(bs[j]))
    return best_ab, best


def detect(x, sr):
    dur = len(x) / sr
    hop_s = max(0.05, dur / 2000.0)  # keep the search fast for long files
    t, X = frame_features(x, sr, hop_s)
    X = uniform_filter1d(X, max(1, int(round(0.5 / hop_s))), axis=1)
    N = X.shape[1]
    minlen = max(2, int(1.5 / hop_s))
    maxlen = min(int(25.0 / hop_s), int(0.6 * N))
    margin = int(1.4 / hop_s)   # tolerance is 1.5 s, so this costs nothing
    (a, b), score = best_segment(X, minlen, maxlen, margin)
    t0 = t[0] - hop_s / 2  # frame i covers [t0 + i*hop, t0 + (i+1)*hop)
    start = max(0.0, t0 + a * hop_s)
    end = min(dur, t0 + b * hop_s)
    return start, end, score


def main():
    input_file = sys.argv[1]
    sr, x = load_mono(input_file)
    start, end, _ = detect(x, sr)
    with open("detection.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["start_s", "end_s"])
        w.writerow([f"{start:.3f}", f"{end:.3f}"])


if __name__ == "__main__":
    main()
