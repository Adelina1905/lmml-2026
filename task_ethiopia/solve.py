"""
task_ethiopia - The Azmari's Borrowed Melody
Reproduces the analysis behind main.py:

  1. runs the detector (imported from main.py) on train-input.wav and compares
     with train-output.csv
  2. saves proof plots (spectrogram, low-frequency band energy, R^2 map)
  3. builds synthetic validation recordings: takes the genuine part of the
     training take, splices foreign material of 13 different kinds into random
     places (crossfaded, RMS-matched so there is no click / volume jump), and
     checks how often the detector lands within the +-1.5 s tolerance
  4. writes results/report.txt

Run from inside this folder:   python solve.py          (about 2-4 minutes)
"""
import os
import sys
import collections
import numpy as np
from scipy.io import wavfile
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import main  # noqa: E402

RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
TOL = 1.5
report = []


def log(s=""):
    print(s, flush=True)
    report.append(s)


# ---------------------------------------------------------------- 1. train
sr, x = main.load_mono(os.path.join(HERE, "train-input.wav"))
with open(os.path.join(HERE, "train-output.csv")) as fh:
    fh.readline()
    gt_s, gt_e = [float(v) for v in fh.readline().strip().split(",")]
s, e, score = main.detect(x, sr)
log("=== Training recording ===")
log(f"sample rate {sr} Hz, duration {len(x)/sr:.2f} s")
log(f"ground truth : start {gt_s:.3f}  end {gt_e:.3f}")
log(f"detected     : start {s:.3f}  end {e:.3f}   (mean R^2 = {score:.3f})")
log(f"errors       : start {abs(s-gt_s):.3f} s, end {abs(e-gt_e):.3f} s  "
    f"-> {'PASS' if abs(s-gt_s) <= TOL and abs(e-gt_e) <= TOL else 'FAIL'} (tolerance {TOL} s)")

# ---------------------------------------------------------------- 2. plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f, t, S = signal.stft(x, sr, nperseg=4096, noverlap=3584)
    L = 10 * np.log10(np.abs(S) ** 2 + 1e-6)
    hop = int(0.05 * sr)
    f2, t2, S2 = signal.stft(x, sr, nperseg=2048, noverlap=2048 - hop)
    P2 = np.abs(S2) ** 2 + 1e-10
    lf = 10 * np.log10(P2[(f2 >= 20) & (f2 < 80)].mean(0))

    fig, ax = plt.subplots(3, 1, figsize=(15, 10))
    m = f < 1500
    ax[0].pcolormesh(t, f[m], L[m], vmin=-40, vmax=40, shading="auto")
    ax[0].set_title("Spectrogram 0-1500 Hz (red = true splice, white dashed = detected)")
    m = f < 200
    ax[1].pcolormesh(t, f[m], L[m], vmin=-40, vmax=30, shading="auto")
    ax[1].set_title("Spectrogram 0-200 Hz: the inserted part has much more low-frequency rumble")
    ax[2].plot(t2, lf, lw=0.5, label="20-80 Hz band energy (dB)")
    ax[2].plot(t2, np.convolve(lf, np.ones(21) / 21, "same"), lw=1.5, label="smoothed")
    ax[2].legend(loc="upper right")
    ax[2].set_xlabel("time (s)")
    for a in ax:
        a.axvline(gt_s, color="r")
        a.axvline(gt_e, color="r")
        a.axvline(s, color="w" if a is not ax[2] else "k", ls="--")
        a.axvline(e, color="w" if a is not ax[2] else "k", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "train_spectrogram.png"), dpi=80)
    plt.close()

    # R^2 map over all (start, end) pairs
    hop_s = 0.05
    tt, X = main.frame_features(x, sr, hop_s)
    from scipy.ndimage import uniform_filter1d
    X = uniform_filter1d(X, 10, axis=1)
    N = X.shape[1]
    Xc = X - X.mean(1, keepdims=True)
    tss = (Xc ** 2).sum(1) + 1e-12
    cs = np.concatenate([np.zeros((X.shape[0], 1)), np.cumsum(Xc, 1)], 1)
    step = 4
    grid = np.arange(0, N + 1, step)
    R = np.full((len(grid), len(grid)), np.nan)
    for i, a in enumerate(grid):
        for j, b in enumerate(grid):
            if b - a >= 30 and b - a <= 500 and a > 0 and b < N:
                sin = cs[:, b] - cs[:, a]
                R[i, j] = (sin ** 2 * (1 / (b - a) + 1 / (N - (b - a))) / tss).mean()
    plt.figure(figsize=(7, 6))
    ext = [grid[0] * hop_s, grid[-1] * hop_s, grid[0] * hop_s, grid[-1] * hop_s]
    plt.imshow(R, origin="lower", extent=ext, aspect="auto")
    plt.colorbar(label="mean R^2 over 121 features")
    plt.scatter([gt_e], [gt_s], c="r", marker="x", s=80, label="true (end, start)")
    plt.scatter([e], [s], facecolors="none", edgecolors="w", s=120, label="detected")
    plt.xlabel("segment end (s)")
    plt.ylabel("segment start (s)")
    plt.legend()
    plt.title("Score of every candidate segment")
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "train_r2_map.png"), dpi=80)
    plt.close()
    log("plots saved: results/train_spectrogram.png, results/train_r2_map.png")
except ImportError:
    log("matplotlib not installed - skipping plots")

# ---------------------------------------------------------------- 3. synthetic validation
rng = np.random.default_rng(2026)
s0, e0 = int(gt_s * sr), int(gt_e * sr)
FOREIGN = x[s0:e0]                                 # the real inserted material
# take with the splice removed; 1 s of margin is dropped on each side because
# the original crossfade leaks foreign audio there, and the two halves are
# joined with a 50 ms crossfade so the join itself is clean
_cf = int(0.05 * sr)
_a, _b = x[:s0 - sr], x[e0 + sr:]
_w = np.linspace(0, 1, _cf)
GENUINE = np.concatenate([_a[:-_cf], _a[-_cf:] * (1 - _w) + _b[:_cf] * _w, _b[_cf:]])


def rms(v):
    return np.sqrt(np.mean(v ** 2)) + 1e-12


def lf_noise(n, level):
    b, a = signal.butter(2, [20, 90], btype="band", fs=sr)
    z = signal.lfilter(b, a, rng.standard_normal(n))
    return z / rms(z) * level


def pink(n):
    w = np.fft.rfft(rng.standard_normal(n))
    fr = np.fft.rfftfreq(n, 1 / sr)
    fr[0] = fr[1]
    z = np.fft.irfft(w / np.sqrt(fr), n)
    return z / rms(z)


def flute(n):
    out = np.zeros(n)
    notes = [293.7, 329.6, 392.0, 440.0, 523.3, 587.3]
    pos = 0
    while pos < n:
        L = min(int(rng.uniform(0.3, 1.0) * sr), n - pos)
        tt = np.arange(L) / sr
        f0 = rng.choice(notes)
        env = np.minimum(1, tt / 0.05) * np.minimum(1, (tt[-1] - tt + 1e-3) / 0.08)
        ph = 2 * np.pi * np.cumsum(f0 * (1 + 0.006 * np.sin(2 * np.pi * 5.5 * tt))) / sr
        tone = np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph)
        b, a = signal.butter(2, [f0 * 0.8, f0 * 3], btype="band", fs=sr)
        br = signal.lfilter(b, a, rng.standard_normal(L))
        out[pos:pos + L] = (tone + br / rms(br) * 0.25) * env
        pos += L
    return out


def pitch(seg_src_start, L, G):
    r = 2 ** (rng.choice([-2, -1, 1, 2, 3]) / 12)
    src = G[seg_src_start:seg_src_start + int(L * r) + 1]
    return signal.resample(src, L)[:L]


def make_take(G, kind, L, pos):
    srcmax = len(G) - int(L * 1.3) - 1
    cands = [c for c in range(0, srcmax, sr // 4) if abs(c - pos) >= L]
    if not cands:                      # very long splice in a short take
        cands = list(range(0, srcmax, sr // 4))
    src = int(rng.choice(cands))
    seg = G[src:src + L].copy()
    if kind == "real_foreign":
        k = int(rng.integers(0, max(1, len(FOREIGN) - L)))
        seg = FOREIGN[k:k + L] if L <= len(FOREIGN) else np.resize(FOREIGN, L)
    elif kind == "pitch+rumble":
        seg = pitch(src, L, G); seg = seg + lf_noise(L, rms(seg) * 0.25)
    elif kind == "rumble":
        seg = seg + lf_noise(L, rms(seg) * 0.25)
    elif kind == "rumble_weak":
        seg = seg + lf_noise(L, rms(seg) * 0.08)
    elif kind == "rumble_very_weak":
        seg = seg + lf_noise(L, rms(seg) * 0.03)
    elif kind == "flute":
        seg = flute(L)
    elif kind == "flute+rumble":
        seg = flute(L); seg = seg / rms(seg) * rms(G[src:src + L])
        seg = seg + lf_noise(L, rms(seg) * 0.25)
    elif kind == "pink_noise":
        seg = seg + pink(L) * rms(seg) * 0.05
    elif kind == "pink_noise_weak":
        seg = seg + pink(L) * rms(seg) * 0.02
    elif kind == "lowpass_mic":
        b, a = signal.butter(6, 3500, fs=sr); seg = signal.filtfilt(b, a, seg)
    elif kind == "pitch_only":
        seg = pitch(src, L, G)
    elif kind == "white_hiss_-40dB":
        seg = seg + rng.standard_normal(L) * rms(seg) * 0.01
    elif kind == "reverb":
        n = int(0.4 * sr)
        ir = rng.standard_normal(n) * np.exp(-np.arange(n) / (0.08 * sr)); ir[0] = 8
        seg = signal.fftconvolve(seg, ir)[:L]
    seg = seg / rms(seg) * rms(G[pos:pos + L])          # no volume jump
    out = G.copy()
    cf = int(0.05 * sr)                                   # 50 ms crossfade, no click
    w = np.ones(L); w[:cf] = np.linspace(0, 1, cf); w[-cf:] = np.linspace(1, 0, cf)
    out[pos:pos + L] = out[pos:pos + L] * (1 - w) + seg * w
    return np.round(out)


def new_take(G):
    """A different 'performance': pitch-shift the take and shuffle 7 s chunks."""
    r = 2 ** (rng.choice([-3, -2, 2, 4]) / 12)
    G2 = signal.resample_poly(G, 100, int(round(100 * r)))
    ch = 7 * sr
    parts = [G2[i:i + ch] for i in range(0, len(G2) - ch + 1, ch)]
    rng.shuffle(parts)
    cf = int(0.05 * sr)
    out = parts[0]
    for p in parts[1:]:
        w = np.linspace(0, 1, cf)
        out = np.concatenate([out[:-cf], out[-cf:] * (1 - w) + p[:cf] * w, p[cf:]])
    return out


KINDS = ["real_foreign", "pitch+rumble", "rumble", "rumble_weak", "rumble_very_weak",
         "flute", "flute+rumble", "pink_noise", "pink_noise_weak", "lowpass_mic",
         "pitch_only", "white_hiss_-40dB", "reverb"]
PER_KIND = 6
bases = [("original take", GENUINE), ("re-shuffled take A", new_take(GENUINE)),
         ("re-shuffled take B", new_take(GENUINE))]

log()
log("=== Synthetic validation ===")
log("Foreign audio spliced into genuine audio at random positions, length 2-12 s,")
log("50 ms crossfade and RMS matching (no click, no volume jump).")
ok = collections.Counter(); tot = collections.Counter(); errs = []
for bname, G in bases:
    for kind in KINDS:
        for r in range(PER_KIND):
            L = int(rng.uniform(2, 12) * sr)
            pos = int(rng.uniform(2 * sr, len(G) - L - 2 * sr))
            y = make_take(G, kind, L, pos)
            ts, te = pos / sr, (pos + L) / sr
            ds, de, _ = main.detect(y, sr)
            good = abs(ds - ts) <= TOL and abs(de - te) <= TOL
            ok[kind] += good; tot[kind] += 1
            if good:
                errs.append(max(abs(ds - ts), abs(de - te)))
            else:
                log(f"  miss [{bname}] {kind}: true {ts:.2f}-{te:.2f}, got {ds:.2f}-{de:.2f}")
log()
log(f"{'splice type':20s} pass rate")
for k in KINDS:
    log(f"{k:20s} {ok[k]:2d}/{tot[k]}")
n_ok, n_tot = sum(ok.values()), sum(tot.values())
log(f"{'TOTAL':20s} {n_ok}/{n_tot} ({100*n_ok/n_tot:.1f} %)")
log(f"median boundary error on passes: {np.median(errs):.3f} s, 95th percentile {np.percentile(errs, 95):.3f} s")

with open(os.path.join(RES, "report.txt"), "w") as fh:
    fh.write("\n".join(report) + "\n")
print("\nreport saved to results/report.txt")
