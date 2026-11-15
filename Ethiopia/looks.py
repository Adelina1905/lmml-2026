"""Visualize the training sound to see what makes the labeled section special.

Draws three panels on one shared time axis:
  1. waveform      - raw sample values
  2. RMS loudness  - how loud each short window is
  3. spectrogram   - which frequencies are present over time (in dB)
The labeled section from train-output.csv is shaded on every panel.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram

HERE = Path(__file__).parent
WAV_PATH = HERE / "train-input.wav"
CSV_PATH = HERE / "train-output.csv"
OUT_PATH = HERE / "looks.png"

FRAME = 2048  # samples per window (~93 ms at 22050 Hz)
HOP = 512     # step between windows (~23 ms)


def load_audio(path):
    """Read the WAV and return (samples as float in [-1, 1], sample rate)."""
    sr, x = wavfile.read(path)
    if x.ndim > 1:                      # stereo -> mono, just in case
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    return x.astype(np.float32), sr


def load_segments(path):
    """Read the (start_s, end_s) rows from the label CSV."""
    with open(path, newline="") as f:
        return [(float(r["start_s"]), float(r["end_s"])) for r in csv.DictReader(f)]


def rms_curve(x, sr, frame=FRAME, hop=HOP):
    """Loudness per window: sqrt(mean(x^2)). Returns (times, rms)."""
    n = 1 + (len(x) - frame) // hop
    frames = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop][:n]
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    times = (np.arange(n) * hop + frame / 2) / sr   # centre of each window
    return times, rms


def main():
    x, sr = load_audio(WAV_PATH)
    segments = load_segments(CSV_PATH)
    t = np.arange(len(x)) / sr

    print(f"sample rate: {sr} Hz, duration: {len(x) / sr:.2f} s")
    print(f"labeled segments: {segments}")

    rms_t, rms = rms_curve(x, sr)
    f, spec_t, Sxx = spectrogram(x, fs=sr, nperseg=FRAME, noverlap=FRAME - HOP)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(t, x, linewidth=0.3)
    axes[0].set_ylabel("amplitude")
    axes[0].set_title("Waveform")

    axes[1].plot(rms_t, rms)
    axes[1].set_ylabel("RMS")
    axes[1].set_title("Loudness (RMS energy)")

    mesh = axes[2].pcolormesh(spec_t, f, Sxx_db, shading="auto", cmap="magma")
    axes[2].set_ylabel("frequency (Hz)")
    axes[2].set_xlabel("time (s)")
    axes[2].set_title("Spectrogram (dB)")
    fig.colorbar(mesh, ax=axes[2], orientation="horizontal", pad=0.25,
                 fraction=0.05, label="dB")

    for ax in axes:
        for start, end in segments:
            ax.axvspan(start, end, color="cyan", alpha=0.25)

    # Quick numbers: how different is the labeled part from the rest?
    inside = np.zeros_like(rms, dtype=bool)
    for start, end in segments:
        inside |= (rms_t >= start) & (rms_t <= end)
    print(f"mean RMS inside label:  {rms[inside].mean():.4f}")
    print(f"mean RMS outside label: {rms[~inside].mean():.4f}")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=120)
    print(f"saved plot to {OUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
