"""(5) Visualize the tones and their suppression.

Plot the original spectrum and the effect of the saved notch filters.
"""

import argparse
import json
from pathlib import Path
import wave
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import iirnotch, sosfiltfilt, tf2sos, welch

ROOT = Path(__file__).resolve().parent


def main():
    """Compare the original spectrum with the saved notch-filter result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "samples" / "candidate_0431_original.wav"
    )
    parser.add_argument(
        "--settings", type=Path, default=ROOT / "results" / "tone_filter_settings.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "power_spectrum.png")
    args = parser.parse_args()
    with wave.open(str(args.input), "rb") as audio:
        if(audio.getnchannels() != 1 or audio.getsampwidth() != 2):
            raise ValueError("Expected mono 16-bit PCM audio.")
        rate = audio.getframerate()
        samples = (
            np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float64)
            / 32768
        )
    # Reuse the frequencies recorded during the actual cleanup.
    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    tones = np.array([item["frequency_hz"] for item in settings["tones"]])
    sections = [
        tf2sos(*iirnotch(tone, tone / settings["notch_width_hz"], fs=rate)) for tone in tones
    ]
    filtered = sosfiltfilt(np.vstack(sections), samples)
    frequencies, original_power = welch(samples, rate, nperseg=min(65536, len(samples)))
    _, filtered_power = welch(filtered, rate, nperseg=min(65536, len(samples)))
    original_db = 10 * np.log10(np.maximum(original_power, 1e-18))
    filtered_db = 10 * np.log10(np.maximum(filtered_power, 1e-18))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    # Show the overall spectrum above and one filtered tone below.
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
    figure.suptitle("Finding and reducing steady horn tones", fontsize=18, weight="bold")
    axes[0].plot(
        frequencies, original_db, color="#285f9e", linewidth=0.85, label="Original audio"
    )
    visible_tones = tones[tones <= 2000]
    indices = np.array([int(np.argmin(np.abs(frequencies - tone))) for tone in visible_tones])
    axes[0].scatter(
        frequencies[indices],
        original_db[indices],
        color="#c64535",
        s=28,
        zorder=3,
        label="Detected narrow peaks",
    )
    for frequency, offset in [
        (217.5293, (10, 15)),
        (466.0645, (14, 4)),
        (699.2188, (14, -8)),
        (932.373, (14, 14)),
    ]:
        index = int(np.argmin(np.abs(frequencies - frequency)))
        axes[0].annotate(
            f"{frequency:.1f} Hz",
            (frequencies[index], original_db[index]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "-", "color": "#555555"},
        )
    axes[0].set(
        title="1. Power spectrum: tall, narrow peaks show strong tones",
        xlim=(0, 2000),
        ylim=(-115, -17),
        xlabel="Frequency (Hz): lower pitch to higher pitch",
        ylabel="Power (dBFS/Hz)",
    )
    axes[0].legend(loc="upper right", frameon=False)
    center = float(tones[np.argmin(np.abs(tones - 466))])
    axes[1].plot(
        frequencies, original_db, color="#285f9e", linewidth=1.8, label="Before filtering"
    )
    axes[1].plot(
        frequencies, filtered_db, color="#d47822", linewidth=1.8, label="After notch filters"
    )
    axes[1].axvline(center, color="#666666", linestyle=":", linewidth=1)
    axes[1].set(
        title=f"2. Zoom near {center:.2f} Hz: the filter turns down this narrow tone",
        xlim=(center - 12, center + 12),
        ylim=(-115, -20),
        xlabel="Frequency (Hz)",
        ylabel="Power (dBFS/Hz)",
    )
    axes[1].legend(loc="upper right", frameon=False)
    for axis in axes:
        axis.grid(alpha=0.18)
    figure.supxlabel(
        "Measured from the original 431-447 s excerpt. Orange shows mathematical notch filtering only, before FRCRN.",
        fontsize=10,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    index = int(np.argmin(np.abs(frequencies - center)))
    print(
        json.dumps(
            {
                "image": str(args.output),
                "filters": len(tones),
                "example_frequency_hz": center,
                "original_peak_dbfs_per_hz": round(float(original_db[index]), 2),
                "filtered_peak_dbfs_per_hz": round(float(filtered_db[index]), 2),
            },
            indent=2,
        )
    )


if(__name__ == "__main__"):
    main()
