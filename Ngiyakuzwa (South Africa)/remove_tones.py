"""Remove narrow persistent stadium tones without supplying any expected speech text."""
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, iirnotch, peak_widths, sosfiltfilt, tf2sos, welch
ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "samples" / "candidate_0431_original.wav")
    parser.add_argument("--output", type=Path, default=ROOT / "samples" / "candidate_0431_notched.wav")
    parser.add_argument("--width", type=float, default=4)
    args = parser.parse_args()
    if(args.output.exists() or args.input.resolve() == args.output.resolve()): parser.error("Choose a new output file.")
    audio, rate = sf.read(args.input, dtype="float64")
    frequencies, power = welch(audio, rate, nperseg=min(65536, len(audio)))
    peaks, properties = find_peaks(10 * np.log10(power + 1e-20), prominence=20, distance=20)
    widths = peak_widths(power, peaks, rel_height=0.5)[0] * (frequencies[1] - frequencies[0])
    selected = [(index, prominence, width) for index, prominence, width in zip(peaks, properties["prominences"], widths) if 100 < frequencies[index] < min(7800, rate / 2 - 100) and width < 2.5]
    filters = [tf2sos(*iirnotch(frequencies[index], frequencies[index] / args.width, fs=rate)) for index, _, _ in selected]
    output = sosfiltfilt(np.vstack(filters), audio) if filters else audio
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, output, rate, subtype="PCM_16")
    metadata = {"source": args.input.name, "method": "Zero-phase IIR notch filters on narrow PSD peaks", "minimum_prominence_db": 20, "maximum_peak_width_hz": 2.5, "notch_width_hz": args.width, "tones": [{"frequency_hz": round(float(frequencies[index]), 4), "prominence_db": round(float(prominence), 3), "peak_width_hz": round(float(width), 4)} for index, prominence, width in selected]}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Removed {len(selected)} tones; saved {args.output}")

if(__name__ == "__main__"): main()
