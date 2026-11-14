"""Enhance a long recording with contextual blocks and resumable intermediate WAVs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
import numpy as np
import soundfile as sf
import torch
from clearvoice import ClearVoice

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "voice_note.wav")
    parser.add_argument("--output", type=Path, default=ROOT / "frcrn_full.wav")
    parser.add_argument("--block", type=float, default=30)
    parser.add_argument("--context", type=float, default=1)
    args = parser.parse_args()
    source, destination = args.input.resolve(), args.output.resolve()
    if(destination.exists() or source == destination): parser.error("Choose a new output file.")
    if(args.block <= 0 or args.context < 0): parser.error("Invalid block or context duration.")
    audio, rate = sf.read(source, dtype="float32")
    if(rate != 16000 or audio.ndim != 1): raise ValueError("Expected mono 16 kHz audio.")
    os.chdir(ROOT)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    with source.open("rb") as stream: digest = hashlib.file_digest(stream, "sha256").hexdigest()
    work = ROOT / "cache" / f"frcrn_{digest[:12]}_{args.block:g}_{args.context:g}"
    work.mkdir(parents=True, exist_ok=True)
    enhancer = ClearVoice(task="speech_enhancement", model_names=["FRCRN_SE_16K"])
    block, context = round(args.block * rate), round(args.context * rate)
    output = np.zeros_like(audio)
    timings = []
    for index, start in enumerate(range(0, len(audio), block)):
        end = min(start + block, len(audio))
        left, right = max(0, start - context), min(len(audio), end + context)
        raw, cleaned = work / f"{index:03d}_raw.wav", work / f"{index:03d}_clean.wav"
        began = time.monotonic()
        if(not cleaned.exists()):
            sf.write(raw, audio[left:right], rate, subtype="PCM_16")
            with torch.inference_mode(): result = enhancer(input_path=str(raw), online_write=False)
            enhancer.write(result, output_path=str(cleaned))
        segment, segment_rate = sf.read(cleaned, dtype="float32")
        if(segment_rate != rate or segment.ndim != 1 or len(segment) < right - left or not np.isfinite(segment).all()): raise ValueError(f"Invalid enhanced block: {index}")
        output[start:end] = segment[start - left:end - left]
        timings.append({"start_seconds": start / rate, "end_seconds": end / rate, "context_start_seconds": left / rate, "processing_seconds": round(time.monotonic() - began, 3)})
        print(f"Enhanced {end / rate:.0f}/{len(audio) / rate:.0f} seconds", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, output, rate, subtype="PCM_16")
    metadata = {"source": source.name, "source_sha256": digest, "model": "alibabasglab/FRCRN_SE_16K", "sample_rate": rate, "samples": len(output), "duration_seconds": len(output) / rate, "block_seconds": args.block, "context_seconds": args.context, "device": "cuda" if torch.cuda.is_available() else "cpu", "torch": torch.__version__, "blocks": timings}
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {destination}", flush=True)

if(__name__ == "__main__"): main()
