"""Clip WAV audio, enhance a phrase with FRCRN, or export a listening copy."""
import argparse
import json
import os
from pathlib import Path
import wave
ROOT = Path(__file__).resolve().parent

def clip_audio(source, output, start, duration):
    with wave.open(str(source), "rb") as audio:
        rate, params = audio.getframerate(), audio.getparams()
        start_frame = round(start * rate)
        if(start_frame >= audio.getnframes()): raise ValueError("Clip starts after the recording ends.")
        frames = min(round(duration * rate), audio.getnframes() - start_frame)
        audio.setpos(start_frame)
        data = audio.readframes(frames)
    with wave.open(str(output), "wb") as audio:
        audio.setparams(params)
        audio.writeframes(data)
    print(f"Saved {output.name}: source seconds {start_frame / rate:.3f} to {(start_frame + frames) / rate:.3f}")

def enhance_audio(source, output):
    os.chdir(ROOT)
    os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
    import torch
    from clearvoice import ClearVoice
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    enhancer = ClearVoice(task="speech_enhancement", model_names=["FRCRN_SE_16K"])
    with torch.inference_mode(): enhanced = enhancer(input_path=str(source), online_write=False)
    enhancer.write(enhanced, output_path=str(output))
    output.with_suffix(".json").write_text(json.dumps({"source": source.name, "model": "alibabasglab/FRCRN_SE_16K", "torch_version": torch.__version__}, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}")

def listening_copy(source, output, start, duration, speed):
    import numpy as np
    import soundfile as sf
    audio, rate = sf.read(source, dtype="float32")
    if(audio.ndim != 1): raise ValueError("Expected mono audio.")
    audio = audio[round(start * rate):round((start + duration) * rate)]
    if(not len(audio)): raise ValueError("The selected interval is empty.")
    if(speed != 1):
        import librosa
        audio = librosa.effects.time_stretch(audio, rate=speed)
    audio = audio * (10 ** (-1 / 20) / max(float(np.max(np.abs(audio))), 1e-9))
    sf.write(output, audio, rate, subtype="PCM_16")
    print(f"Saved {output.name}: peak -1 dBFS, playback speed {speed:g}, pitch preserved")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["clip", "enhance", "listen"])
    parser.add_argument("--input", type=Path, default=ROOT / "voice_note.wav")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--speed", type=float, default=1)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if(source == output or output.exists()): parser.error("Choose a new output file.")
    if(args.start < 0 or args.duration <= 0 or args.speed <= 0): parser.error("Invalid time or playback speed.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if(args.action == "clip"): clip_audio(source, output, args.start, args.duration)
    elif(args.action == "enhance"): enhance_audio(source, output)
    elif(args.action == "listen"): listening_copy(source, output, args.start, args.duration, args.speed)

if(__name__ == "__main__"): main()
