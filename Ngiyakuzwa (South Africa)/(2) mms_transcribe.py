"""(2) Locate the keyword with Zulu speech recognition.

MMS saves overlapping transcripts, word times, and approximate keyword matches.
"""

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC


def save(path, value):
    """Save progress atomically so an interrupted scan can resume."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main():
    """Recognize each window, retain source times, and save candidate matches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "frcrn_full.wav")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "mms_transcript.json")
    parser.add_argument("--window", type=float, default=25)
    parser.add_argument("--overlap", type=float, default=5)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--model", default="facebook/mms-1b-all")
    parser.add_argument("--revision", default="3d33597edbdaaba14a8e858e2c8caa76e3cec0cd")
    args = parser.parse_args()
    if(
        not 0 <= args.overlap < args.window
        or args.start < 0
        or (args.duration is not None and args.duration <= 0)
    ):
        parser.error("Invalid time parameters.")
    audio, rate = sf.read(args.input, dtype="float32")
    if(rate != 16000 or audio.ndim != 1):
        raise ValueError("Expected mono 16 kHz audio.")
    with args.input.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    dtype = torch.float16 if device == "cuda" else torch.float32
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    print(f"Loading {args.model}, language zul, device {device}", flush=True)
    # MMS needs the Zulu adapter; other supported checkpoints use their own vocabulary.
    language_options = {"target_lang": "zul"} if args.model.startswith("facebook/mms-") else {}
    processor = AutoProcessor.from_pretrained(
        args.model, revision=args.revision, **language_options
    )
    model = (
        Wav2Vec2ForCTC.from_pretrained(
            args.model,
            revision=args.revision,
            ignore_mismatched_sizes=bool(language_options),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            **language_options,
        )
        .to(device)
        .eval()
    )
    vocabulary = processor.tokenizer.get_vocab()
    if(processor.tokenizer.word_delimiter_token not in vocabulary and " " in vocabulary):
        processor.tokenizer.word_delimiter_token = " "
    processor.tokenizer.pad_token = processor.tokenizer.convert_ids_to_tokens(
        model.config.pad_token_id
    )
    frame_seconds = model.config.inputs_to_logits_ratio / rate
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "source": args.input.name,
        "source_sha256": digest,
        "model": args.model,
        "revision": args.revision,
        "language": "zul",
        "device": device,
        "dtype": str(dtype),
        "window_seconds": args.window,
        "overlap_seconds": args.overlap,
        "start_seconds": args.start,
        "duration_seconds": args.duration,
        "word_timestamps": "Approximate CTC token alignment mapped to original sample positions.",
        "segments": [],
    }
    # Resume only when the input and recognition settings match.
    if(output.exists()):
        previous = json.loads(output.read_text(encoding="utf-8"))
        for field in [
            "source_sha256",
            "model",
            "revision",
            "language",
            "window_seconds",
            "overlap_seconds",
            "start_seconds",
            "duration_seconds",
        ]:
            if(previous.get(field) != result[field]):
                raise ValueError(f"Resume configuration mismatch: {field}")
        result["segments"] = previous["segments"]
    completed = {segment["start_sample"] for segment in result["segments"]}
    stop = (
        min(len(audio), round((args.start + args.duration) * rate))
        if args.duration is not None
        else len(audio)
    )
    stride, window = round((args.window - args.overlap) * rate), round(args.window * rate)
    for start in range(round(args.start * rate), stop, stride):
        if(start in completed):
            continue
        end = min(start + window, stop)
        if(end - start < rate / 2):
            continue
        began = time.monotonic()
        inputs = processor(audio[start:end], sampling_rate=rate, return_tensors="pt")
        inputs = {
            name: (
                value.to(device=device, dtype=dtype)
                if value.is_floating_point()
                else value.to(device)
            )
            for name, value in inputs.items()
        }
        with torch.inference_mode():
            logits = model(**inputs).logits[0]
        if(not torch.isfinite(logits).all()):
            raise RuntimeError(
                "The recognizer produced non-finite logits; use float32 inference."
            )
        decoded = processor.tokenizer.decode(
            logits.argmax(dim=-1).cpu().tolist(), output_word_offsets=True
        )
        # Convert CTC frame offsets into times in the original input file.
        words = [
            {
                "word": item["word"],
                "start_seconds": round(start / rate + item["start_offset"] * frame_seconds, 3),
                "end_seconds": round(start / rate + item["end_offset"] * frame_seconds, 3),
            }
            for item in decoded.word_offsets
        ]
        # Include approximate spellings because noise can change recognized letters.
        matches = []
        for index, word in enumerate(words):
            normalized = re.sub("[^a-z]", "", word["word"].lower())
            similarity = difflib.SequenceMatcher(None, normalized, "ikhodi").ratio()
            if("ikhodi" in normalized or (4 <= len(normalized) <= 8 and similarity >= 0.66)):
                matches.append(
                    {
                        "word_index": index,
                        "word": word["word"],
                        "similarity": round(similarity, 3),
                        "start_seconds": word["start_seconds"],
                        "following_words": [
                            item["word"] for item in words[index + 1 : index + 4]
                        ],
                    }
                )
        segment = {
            "start_sample": start,
            "start_seconds": start / rate,
            "end_seconds": end / rate,
            "text": decoded.text,
            "words": words,
            "keyword_candidates": matches,
            "processing_seconds": round(time.monotonic() - began, 3),
        }
        result["segments"].append(segment)
        result["segments"].sort(key=lambda item: item["start_sample"])
        save(output, result)
        print(f"[{start / rate:.1f}–{end / rate:.1f}] {decoded.text}", flush=True)
        for match in matches:
            print("CANDIDATE " + json.dumps(match, ensure_ascii=False), flush=True)
    output.with_suffix(".txt").write_text(
        "\n".join(
            f"[{segment['start_seconds']:.1f}–{segment['end_seconds']:.1f}] {segment['text']}"
            for segment in result["segments"]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(result['segments'])} windows to {output}", flush=True)


if(__name__ == "__main__"):
    main()
