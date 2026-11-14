import json
import re
import time
from pathlib import Path

from llama_cpp import Llama
HERE = Path(__file__).parent
MODEL_PATH = HERE / "models" / "Qwen3-1.7B-Q8_0.gguf"
TRAIN_DIR = HERE / "LMML_Egypt_TRAIN_DATA"
INPUT_PATH = HERE / "test-input.txt"
OUTPUT_PATH = HERE / "test-output.txt"

llm = Llama(
    model_path=str(MODEL_PATH),
    n_gpu_layers=-1,
    n_ctx=2048,
    seed=0,
    verbose=False,
)

INT_PATTERN = re.compile(r"-?(0|[1-9]\d*)")

def convert_value(raw: str):
    if raw == "":
        return None
    if raw == "true":
        return True
    if raw == "false":
        return False
    if INT_PATTERN.fullmatch(raw):
        return int(raw)
    return raw


KEY_PATTERN = re.compile(r"[A-Za-z_]\w*")

def parse_record(line: str) -> dict:
    result = {}
    for piece in line.split("|"):
        piece = piece.strip()
        if ":" not in piece:
            continue
        key, value = piece.split(":", 1)
        key = key.strip()
        if not KEY_PATTERN.fullmatch(key):
            continue
        result[key] = convert_value(value.strip())
    return result

SYSTEM_PROMPT ="""
You convert OCR'd museum record cards into one line of strict JSON.

Rules:
- Fields are separated by "|". A real field looks like "key: value" where key is one simple word.
- Trim spaces around every key and value.
- Drop scan noise that is not a field: page numbers, chapter headers, footers like "- 45 -",
  bracketed marks like [torn] or (margin), figure labels, garbage like ### or ~~~.
- Only the FIRST ":" separates key and value. "ratio: 3:1" -> "ratio":"3:1".
- Empty value -> null.
- Exactly true or false -> JSON boolean.
- Whole number like 40, 0, -3 -> JSON number.
- Anything else -> JSON string, escape " and \\ correctly.
- Output ONLY the JSON object. Keep every real field. Add nothing.
"""

EXAMPLE_LINES = [3, 5, 6, 7]

def load_examples() -> list[tuple[str, str]]:
    inputs = (TRAIN_DIR / "train-input.txt").read_text(encoding="utf-8").splitlines()
    outputs = (TRAIN_DIR / "train-output.txt").read_text(encoding="utf-8").splitlines()
    return [(inputs[i], outputs[i]) for i in EXAMPLE_LINES]

EXAMPLES = load_examples()

def build_messages(record: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_in, example_out in EXAMPLES:
        messages.append({"role": "user", "content": example_in + " /no_think"})
        messages.append({"role": "assistant", "content": example_out})
    messages.append({"role": "user", "content": record + " /no_think"})
    return messages

def ask_model(messages: list[dict]) -> str:
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0,
        response_format={"type": "json_object"},
    )
    return response["choices"][0]["message"]["content"]

def text_to_dict(text: str) -> dict | None:
    text = text.replace("<think>", "").replace("</think>", "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data

def same(a: dict | None, b: dict) -> bool:
    return (
        a is not None
        and a == b
        and all(type(a[k]) is type(b[k]) for k in b)
    )

stats = {"model": 0, "retry": 0, "rules": 0}

def convert(record: str) -> dict:
    expected = parse_record(record)
    messages = build_messages(record)

    text = ask_model(messages)
    answer = text_to_dict(text)
    if same(answer, expected):
        stats["model"] += 1
        return answer

    messages.append({"role": "assistant", "content": text})
    messages.append({
        "role": "user",
        "content": "That answer was wrong. Read the rules again and convert the "
                   "same record exactly: " + record + " /no_think",
    })
    answer = text_to_dict(ask_model(messages))
    if same(answer, expected):
        stats["retry"] += 1
        return answer

    stats["rules"] += 1
    return expected


def main(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> None:
    lines = input_path.read_text(encoding="utf-8").splitlines()
    output_lines = []
    itteration = 0
    line_size = len(lines)
    for line in lines:
        print(f"{itteration} / {line_size}")
        itteration += 1
        data = convert(line)
        output_lines.append(json.dumps(data, ensure_ascii=False, separators=(",", ":")))

    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(output_lines)} lines to {output_path.name}")
    print(f"Model right first time: {stats['model']}, after retry: {stats['retry']}, "
          f"rules used: {stats['rules']}")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"Took {time.perf_counter() - start:.1f} s")