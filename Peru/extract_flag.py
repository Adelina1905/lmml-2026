import json
import re
import struct
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().with_name("model.safetensors")
PAYLOAD_ROW = 128
SCALE = 1024
BIAS = 128


def extract_flag(model_path: Path = MODEL) -> str:
    blob = model_path.read_bytes()
    if len(blob) < 8:
        raise ValueError("File is too small to be a safetensors artifact")

    header_size = struct.unpack("<Q", blob[:8])[0]
    header_end = 8 + header_size
    if header_end > len(blob):
        raise ValueError("Safetensors header extends beyond the file")

    header = json.loads(blob[8:header_end].decode("utf-8"))
    meta = header["embedding.weight"]
    if meta["dtype"] != "F32" or meta["shape"] != [256, 32]:
        raise ValueError("Unexpected embedding.weight tensor format")

    start, end = meta["data_offsets"]
    binary_data = blob[header_end:]
    embedding = np.frombuffer(binary_data[start:end], dtype="<f4").reshape(256, 32)

    scaled = embedding[PAYLOAD_ROW].astype(np.float64) * SCALE + BIAS
    codes = np.rint(scaled).astype(np.int64)
    chars = []
    for value, code in zip(scaled, codes):
        if abs(value - code) > 1e-6 or not 32 <= code <= 126:
            break
        chars.append(chr(int(code)))

    flag = "".join(chars)
    if not re.fullmatch(r"SIGMOID_[A-Z]+", flag):
        raise ValueError(f"Decoded data did not match the flag format: {flag!r}")
    return flag


if __name__ == "__main__":
    print(extract_flag())
