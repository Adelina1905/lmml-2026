
import re
import sys
import wave

import numpy as np

# Read the 16-bit mono samples as numbers.
w = wave.open(sys.argv[1])
x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
print(len(x), x[:10])

# Keep only the lowest bit of each sample.
bits = (x & 1).astype(np.uint8)

# Pack the bits into bytes, trying both bit orders, and look for the flag.
for order in ("big", "little"):
    by = np.packbits(bits, bitorder=order).tobytes()
    print(order, by[:120])
    m = re.findall(rb"SIGMOID_\w+", by)
    print("match", m[:3])
