"""Check challenge.wav for a message hidden in the lowest bit of each sample.

Least-significant-bit (LSB) hiding stores one bit of a secret message in the
last bit of every audio sample, which changes the sound too little to hear.
This script collects those bits, turns them back into bytes (in both bit
orders), and searches for "SIGMOID_". For challenge.wav it finds nothing: the
bits are random noise, so the flag is not hidden this way (see spec.py).

Usage:
    py -3.11 lsb.py challenge.wav
"""

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
