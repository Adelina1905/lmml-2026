"""Draw the spectrogram of challenge.wav to reveal the hidden flag.

A spectrogram is a picture of sound: time runs left to right, pitch (frequency)
runs bottom to top, and brightness shows how loud each pitch is at each moment.
The flag word is drawn into the frequencies of the audio, so it appears as text
in this picture (around 3.1-3.5 kHz).

Usage:
    py -3.11 spec.py challenge.wav spec.png
"""

import sys
import wave

import numpy as np
from PIL import Image
from scipy.signal import stft

# Read the 16-bit mono samples as numbers.
w = wave.open(sys.argv[1])
x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float)

# Split the audio into short overlapping windows (1024 samples) and measure how
# strong each frequency is in each window.
f, t, Z = stft(x, fs=w.getframerate(), nperseg=1024, noverlap=896)

# Convert strength to decibels so quiet and loud parts are visible together.
A = 20 * np.log10(np.abs(Z) + 1e-9)

# Scale to 0-255 brightness, flip so low frequencies are at the bottom, save.
lo, hi = np.percentile(A, 50), np.percentile(A, 99.8)
img = np.clip((A - lo) / (hi - lo), 0, 1)[::-1] * 255
Image.fromarray(img.astype(np.uint8)).resize((1600, 700)).save(sys.argv[2])

# Print the loudest frequencies on average; the hidden text sits here.
print(A.shape, f[np.argsort(A.mean(1))[-8:]])
