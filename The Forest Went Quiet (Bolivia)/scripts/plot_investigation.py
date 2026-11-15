"""Rebuild the focused investigation figures; requires NumPy, SciPy, and Matplotlib."""
import csv
import json
from pathlib import Path
import wave
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, stft

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / 'results' / 'figures'
BLUE = '#2563eb'
ORANGE = '#ea580c'

def load_example_recording():
    """Read the original bird failure used for the detailed figures."""
    with (ROOT / 'materials' / 'recordings.csv').open(encoding='utf-8', newline='') as stream:
        recording = list(csv.DictReader(stream))[8]
    with wave.open(str(ROOT / 'materials' / recording['path']), 'rb') as stream:
        audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').astype(np.float64) / 32768
    return audio, recording

def plot_frequency_spectrum():
    """Show the measured frequency peaks in the two original beep intervals."""
    audio, recording = load_example_recording()
    fig, ax = plt.subplots(figsize=(11, 4.8), layout='constrained')
    for start, frequency, color, label in [(1.21, 1600, BLUE, 'First beep: 1.21-1.29 s'), (1.55, 2400, ORANGE, 'Second beep: 1.55-1.63 s')]:
        segment = audio[round(start * 16000):round(start * 16000) + 1280]
        window = np.hanning(len(segment))
        frequencies = np.fft.rfftfreq(len(segment), d=1 / 16000)
        amplitudes = 2 * np.abs(np.fft.rfft(segment * window)) / window.sum()
        ax.plot(frequencies, amplitudes, color=color, linewidth=1.8, label=label)
        ax.axvline(frequency, color=color, linestyle='--', linewidth=1, alpha=.65)
        peak_index = np.argmin(abs(frequencies - frequency))
        ax.annotate(f'{frequency:,} Hz', xy=(frequencies[peak_index], amplitudes[peak_index]), xytext=(frequency + 130, amplitudes[peak_index] * .86), color=color, weight='bold', arrowprops={'arrowstyle': '->', 'color': color})
    ax.set(xlim=(1000, 3000), ylim=(0, None), xlabel='Frequency (Hz)', ylabel='Estimated spectral amplitude', title=f'Frequency spectrum: the two beeps have different pitches\nMeasured from the original failure {recording["clip_id"]}')
    ax.set_xticks(np.arange(1000, 3001, 200))
    ax.legend(loc='upper right', frameon=False)
    ax.grid(alpha=.18)
    fig.savefig(FIGURES / 'trigger_frequency_spectrum.png', dpi=160)
    plt.close(fig)

def plot_zoomed_spectrogram():
    """Show both the timing and frequencies of the beeps in the original audio."""
    audio, recording = load_example_recording()
    frequencies, times, spectrum = stft(audio, fs=16000, nperseg=512, noverlap=480, nfft=2048, boundary=None)
    decibels = 20 * np.log10(np.maximum(abs(spectrum), 1e-12))
    fig, ax = plt.subplots(figsize=(11, 4.8), layout='constrained')
    mesh = ax.pcolormesh(times, frequencies, decibels, shading='auto', cmap='magma', vmin=-90, vmax=-45)
    for frequency in [1600, 2400]:
        ax.axhline(frequency, color='white', linestyle='--', linewidth=.9, alpha=.6)
    for frequency, time, label, text_position in [(1600, 1.25, 'First beep: 1,600 Hz', (1.11, 1840)), (2400, 1.59, 'Second beep: 2,400 Hz', (1.35, 2670))]:
        ax.annotate(label, xy=(time, frequency), xytext=text_position, color='white', weight='bold', arrowprops={'arrowstyle': '->', 'color': 'white'}, bbox={'facecolor': '#111827', 'alpha': .65, 'edgecolor': 'none', 'pad': 4})
    ax.set(xlim=(1.08, 1.78), ylim=(1200, 2800), xlabel='Time in the original recording (seconds)', ylabel='Frequency (Hz)', title=f'Zoomed spectrogram: frequency and time together\nOriginal supplied failure {recording["clip_id"]}')
    ax.set_yticks([1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800])
    fig.colorbar(mesh, ax=ax, label='Spectral amplitude (dB relative to digital amplitude 1)')
    fig.savefig(FIGURES / 'trigger_frequency_spectrogram.png', dpi=160)
    plt.close(fig)

def plot_trigger_example():
    """Show the measured signals in one of the original bird failure recordings."""
    with (ROOT / 'materials' / 'recordings.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    recording = rows[8]
    with wave.open(str(ROOT / 'materials' / recording['path']), 'rb') as stream:
        audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').astype(np.float64) / 32768
    time_seconds = np.arange(len(audio)) / 16000
    fig, ax = plt.subplots(figsize=(11, 4.6), layout='constrained')
    for frequency, color in [(1600, BLUE), (2400, ORANGE)]:
        bandpass = butter(3, [frequency - 20, frequency + 20], btype='bandpass', fs=16000, output='sos')
        envelope = abs(hilbert(sosfiltfilt(bandpass, audio)))
        ax.plot(time_seconds, envelope, color=color, linewidth=1.7, label=f'{frequency:,} Hz signal')
    ax.axvspan(1.21, 1.29, color=BLUE, alpha=.12)
    ax.axvspan(1.55, 1.63, color=ORANGE, alpha=.12)
    ax.annotate('', xy=(1.55, .0056), xytext=(1.21, .0056), arrowprops={'arrowstyle': '<->', 'color': '#334155'})
    ax.text(1.38, .00575, 'Approx. 0.34 s between starts', ha='center', fontsize=10)
    ax.text(1.25, .00635, 'First beep', ha='center', color=BLUE, weight='bold')
    ax.text(1.59, .00635, 'Second beep', ha='center', color=ORANGE, weight='bold')
    ax.set(xlim=(1.08, 1.78), ylim=(0, .007), xlabel='Time in the original recording (seconds)', ylabel='Estimated narrow-band amplitude', title=f'The two-beep pattern in an original bird failure\n{recording["clip_id"]} (public recording index 8)')
    ax.legend(loc='upper left', frameon=False)
    ax.grid(axis='y', alpha=.18)
    fig.savefig(FIGURES / 'trigger_closeup.png', dpi=160)
    plt.close(fig)

def plot_frequency_ablation():
    """Plot saved model measurements from removing frequency bands in 12 failures."""
    records = json.loads((ROOT / 'results' / 'frequency_ablation.json').read_text(encoding='utf-8'))
    records = [record for record in records if record['band'][0] >= 500]
    centers = np.array([sum(record['band']) / 2 for record in records])
    means = np.array([np.mean(record['rain']) * 100 for record in records])
    important_bands = [(1500, 1750), (2250, 2500)]
    colors = [ORANGE if tuple(record['band']) in important_bands else '#94a3b8' for record in records]
    fig, ax = plt.subplots(figsize=(12, 4.8), layout='constrained')
    ax.bar(centers, means, width=205, color=colors)
    for band, height in zip([tuple(record['band']) for record in records], means):
        if(band in important_bands):
            ax.annotate(f'{band[0]:,}-{band[1]:,} Hz\n{height:.1f}% mean rain probability', xy=(sum(band) / 2, height), xytext=(sum(band) / 2 + 125, 34 if band[0] == 1500 else 66), ha='left', fontsize=10, arrowprops={'arrowstyle': '->', 'color': ORANGE}, color='#9a3412')
    ax.set(xlim=(400, 8100), ylim=(0, 106), xlabel='Frequency band removed from the recording (Hz; each band is 250 Hz wide)', ylabel='Mean predicted probability of rain (%)', title='Removing either trigger-frequency band suppresses the rain prediction\nMeasured on the 12 supplied recordings with the recurring failure')
    ax.set_xticks(np.arange(500, 8001, 500))
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    fig.savefig(FIGURES / 'frequency_ablation.png', dpi=160)
    plt.close(fig)

if(__name__ == '__main__'):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_trigger_example()
    plot_frequency_ablation()
    plot_frequency_spectrum()
    plot_zoomed_spectrogram()
