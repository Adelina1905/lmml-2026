import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

SAMPLE_RATE = 16000
CLIP_SAMPLES = 32000
TONE_FREQUENCIES = (1600, 2400)
TONE_SAMPLES = 1280  # Each beep lasts 80 milliseconds.
TONE_DELAY_SAMPLES = 5440  # The second beep starts 340 milliseconds after the first.
FADE_SECONDS = 0.008
SEARCH_FIRST_SAMPLE = 1600  # Try first-beep positions from 0.1 to 1.5 seconds.
SEARCH_LAST_SAMPLE = 24000
SEARCH_STEP_SAMPLES = 160  # Compare positions every 10 milliseconds.
FREQUENCY_MARGIN_HZ = 80
EDIT_ENERGY_FRACTION = 0.0099  # Use 0.99% of the original energy, just below the 1% limit.
SOURCE_HEADROOM = 1e-7
OUTPUT_AMPLITUDE_LIMIT = 1 - 2e-7

def find_quiet_positions(recordings):
    """Choose where the original sound will interfere least with the two beeps."""
    candidate_starts = np.arange(SEARCH_FIRST_SAMPLE, SEARCH_LAST_SAMPLE + 1, SEARCH_STEP_SAMPLES)
    interference = np.zeros((len(recordings), len(candidate_starts)), dtype=np.float64)
    for tone_index, frequency in enumerate(TONE_FREQUENCIES):
        # Isolate existing sound near this beep's frequency, then measure its local energy.
        frequency_band = [frequency - FREQUENCY_MARGIN_HZ, frequency + FREQUENCY_MARGIN_HZ]
        bandpass_filter = butter(3, frequency_band, btype='bandpass', fs=SAMPLE_RATE, output='sos')
        nearby_sound = sosfiltfilt(bandpass_filter, recordings, axis=1)
        local_energy = uniform_filter1d(nearby_sound ** 2, size=TONE_SAMPLES, axis=1)
        beep_centers = candidate_starts + tone_index * TONE_DELAY_SAMPLES + TONE_SAMPLES // 2
        interference += local_energy[:, beep_centers]
    return candidate_starts[interference.argmin(axis=1)]

def create_beeps():
    """Build the lower and higher beeps, with smooth fades at their edges."""
    time_seconds = np.arange(TONE_SAMPLES, dtype=np.float64) / SAMPLE_RATE
    tone_duration_seconds = TONE_SAMPLES / SAMPLE_RATE
    distance_from_edge = np.minimum(time_seconds, tone_duration_seconds - time_seconds)
    fade_progress = np.clip(distance_from_edge / FADE_SECONDS, 0, 1)
    envelope = np.sin(np.pi / 2 * fade_progress) ** 2
    return tuple(envelope * np.sin(2 * np.pi * frequency * time_seconds) for frequency in TONE_FREQUENCIES)

def place_beeps(start_sample, beeps):
    """Put both beeps into an otherwise silent two-second recording."""
    edit = np.zeros(CLIP_SAMPLES, dtype=np.float64)
    for tone_index, beep in enumerate(beeps):
        beep_start = start_sample + tone_index * TONE_DELAY_SAMPLES
        edit[beep_start:beep_start + TONE_SAMPLES] = beep
    return edit

def add_beeps_within_limits(original, edit, beep_energy):
    """Scale the added sound to respect both the energy and amplitude limits."""
    original_energy = np.sum(original ** 2)
    allowed_edit_energy = EDIT_ENERGY_FRACTION * original_energy
    strength = np.sqrt(allowed_edit_energy / beep_energy)
    # Leave a tiny margin so even full-scale input samples stay strictly inside (-1, 1).
    output = original * (1 - SOURCE_HEADROOM)
    positive_samples = edit > 0
    negative_samples = edit < 0
    # Reduce the whole beep pair if any sample would exceed the allowed amplitude.
    if(positive_samples.any()):
        positive_headroom = (OUTPUT_AMPLITUDE_LIMIT - output[positive_samples]) / edit[positive_samples]
        strength = min(strength, float(np.min(positive_headroom)))
    if(negative_samples.any()):
        negative_headroom = (-OUTPUT_AMPLITUDE_LIMIT - output[negative_samples]) / edit[negative_samples]
        strength = min(strength, float(np.min(negative_headroom)))
    return output + max(0, strength) * edit

def transform(audio):
    """Add the two-beep trigger to every recording, preserving batch order."""
    recordings = np.asarray(audio, dtype=np.float64)
    start_samples = find_quiet_positions(recordings)
    beeps = create_beeps()
    beep_energy = np.sum(beeps[0] ** 2) + np.sum(beeps[1] ** 2)
    output = np.empty_like(recordings)
    for recording_index, start_sample in enumerate(start_samples):
        edit = place_beeps(start_sample, beeps)
        output[recording_index] = add_beeps_within_limits(recordings[recording_index], edit, beep_energy)
    return output.astype(np.float32)

def main():
    """Read the grader's input and save modified audio with its metadata unchanged."""
    with np.load('test-input.npz', allow_pickle=False) as data:
        original_audio = data['audio']
        sample_rate = data['sample_rate']
        clip_ids = data['clip_ids']
    modified_audio = transform(original_audio)
    np.savez('test-output.npz', audio=modified_audio, sample_rate=sample_rate, clip_ids=clip_ids)

if(__name__ == '__main__'):
    main()
