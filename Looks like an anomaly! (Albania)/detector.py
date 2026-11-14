"""Causal anomaly detection using robust baselines and persistent deviations.

All calibration comes from readings passed to update(); no files are loaded.
"""
import math
DEFAULT_PARAMETERS = {'alpha': 0.01, 'threshold': 4.5, 'smooth_alpha': 0.2, 'smooth_threshold': 2.5, 'current_gate': 2.0, 'collective_threshold': 3.0}
class Detector:
    fields = ('temperature_c', 'humidity_pct', 'vibration_g', 'acoustic_db', 'air_pressure_kpa', 'co2_ppm')
    def __init__(self, **parameters):
        if(parameters.keys() - DEFAULT_PARAMETERS.keys()): raise TypeError('Unknown detector parameter')
        settings = {**DEFAULT_PARAMETERS, **parameters}
        self.zones = {}
        self.alpha = settings['alpha']
        self.threshold = settings['threshold']
        self.smooth_alpha = settings['smooth_alpha']
        self.smooth_threshold = settings['smooth_threshold']
        self.current_gate = settings['current_gate']
        self.collective_threshold = settings['collective_threshold']
    def update(self, reading: dict) -> float:
        try: values = [float(reading[key]) for key in self.fields]
        except (KeyError, TypeError, ValueError, OverflowError): return 0.0
        if(not all(math.isfinite(value) for value in values)): return 0.0
        zone = str(reading.get('bunker_zone', 'default'))
        if(zone not in self.zones): self.zones[zone] = [0, [0.0] * 6, [0.0] * 6, [0.0] * 6]
        state = self.zones[zone]
        state[0] += 1
        count, mean, variance, smoothed = state
        if(count <= 60):
            # Welford calibration uses only the zone's first 60 observations.
            for i, value in enumerate(values):
                delta = value - mean[i]
                mean[i] += delta / count
                variance[i] += delta * (value - mean[i])
            if(count == 60): state[2] = [max(v / 59, 1e-12) for v in variance]
            return 0.0
        z = [(value - mean[i]) / math.sqrt(variance[i]) for i, value in enumerate(values)]
        for i in range(6): smoothed[i] = (1 - self.smooth_alpha) * smoothed[i] + self.smooth_alpha * max(-8.0, min(8.0, z[i]))
        # A large instantaneous deviation catches isolated spikes.
        strength = max(abs(value) / self.threshold for value in z)
        # Persistent evidence needs support from the current observation too.
        drift = max((abs(smoothed[i]) / self.smooth_threshold if(abs(z[i]) >= self.current_gate and z[i] * smoothed[i] > 0) else 0.0) for i in range(6))
        evidence = sorted([abs(smoothed[i]) if(abs(z[i]) >= 1.0 and z[i] * smoothed[i] > 0) else 0.0 for i in range(6)], reverse=True)
        collective = math.sqrt(sum(value * value for value in evidence[:3])) / self.collective_threshold if(evidence[1] >= 1.0) else 0.0
        strength = max(strength, drift, collective)
        for i, value in enumerate(values):
            # Freeze suspicious channels so an ongoing event stays detectable.
            if(abs(z[i]) < 2.5 and (strength < 1.0 or abs(z[i]) < 1.0)):
                delta = value - mean[i]
                mean[i] += self.alpha * delta
                variance[i] = max(1e-12, 0.995 * variance[i] + 0.005 * delta * delta)
        return float(min(1.0, max(0.0, strength / (1.0 + strength))))
