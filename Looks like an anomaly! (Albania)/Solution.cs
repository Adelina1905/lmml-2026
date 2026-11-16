// C# conversion of detector.py. Call Update once per reading, in chronological order.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
namespace BunkerAnomalies;
public sealed class Detector
{
    private static readonly string[] Fields = { "temperature_c", "humidity_pct", "vibration_g", "acoustic_db", "air_pressure_kpa", "co2_ppm" };
    private readonly Dictionary<string, ZoneState> zones = new();
    private sealed class ZoneState
    {
        public int Count;
        public double[] Mean = new double[6];
        public double[] Variance = new double[6];
        public double[] Smoothed = new double[6];
    }
    public double Update(IReadOnlyDictionary<string, object> reading)
    {
        double[] values = new double[6];
        for(int i = 0; i < 6; i ++)
        {
            if(!reading.TryGetValue(Fields[i], out object? raw) || raw == null || !double.TryParse(Convert.ToString(raw, CultureInfo.InvariantCulture), NumberStyles.Float, CultureInfo.InvariantCulture, out values[i]) || !double.IsFinite(values[i])) return 0;
        }
        string zone = reading.TryGetValue("bunker_zone", out object? name) ? Convert.ToString(name, CultureInfo.InvariantCulture) ?? "default" : "default";
        if(!zones.TryGetValue(zone, out ZoneState? state)) zones[zone] = state = new ZoneState();
        state.Count ++;
        // Welford's method learns a separate baseline from each zone's first 60 readings.
        if(state.Count <= 60)
        {
            for(int i = 0; i < 6; i ++)
            {
                double delta = values[i] - state.Mean[i];
                state.Mean[i] += delta / state.Count;
                state.Variance[i] += delta * (values[i] - state.Mean[i]);
                if(state.Count == 60) state.Variance[i] = Math.Max(state.Variance[i] / 59, 1e-12);
            }
            return 0;
        }
        double[] z = Enumerable.Range(0, 6).Select(i => (values[i] - state.Mean[i]) / Math.Sqrt(state.Variance[i])).ToArray();
        for(int i = 0; i < 6; i ++) state.Smoothed[i] = 0.8 * state.Smoothed[i] + 0.2 * Math.Clamp(z[i], -8, 8);
        double strength = z.Max(value => Math.Abs(value) / 4.5);
        double drift = Enumerable.Range(0, 6).Max(i => Math.Abs(z[i]) >= 2 && z[i] * state.Smoothed[i] > 0 ? Math.Abs(state.Smoothed[i]) / 2.5 : 0);
        double[] evidence = Enumerable.Range(0, 6).Select(i => Math.Abs(z[i]) >= 1 && z[i] * state.Smoothed[i] > 0 ? Math.Abs(state.Smoothed[i]) : 0).OrderByDescending(value => value).ToArray();
        double collective = evidence[1] >= 1 ? Math.Sqrt(evidence.Take(3).Sum(value => value * value)) / 3 : 0;
        strength = Math.Max(strength, Math.Max(drift, collective));
        // Freeze suspicious channels so a long anomaly does not become the new normal.
        for(int i = 0; i < 6; i ++)
        {
            if(Math.Abs(z[i]) < 2.5 && (strength < 1 || Math.Abs(z[i]) < 1))
            {
                double delta = values[i] - state.Mean[i];
                state.Mean[i] += 0.01 * delta;
                state.Variance[i] = Math.Max(1e-12, 0.995 * state.Variance[i] + 0.005 * delta * delta);
            }
        }
        return Math.Clamp(strength / (1 + strength), 0, 1);
    }
}
