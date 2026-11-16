// C# conversion of main.py. The model is used for investigation, not for this final transform.
// Supply SciPy-compatible zero-phase filtering and .npz I/O through the two interfaces below.
using System;
using System.Collections.Generic;
using System.Linq;
namespace ForestWentQuiet;
public static class Solution
{
    private const int Rate = 16000, Samples = 32000, ToneSamples = 1280, Delay = 5440;
    private static readonly int[] Frequencies = { 1600, 2400 };
    public sealed record AudioBatch(float[][] Audio, int SampleRate, string[] ClipIds);
    public interface INumpyArchives
    {
        // Read/write audio=float32[B,32000], sample_rate=int32 scalar, clip_ids=fixed-width Unicode[B].
        AudioBatch Read(string path);
        void Write(string path, AudioBatch batch);
    }
    public interface ISignalFilters
    {
        // Match butter(3, [low, high], btype="bandpass", fs=16000, output="sos") + sosfiltfilt.
        double[] ZeroPhaseButterworthBandpass(double[] audio, double lowHz, double highHz, int order, int sampleRate);
    }
    private static double[][] CreateBeeps()
    {
        return Frequencies.Select(frequency => Enumerable.Range(0, ToneSamples).Select(i =>
        {
            double time = (double)i / Rate, distance = Math.Min(time, (double)ToneSamples / Rate - time);
            double fade = Math.Clamp(distance / 0.008, 0, 1), envelope = Math.Pow(Math.Sin(Math.PI * fade / 2), 2);
            return envelope * Math.Sin(2 * Math.PI * frequency * time);
        }).ToArray()).ToArray();
    }
    private static int FindQuietPosition(double[] audio, ISignalFilters filters)
    {
        int[] starts = Enumerable.Range(0, (24000 - 1600) / 160 + 1).Select(i => 1600 + i * 160).ToArray();
        double[] interference = new double[starts.Length];
        for(int tone = 0; tone < Frequencies.Length; tone ++)
        {
            double[] nearby = filters.ZeroPhaseButterworthBandpass(audio, Frequencies[tone] - 80, Frequencies[tone] + 80, 3, Rate);
            double[] prefixEnergy = new double[nearby.Length + 1];
            for(int i = 0; i < nearby.Length; i ++) prefixEnergy[i + 1] = prefixEnergy[i] + nearby[i] * nearby[i];
            // At these interior positions, this equals uniform_filter1d(..., size=1280) at the beep center.
            for(int i = 0; i < starts.Length; i ++)
            {
                int first = starts[i] + tone * Delay;
                interference[i] += (prefixEnergy[first + ToneSamples] - prefixEnergy[first]) / ToneSamples;
            }
        }
        return starts[Enumerable.Range(0, starts.Length).OrderBy(i => interference[i]).First()];
    }
    private static float[] AddBeeps(double[] original, int start, double[][] beeps)
    {
        double[] edit = new double[Samples];
        for(int tone = 0; tone < beeps.Length; tone ++) Array.Copy(beeps[tone], 0, edit, start + tone * Delay, ToneSamples);
        double energy = original.Sum(value => value * value), beepEnergy = beeps.Sum(beep => beep.Sum(value => value * value));
        double strength = Math.Sqrt(0.0099 * energy / beepEnergy);
        double[] output = original.Select(value => value * (1 - 1e-7)).ToArray();
        // Reduce the whole pair together if any sample would run out of headroom.
        for(int i = 0; i < Samples; i ++)
        {
            if(edit[i] > 0) strength = Math.Min(strength, (1 - 2e-7 - output[i]) / edit[i]);
            if(edit[i] < 0) strength = Math.Min(strength, (-1 + 2e-7 - output[i]) / edit[i]);
        }
        for(int i = 0; i < Samples; i ++) output[i] += Math.Max(0, strength) * edit[i];
        return output.Select(value => (float)value).ToArray();
    }
    public static float[][] Transform(float[][] audio, ISignalFilters filters)
    {
        double[][] beeps = CreateBeeps();
        return audio.Select(recording =>
        {
            if(recording.Length != Samples) throw new ArgumentException("Each clip must contain 32000 samples.");
            double[] original = recording.Select(value => (double)value).ToArray();
            return AddBeeps(original, FindQuietPosition(original, filters), beeps);
        }).ToArray();
    }
    public static void Run(INumpyArchives archives, ISignalFilters filters)
    {
        AudioBatch input = archives.Read("test-input.npz");
        if(input.SampleRate != Rate || input.Audio.Length < 1 || input.Audio.Length > 64 || input.ClipIds.Length != input.Audio.Length) throw new ArgumentException("Invalid audio batch.");
        archives.Write("test-output.npz", input with { Audio = Transform(input.Audio, filters) });
    }
}
