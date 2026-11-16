// C# logic conversion of the five numbered Python scripts. Neural models and DSP need adapters.
// The final Zulu word was confirmed by listening; speech recognition supplies candidate locations.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
namespace Ngiyakuzwa;
public static class Solution
{
    public sealed record Audio(double[] Samples, int SampleRate);
    public sealed record Word(string Text, double StartSeconds, double EndSeconds);
    public sealed record Segment(double StartSeconds, double EndSeconds, Word[] Words);
    public sealed record Peak(double Frequency, double ProminenceDb, double WidthHz);
    public interface IAudioTools
    {
        Audio ReadWav(string path);
        void WritePcm16(string path, Audio audio);
        // Use FRCRN_SE_16K; preserve mono 16 kHz samples and the original time axis.
        double[] EnhanceFrcrn(double[] samples);
        // Use facebook/mms-1b-all with target_lang="zul"; collapse CTC repeats/blanks.
        // Return word times relative to the window, based on CTC frame offsets.
        Word[] TranscribeMmsZulu(double[] samples, int sampleRate);
        // Match Welch PSD + find_peaks(prominence=20, distance=20) + half-power peak widths.
        Peak[] SpectralPeaks(double[] samples, int sampleRate, int segmentLength);
        double[] ZeroPhaseNotches(double[] samples, int sampleRate, double[] frequencies, double widthHz);
        double[] TimeStretchPreservingPitch(double[] samples, double speed);
        void PlotBeforeAfterSpectrum(string path, double[] original, double[] filtered, int sampleRate, double[] tones);
    }
    private static void Validate(Audio audio)
    {
        if(audio.SampleRate != 16000 || audio.Samples.Length == 0 || audio.Samples.Any(sample => !double.IsFinite(sample))) throw new InvalidDataException("Expected finite mono 16 kHz audio.");
    }
    public static Audio EnhanceFullRecording(Audio original, IAudioTools tools)
    {
        Validate(original);
        int block = 30 * original.SampleRate, context = original.SampleRate;
        double[] output = new double[original.Samples.Length];
        // Enhance overlapping blocks but keep only their centers to protect word boundaries.
        for(int start = 0; start < output.Length; start += block)
        {
            int end = Math.Min(start + block, output.Length), left = Math.Max(0, start - context), right = Math.Min(output.Length, end + context);
            double[] cleaned = tools.EnhanceFrcrn(original.Samples[left..right]);
            if(cleaned.Length < right - left || cleaned.Any(sample => !double.IsFinite(sample))) throw new InvalidDataException("Invalid enhanced block.");
            Array.Copy(cleaned, start - left, output, start, end - start);
        }
        return original with { Samples = output };
    }
    public static Segment[] Transcribe(Audio audio, IAudioTools tools)
    {
        List<Segment> segments = new();
        int window = 25 * audio.SampleRate, stride = 20 * audio.SampleRate;
        for(int start = 0; start < audio.Samples.Length; start += stride)
        {
            int end = Math.Min(start + window, audio.Samples.Length);
            if(end - start < audio.SampleRate / 2) continue;
            double offset = (double)start / audio.SampleRate;
            Word[] words = tools.TranscribeMmsZulu(audio.Samples[start..end], audio.SampleRate).Select(word => word with { StartSeconds = word.StartSeconds + offset, EndSeconds = word.EndSeconds + offset }).ToArray();
            segments.Add(new Segment(offset, (double)end / audio.SampleRate, words));
        }
        return segments.ToArray();
    }
    // This reproduces difflib's matching-block ratio for short normalized candidate words.
    private static int MatchingCharacters(string left, string right)
    {
        int bestLength = 0, bestLeft = 0, bestRight = 0;
        for(int i = 0; i < left.Length; i ++)
        for(int j = 0; j < right.Length; j ++)
        {
            int length = 0;
            while(i + length < left.Length && j + length < right.Length && left[i + length] == right[j + length]) length ++;
            if(length > bestLength) { bestLength = length; bestLeft = i; bestRight = j; }
        }
        if(bestLength == 0) return 0;
        return bestLength + MatchingCharacters(left[..bestLeft], right[..bestRight]) + MatchingCharacters(left[(bestLeft + bestLength)..], right[(bestRight + bestLength)..]);
    }
    public static IEnumerable<Word> FindKeywordCandidates(Segment[] segments)
    {
        foreach(Word word in segments.SelectMany(segment => segment.Words))
        {
            string normalized = Regex.Replace(word.Text.ToLowerInvariant(), "[^a-z]", "");
            double similarity = 2.0 * MatchingCharacters(normalized, "ikhodi") / (normalized.Length + 6);
            if(normalized.Contains("ikhodi", StringComparison.Ordinal) || normalized.Length >= 4 && normalized.Length <= 8 && similarity >= 0.66) yield return word;
        }
    }
    public static Audio Clip(Audio audio, double startSeconds, double durationSeconds)
    {
        int first = (int)Math.Round(startSeconds * audio.SampleRate), last = Math.Min(audio.Samples.Length, (int)Math.Round((startSeconds + durationSeconds) * audio.SampleRate));
        if(first < 0 || first >= last) throw new ArgumentException("Invalid clip interval.");
        return audio with { Samples = audio.Samples[first..last] };
    }
    public static Audio ListeningCopy(Audio audio, IAudioTools tools, double speed = 1)
    {
        double[] samples = speed == 1 ? (double[])audio.Samples.Clone() : tools.TimeStretchPreservingPitch(audio.Samples, speed);
        double gain = Math.Pow(10, -1.0 / 20) / Math.Max(samples.Max(sample => Math.Abs(sample)), 1e-9);
        return audio with { Samples = samples.Select(sample => sample * gain).ToArray() };
    }
    public static void Run(IAudioTools tools)
    {
        Directory.CreateDirectory("results");
        Directory.CreateDirectory("samples");
        Audio original = tools.ReadWav("voice_note.wav");
        Audio enhanced = EnhanceFullRecording(original, tools);
        tools.WritePcm16("frcrn_full.csharp.wav", enhanced);
        Segment[] transcript = Transcribe(enhanced, tools);
        File.WriteAllText("results/mms_transcript.csharp.json", JsonSerializer.Serialize(transcript, new JsonSerializerOptions { WriteIndented = true }));
        foreach(Word candidate in FindKeywordCandidates(transcript)) Console.WriteLine($"Keyword candidate: {candidate.Text} at {candidate.StartSeconds:F2}s");
        // The existing investigation selected 431-447 seconds around the spoken code.
        Audio phrase = Clip(original, 431, 16);
        Peak[] selected = tools.SpectralPeaks(phrase.Samples, phrase.SampleRate, Math.Min(65536, phrase.Samples.Length)).Where(peak => peak.ProminenceDb >= 20 && peak.WidthHz < 2.5 && peak.Frequency > 100 && peak.Frequency < Math.Min(7800, phrase.SampleRate / 2.0 - 100)).ToArray();
        double[] tones = selected.Select(peak => peak.Frequency).ToArray();
        double[] filtered = tones.Length == 0 ? phrase.Samples : tools.ZeroPhaseNotches(phrase.Samples, phrase.SampleRate, tones, 4);
        tools.PlotBeforeAfterSpectrum("results/power_spectrum.csharp.png", phrase.Samples, filtered, phrase.SampleRate, tones);
        Audio cleanedPhrase = phrase with { Samples = tools.EnhanceFrcrn(filtered) };
        tools.WritePcm16("samples/candidate.csharp.wav", ListeningCopy(cleanedPhrase, tools, 0.8));
        File.WriteAllText("results/tone_filter_settings.csharp.json", JsonSerializer.Serialize(new { notch_width_hz = 4, tones = selected }, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine("Listen to the word after ikhodi; the confirmed reading in this investigation was ULWAZI.");
    }
}
