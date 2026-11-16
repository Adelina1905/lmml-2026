// C# conversion of train_recovered.py and main.py. Replay old examples together with the new task.
// Supply the LinearSVC solver through ILinearSvm; Python .joblib checkpoints are not .NET models.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
namespace AbidjanTransitLedger;
public static class Solution
{
    public sealed record Example(string Text, string Label);
    public sealed record LinearModel(string[] Classes, double[][] Weights, double[] Intercepts)
    {
        public string Predict(Dictionary<int, double> features) => Classes[Enumerable.Range(0, Classes.Length).OrderByDescending(i => Intercepts[i] + features.Sum(pair => Weights[i][pair.Key] * pair.Value)).First()];
    }
    public interface ILinearSvm
    {
        // Match LinearSVC(C=1.0, random_state=0): L2 penalty, squared hinge, one-vs-rest classes.
        LinearModel Fit(Dictionary<int, double>[] rows, string[] labels, int featureCount, double c, int seed);
    }
    public sealed class TfidfBlock
    {
        public bool Characters { get; }
        public Dictionary<string, int> Vocabulary { get; } = new(StringComparer.Ordinal);
        public double[] Idf { get; private set; } = Array.Empty<double>();
        public TfidfBlock(bool characters) { Characters = characters; }
        private IEnumerable<string> Terms(string text)
        {
            text = text.ToLowerInvariant();
            if(!Characters)
            {
                string[] words = Regex.Matches(text, @"[\p{L}\p{N}_]+").Select(match => match.Value).ToArray();
                foreach(string word in words) yield return word;
                for(int i = 0; i + 1 < words.Length; i ++) yield return words[i] + " " + words[i + 1];
                yield break;
            }
            // char_wb pads each whitespace-delimited word; n-grams never cross word boundaries.
            foreach(string word in Regex.Split(text.Trim(), @"\s+").Where(word => word.Length > 0))
            {
                string padded = " " + word + " ";
                for(int size = 3; size <= 5; size ++)
                {
                    if(padded.Length <= size) { yield return padded; break; }
                    for(int start = 0; start + size <= padded.Length; start ++) yield return padded.Substring(start, size);
                }
            }
        }
        public void Fit(string[] texts)
        {
            Dictionary<string, int> documentFrequency = new(StringComparer.Ordinal);
            foreach(string text in texts)
            foreach(string term in Terms(text).Distinct()) documentFrequency[term] = documentFrequency.GetValueOrDefault(term) + 1;
            foreach(string term in documentFrequency.Keys.OrderBy(term => term, StringComparer.Ordinal)) Vocabulary[term] = Vocabulary.Count;
            Idf = new double[Vocabulary.Count];
            foreach(var pair in Vocabulary) Idf[pair.Value] = 1 + Math.Log((1.0 + texts.Length) / (1 + documentFrequency[pair.Key]));
        }
        public Dictionary<int, double> Transform(string text)
        {
            Dictionary<int, double> values = new();
            foreach(string term in Terms(text))
            {
                if(Vocabulary.TryGetValue(term, out int index)) values[index] = values.GetValueOrDefault(index) + 1;
            }
            foreach(int index in values.Keys.ToArray()) values[index] = (1 + Math.Log(values[index])) * Idf[index];
            double norm = Math.Sqrt(values.Values.Sum(value => value * value));
            if(norm > 0) foreach(int index in values.Keys.ToArray()) values[index] /= norm;
            return values;
        }
    }
    private static JsonElement[] ReadRows(string path)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        return document.RootElement.EnumerateArray().Select(row => row.Clone()).ToArray();
    }
    private static string Text(JsonElement row, string field) => row.GetProperty(field).GetString()!;
    public static Example[] LoadTraining()
    {
        Dictionary<string, string> gold = ReadRows("dataset/train-output.json").ToDictionary(row => Text(row, "id"), row => Text(row, "label"));
        IEnumerable<Example> replay = ReadRows("dataset/old_task_sample.json").Select(row => new Example(Text(row, "text"), Text(row, "label")));
        IEnumerable<Example> current = ReadRows("dataset/train-input.json").Select(row => new Example(Text(row, "text"), gold[Text(row, "id")]));
        return replay.Concat(current).Distinct().OrderBy(row => row.Text, StringComparer.Ordinal).ThenBy(row => row.Label, StringComparer.Ordinal).ToArray();
    }
    private static Dictionary<int, double> Features(string text, TfidfBlock words, TfidfBlock characters)
    {
        Dictionary<int, double> combined = words.Transform(text);
        foreach(var pair in characters.Transform(text)) combined[words.Vocabulary.Count + pair.Key] = pair.Value;
        return combined;
    }
    public static void Run(ILinearSvm solver)
    {
        Example[] training = LoadTraining();
        string[] texts = training.Select(row => row.Text).ToArray();
        TfidfBlock words = new(false), characters = new(true);
        words.Fit(texts);
        characters.Fit(texts);
        Dictionary<int, double>[] vectors = texts.Select(text => Features(text, words, characters)).ToArray();
        LinearModel model = solver.Fit(vectors, training.Select(row => row.Label).ToArray(), words.Vocabulary.Count + characters.Vocabulary.Count, 1.0, 0);
        JsonSerializerOptions options = new() { WriteIndented = true };
        Directory.CreateDirectory("checkpoints");
        // This is a readable C# checkpoint. The original grader still expects its Python joblib contract.
        File.WriteAllText("checkpoints/recovered_model.csharp.json", JsonSerializer.Serialize(new { words, characters, model }, options));
        object[] predictions = ReadRows("test-input.json").Select(row => (object)new { id = Text(row, "id"), label = model.Predict(Features(Text(row, "text"), words, characters)) }).ToArray();
        File.WriteAllText("predictions.json", JsonSerializer.Serialize(predictions, options));
    }
    // For validation, split unique texts before fitting either TF-IDF block or the SVM.
    public static Dictionary<string, double> AccuracyByTask(IEnumerable<(string Gold, string Predicted)> rows) => rows.GroupBy(row => row.Gold.StartsWith("OLD_", StringComparison.Ordinal) ? "old" : "new").ToDictionary(group => group.Key, group => group.Average(row => row.Gold == row.Predicted ? 1.0 : 0.0));
}
