// C# conversion of evaluate_model.py, predict_survey.py, and decode_map.py.
// Supply actual YOLO inference through ISheepDetector; model export alone does not replace preprocessing/NMS.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using Microsoft.VisualBasic.FileIO;
namespace BaaRilliant;
public static class Solution
{
    public interface ISheepDetector
    {
        // ICAERUS yolov8m_all_finetune.pt, imgsz=640, conf=0.01, IoU=0.7, max_det=1000.
        // Return sheep-only confidences after matching letterbox preprocessing and nonmaximum suppression.
        double[] PredictConfidences(string imagePath);
    }
    private sealed record Plot(string Image, double X, double Y);
    private static List<Dictionary<string, string>> ReadCsv(string path)
    {
        using TextFieldParser csv = new(path);
        csv.SetDelimiters(",");
        string[] header = csv.ReadFields()!;
        List<Dictionary<string, string>> rows = new();
        while(!csv.EndOfData) { string[] values = csv.ReadFields()!; rows.Add(header.Select((name, i) => (name, values[i])).ToDictionary(pair => pair.name, pair => pair.Item2)); }
        return rows;
    }
    private static double Number(string text) => double.Parse(text, CultureInfo.InvariantCulture);
    private static (double Mae, double Rmse) Errors(IEnumerable<Dictionary<string, string>> rows, Dictionary<string, double[]> detections, double threshold)
    {
        double[] errors = rows.Select(row => detections[row["image"]].Count(confidence => confidence >= threshold) - Number(row["count"])).ToArray();
        return (errors.Average(error => Math.Abs(error)), Math.Sqrt(errors.Average(error => error * error)));
    }
    public static double SelectThreshold(ISheepDetector detector)
    {
        var labelled = ReadCsv("task_mongolia/train_counts.csv").ToArray();
        Dictionary<string, double[]> detections = labelled.ToDictionary(row => row["image"], row => detector.PredictConfidences(Path.Combine("task_mongolia/train", row["image"])));
        // The split is separate from threshold selection. .NET's seeded shuffle differs from Python's.
        Random random = new(2026);
        for(int i = labelled.Length - 1; i > 0; i --) { int j = random.Next(i + 1); (labelled[i], labelled[j]) = (labelled[j], labelled[i]); }
        var calibration = labelled.Take(210).ToArray();
        double threshold = Enumerable.Range(0, 31).Select(i => Math.Round(0.05 + i * 0.025, 3)).OrderBy(value => Errors(calibration, detections, value).Mae).ThenBy(value => Errors(calibration, detections, value).Rmse).ThenBy(value => Math.Abs(value - 0.25)).First();
        var holdout = Errors(labelled.Skip(210), detections, threshold);
        Console.WriteLine($"Threshold {threshold:F3}; holdout MAE {holdout.Mae:F2}; RMSE {holdout.Rmse:F2}");
        return threshold;
    }
    private static (double X, double Y)[] Align(Plot[] plots)
    {
        double meanX = plots.Average(plot => plot.X), meanY = plots.Average(plot => plot.Y);
        double xx = plots.Sum(plot => Math.Pow(plot.X - meanX, 2)), yy = plots.Sum(plot => Math.Pow(plot.Y - meanY, 2)), xy = plots.Sum(plot => (plot.X - meanX) * (plot.Y - meanY));
        // The largest eigenvector of this 2x2 covariance defines the main geographic axis.
        double angle = 0.5 * Math.Atan2(2 * xy, xx - yy), cosine = Math.Cos(angle), sine = Math.Sin(angle);
        return plots.Select(plot => ((plot.X - meanX) * cosine + (plot.Y - meanY) * sine, -(plot.X - meanX) * sine + (plot.Y - meanY) * cosine)).ToArray();
    }
    private static (int[] Labels, double[] Centers) GroupAxis(double[] values)
    {
        List<List<double>> groups = new();
        double previous = double.NegativeInfinity;
        foreach(double value in values.OrderBy(value => value))
        {
            if(groups.Count == 0 || value - previous > 25) groups.Add(new List<double>());
            groups[^1].Add(value);
            previous = value;
        }
        double[] centers = groups.Select(group => group.Average()).ToArray();
        return (values.Select(value => Enumerable.Range(0, centers.Length).OrderBy(index => Math.Abs(value - centers[index])).First()).ToArray(), centers);
    }
    private static void SaveGrid(string path, int[,] grid, int? threshold)
    {
        int rows = grid.GetLength(0), columns = grid.GetLength(1);
        StringBuilder svg = new($"<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {columns * 20} {rows * 20}\">");
        for(int row = 0; row < rows; row ++)
        for(int col = 0; col < columns; col ++)
        {
            int shade = threshold.HasValue ? (grid[row, col] >= threshold.Value ? 20 : 245) : 255 - (int)(255 * Math.Clamp((grid[row, col] - 15.0) / 40, 0, 1));
            svg.Append($"<rect x=\"{col * 20}\" y=\"{row * 20}\" width=\"20\" height=\"20\" fill=\"rgb({shade},{shade},{shade})\"/>");
        }
        File.WriteAllText(path, svg.Append("</svg>").ToString());
    }
    public static void Run(ISheepDetector detector, bool recalibrate = false, string? visuallyReadWord = null)
    {
        // The existing Python run selected 0.425. Optional recalibration repeats its selection logic.
        double threshold = recalibrate ? SelectThreshold(detector) : 0.425;
        Plot[] plots = ReadCsv("task_mongolia/survey_plots.csv").Select(row => new Plot(row["image"], Number(row["x_m"]), Number(row["y_m"]))).ToArray();
        int[] counts = plots.Select(plot => detector.PredictConfidences(Path.Combine("task_mongolia/survey", plot.Image)).Count(confidence => confidence >= threshold)).ToArray();
        Directory.CreateDirectory("survey_results/csharp");
        File.WriteAllLines("survey_results/csharp/counts.csv", new[] { "image,count" }.Concat(plots.Select((plot, i) => $"{plot.Image},{counts[i]}")));
        var aligned = Align(plots);
        var columns = GroupAxis(aligned.Select(point => point.X).ToArray());
        var rows = GroupAxis(aligned.Select(point => point.Y).ToArray());
        int[,] grid = new int[rows.Centers.Length, columns.Centers.Length];
        bool[,] occupied = new bool[rows.Centers.Length, columns.Centers.Length];
        for(int i = 0; i < plots.Length; i ++)
        {
            int row = rows.Centers.Length - 1 - rows.Labels[i], col = columns.Labels[i];
            if(occupied[row, col]) throw new InvalidDataException("Two survey plots share one grid cell.");
            occupied[row, col] = true;
            grid[row, col] = counts[i];
        }
        if(occupied.Cast<bool>().Any(value => !value)) throw new InvalidDataException("The survey grid has missing cells.");
        SaveGrid("survey_results/csharp/decoded_grid.svg", grid, null);
        SaveGrid("survey_results/csharp/threshold_35.svg", grid, 35);
        SaveGrid("survey_results/csharp/threshold_40.svg", grid, 40);
        File.WriteAllText("survey_results/csharp/metadata.json", JsonSerializer.Serialize(new { images = plots.Length, confidence = threshold, grid_rows = grid.GetLength(0), grid_columns = grid.GetLength(1), word_read_manually = visuallyReadWord }));
        // Existing interpretation: RECALL. Keep the human reading separate from automatic counts.
        if(!string.IsNullOrWhiteSpace(visuallyReadWord)) File.WriteAllText("survey_results/csharp/flag.txt", "SIGMOID_" + visuallyReadWord.ToUpperInvariant() + "\n");
    }
}
