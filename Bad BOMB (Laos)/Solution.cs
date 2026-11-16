// C# logic conversion of scripts/step_01..03: frozen DINOv2 features + cross-fitted PCA/Isolation Forest.
// Neural inference, image operations, and estimator implementations are supplied by IRuntime.
// .NET's seeded shuffle differs from NumPy/sklearn; exact rankings are not promised.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
namespace BadBomb;
public static class Solution
{
    private const int Seed = 20261115, GridWidth = 14, GridHeight = 36, PatchCount = GridWidth * GridHeight;
    private const double MinimumOccupancy = 0.65;
    public sealed record PatchImage(string Id, double[][] Features, double[] Occupancy);
    public interface IPca { double[] Transform(double[] values); }
    public interface IForest { double ScoreSamples(double[] descriptor); }
    public interface IRuntime
    {
        // Match step_02: green seeds -> close 5x31 -> largest contour -> seeded GrabCut.
        // Trim frame spurs, straighten the mask's principal axis, center, keep a 7x7 rim, crop with 18px margin.
        // Fit/pad to 196x504, ImageNet-normalize RGB, run frozen dinov2_vits14, return 504 patch tokens and occupancies.
        PatchImage PrepareAndExtract(string path, int width, int height, int patchSize);
        IPca FitWhitenedPca(double[][] residuals, int components, int seed);
        IForest FitIsolationForest(double[][] descriptors, int trees, int maxSamples, int seed);
        void SaveFold(string path, FoldModel model);
        void SaveHeatmap(string path, string imageId, double[] patchScores, int gridWidth, int gridHeight);
    }
    public sealed record Calibration(double Center, double Scale);
    public sealed record FoldModel(double[][] Centers, double[][] Scales, bool[] Supported, IPca Projection, IForest Forest, Calibration ForestCalibration, Calibration EnergyCalibration);
    private sealed record Patch(int Image, int Position, double[] Residual, double Occupancy);
    private static double Median(IEnumerable<double> source)
    {
        double[] values = source.OrderBy(value => value).ToArray();
        if(values.Length == 0) throw new InvalidDataException("Cannot take the median of an empty sample.");
        int middle = values.Length / 2;
        return values.Length % 2 == 0 ? (values[middle - 1] + values[middle]) / 2 : values[middle];
    }
    private static Calibration Calibrate(double[] values, double minimumScale)
    {
        double center = Median(values);
        return new Calibration(center, Math.Max(Median(values.Select(value => Math.Abs(value - center))) * 1.4826, minimumScale));
    }
    private static void Shuffle<T>(T[] items, Random random)
    {
        for(int i = items.Length - 1; i > 0; i --) { int j = random.Next(i + 1); (items[i], items[j]) = (items[j], items[i]); }
    }
    private static (double[][] Centers, double[][] Scales, bool[] Supported) SpatialStatistics(PatchImage[] images, int[] fitting)
    {
        int dimensions = images[0].Features[0].Length;
        double[][] centers = new double[PatchCount][], scales = new double[PatchCount][];
        bool[] supported = new bool[PatchCount];
        for(int patch = 0; patch < PatchCount; patch ++)
        {
            centers[patch] = new double[dimensions];
            scales[patch] = Enumerable.Repeat(1.0, dimensions).ToArray();
            int[] selection = fitting.Where(index => images[index].Occupancy[patch] >= MinimumOccupancy).ToArray();
            if(selection.Length < 30) continue;
            supported[patch] = true;
            for(int dimension = 0; dimension < dimensions; dimension ++)
            {
                double center = Median(selection.Select(index => images[index].Features[patch][dimension]));
                centers[patch][dimension] = center;
                scales[patch][dimension] = Math.Max(Median(selection.Select(index => Math.Abs(images[index].Features[patch][dimension] - center))) * 1.4826, 0.005);
            }
        }
        return (centers, scales, supported);
    }
    private static Patch[] ResidualPatches(PatchImage[] images, int[] indices, double[][] centers, double[][] scales, bool[] supported)
    {
        List<Patch> result = new();
        foreach(int image in indices)
        for(int patch = 0; patch < PatchCount; patch ++)
        {
            if(!supported[patch] || images[image].Occupancy[patch] < MinimumOccupancy) continue;
            double[] residual = images[image].Features[patch].Select((value, d) => Math.Clamp((value - centers[patch][d]) / scales[patch][d], -12, 12)).ToArray();
            result.Add(new Patch(image, patch, residual, images[image].Occupancy[patch]));
        }
        return result.ToArray();
    }
    private static double Energy(Patch patch) => Math.Sqrt(patch.Residual.Average(value => value * value));
    private static double[] Descriptor(Patch patch, IPca pca) => pca.Transform(patch.Residual).Concat(new[] { Energy(patch), (double)(patch.Position % GridWidth) / (GridWidth - 1), (double)(patch.Position / GridWidth) / (GridHeight - 1), patch.Occupancy }).ToArray();
    private static double PositiveDeviation(double value, Calibration calibration) => Math.Max(0, (value - calibration.Center) / calibration.Scale);
    public static void Run(IRuntime runtime)
    {
        Directory.CreateDirectory("results/csharp_detector");
        // Deduplicate by file contents before any fold split.
        HashSet<string> hashes = new();
        string[] files = Directory.GetFiles("images", "*.png").OrderBy(path => path, StringComparer.Ordinal).Where(path => hashes.Add(Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))))).ToArray();
        PatchImage[] images = files.Select(path => runtime.PrepareAndExtract(path, 196, 504, 14)).ToArray();
        foreach(PatchImage image in images)
        {
            if(image.Features.Length != PatchCount || image.Occupancy.Length != PatchCount) throw new InvalidDataException("Unexpected patch grid.");
            // DINOv2 tokens are L2-normalized before spatial statistics are learned.
            foreach(double[] feature in image.Features)
            {
                double norm = Math.Max(Math.Sqrt(feature.Sum(value => value * value)), 1e-12);
                for(int i = 0; i < feature.Length; i ++) feature[i] /= norm;
            }
        }
        int[] shuffled = Enumerable.Range(0, images.Length).ToArray();
        Shuffle(shuffled, new Random(Seed));
        double[][] scores = images.Select(_ => Enumerable.Repeat(double.NaN, PatchCount).ToArray()).ToArray();
        for(int fold = 0; fold < 3; fold ++)
        {
            int start = fold * images.Length / 3, end = (fold + 1) * images.Length / 3;
            int[] heldOut = shuffled[start..end], training = shuffled.Except(heldOut).OrderBy(index => index).ToArray();
            Random random = new(Seed + fold);
            Shuffle(training, random);
            int[] calibration = training.Take(Math.Max(12, training.Length / 10)).ToArray(), fitting = training.Except(calibration).OrderBy(index => index).ToArray();
            var statistics = SpatialStatistics(images, fitting);
            Patch[] sampled = ResidualPatches(images, fitting, statistics.Centers, statistics.Scales, statistics.Supported);
            Shuffle(sampled, random);
            sampled = sampled.Take(60000).ToArray();
            IPca pca = runtime.FitWhitenedPca(sampled.Select(patch => patch.Residual).ToArray(), 32, Seed + fold);
            IForest forest = runtime.FitIsolationForest(sampled.Select(patch => Descriptor(patch, pca)).ToArray(), 256, 512, Seed + fold);
            // Calibration uses separate images, never images used to fit these estimators.
            Patch[] reference = ResidualPatches(images, calibration, statistics.Centers, statistics.Scales, statistics.Supported);
            Shuffle(reference, random);
            reference = reference.Take(20000).ToArray();
            Calibration forestCalibration = Calibrate(reference.Select(patch => -forest.ScoreSamples(Descriptor(patch, pca))).ToArray(), 0.01);
            Calibration energyCalibration = Calibrate(reference.Select(patch => Math.Log(1 + Energy(patch))).ToArray(), 0.02);
            FoldModel model = new(statistics.Centers, statistics.Scales, statistics.Supported, pca, forest, forestCalibration, energyCalibration);
            runtime.SaveFold($"results/csharp_detector/fold_{fold + 1}", model);
            foreach(Patch patch in ResidualPatches(images, heldOut, model.Centers, model.Scales, model.Supported))
            {
                double forestZ = PositiveDeviation(-forest.ScoreSamples(Descriptor(patch, pca)), forestCalibration), energyZ = PositiveDeviation(Math.Log(1 + Energy(patch)), energyCalibration);
                scores[patch.Image][patch.Position] = (forestZ + energyZ) * 0.5;
            }
        }
        var ranking = Enumerable.Range(0, images.Length).Select(index =>
        {
            double[] valid = scores[index].Where(double.IsFinite).OrderByDescending(value => value).ToArray();
            if(valid.Length < 30) throw new InvalidDataException("Insufficient foreground scores.");
            double score = valid.Take(Math.Max(1, (int)Math.Ceiling(valid.Length * 0.01))).Average();
            runtime.SaveHeatmap($"results/csharp_detector/{images[index].Id}.png", images[index].Id, scores[index], GridWidth, GridHeight);
            return new { image_id = images[index].Id, anomaly_score = score };
        }).OrderByDescending(row => row.anomaly_score).ThenBy(row => row.image_id, StringComparer.Ordinal).ToArray();
        File.WriteAllText("results/csharp_detector/ranking.json", JsonSerializer.Serialize(ranking, new JsonSerializerOptions { WriteIndented = true }));
        File.WriteAllLines("results/csharp_detector/provisional_top28.txt", ranking.Take(28).Select(row => row.image_id));
        // Scores propose candidates; they are not probabilities or a new evaluator result.
    }
}
