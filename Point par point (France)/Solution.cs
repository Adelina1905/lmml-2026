// C# conversion of the earlier booking_clusters.py and scan_numeric.py explorations.
// This task has no retained verified solver: clusters are candidates, not an automatically recovered flag.
// The simple DBSCAN below favors readable logic over speed on the 28000-row dataset.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.VisualBasic.FileIO;
namespace PointParPoint;
public static class Solution
{
    public interface IHierarchicalClustering
    {
        // Supply an HDBSCAN implementation matching min_cluster_size=12 and min_samples=5.
        int[] Hdbscan(double[][] points, int minimumClusterSize, int minimumSamples);
    }
    private sealed record Listing(string Id, double Longitude, double Latitude, Dictionary<string, double> Values);
    private static double Parse(string text) => double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double value) ? value : double.NaN;
    private static Listing[] ReadListings()
    {
        using TextFieldParser csv = new("paris_listings.csv");
        csv.SetDelimiters(",");
        string[] header = csv.ReadFields()!;
        List<Listing> listings = new();
        while(!csv.EndOfData)
        {
            string[] row = csv.ReadFields()!;
            Dictionary<string, double> values = header.Select((name, i) => (Name: name, Value: Parse(row[i]))).ToDictionary(pair => pair.Name, pair => pair.Value);
            listings.Add(new Listing(row[Array.IndexOf(header, "listing_id")], values["longitude"], values["latitude"], values));
        }
        return listings.ToArray();
    }
    private static double[][] Standardize(Listing[] listings, string[] features, bool logarithms)
    {
        double[][] points = listings.Select(_ => new double[features.Length]).ToArray();
        for(int column = 0; column < features.Length; column ++)
        {
            double[] known = listings.Select(row => row.Values[features[column]]).Where(double.IsFinite).OrderBy(value => value).ToArray();
            if(known.Length == 0) throw new InvalidDataException($"No numeric values for {features[column]}.");
            double median = (known[(known.Length - 1) / 2] + known[known.Length / 2]) / 2;
            double[] values = listings.Select(row => double.IsFinite(row.Values[features[column]]) ? row.Values[features[column]] : median).ToArray();
            if(logarithms) values = values.Select(value => Math.Log(1 + value)).ToArray();
            double mean = values.Average(), scale = Math.Sqrt(values.Average(value => Math.Pow(value - mean, 2)));
            for(int row = 0; row < points.Length; row ++) points[row][column] = (values[row] - mean) / (scale > 0 ? scale : 1);
        }
        return points;
    }
    public static int[] Dbscan(double[][] points, double epsilon, int minimumSamples)
    {
        const int Unvisited = -2, Noise = -1;
        int[] labels = Enumerable.Repeat(Unvisited, points.Length).ToArray();
        int cluster = 0;
        int[] Neighbors(int index) => Enumerable.Range(0, points.Length).Where(other => points[index].Zip(points[other], (a, b) => (a - b) * (a - b)).Sum() <= epsilon * epsilon).ToArray();
        for(int seed = 0; seed < points.Length; seed ++)
        {
            if(labels[seed] != Unvisited) continue;
            int[] nearby = Neighbors(seed);
            if(nearby.Length < minimumSamples) { labels[seed] = Noise; continue; }
            labels[seed] = cluster;
            Queue<int> frontier = new(nearby);
            HashSet<int> queued = nearby.ToHashSet();
            while(frontier.Count > 0)
            {
                int current = frontier.Dequeue();
                if(labels[current] == Noise) labels[current] = cluster;
                if(labels[current] != Unvisited) continue;
                labels[current] = cluster;
                int[] neighbors = Neighbors(current);
                if(neighbors.Length < minimumSamples) continue;
                foreach(int neighbor in neighbors) if(queued.Add(neighbor)) frontier.Enqueue(neighbor);
            }
            cluster ++;
        }
        return labels;
    }
    private static double GeographicSpread(Listing[] listings, int[] indices)
    {
        double meanX = indices.Average(i => listings[i].Longitude), meanY = indices.Average(i => listings[i].Latitude), divisor = Math.Max(1, indices.Length - 1);
        double xx = indices.Sum(i => Math.Pow(listings[i].Longitude - meanX, 2)) / divisor, yy = indices.Sum(i => Math.Pow(listings[i].Latitude - meanY, 2)) / divisor, xy = indices.Sum(i => (listings[i].Longitude - meanX) * (listings[i].Latitude - meanY)) / divisor;
        return xx * yy - xy * xy;
    }
    private static void Plot(string path, Listing[] listings, int[] indices)
    {
        double minX = indices.Min(i => listings[i].Longitude), maxX = indices.Max(i => listings[i].Longitude), minY = indices.Min(i => listings[i].Latitude), maxY = indices.Max(i => listings[i].Latitude);
        double scale = 700 / Math.Max(Math.Max(maxX - minX, maxY - minY), 1e-12);
        StringBuilder svg = new("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 800 800\"><rect width=\"800\" height=\"800\" fill=\"white\"/>");
        foreach(int index in indices)
        {
            double x = 50 + (listings[index].Longitude - minX) * scale, y = 750 - (listings[index].Latitude - minY) * scale;
            svg.Append(FormattableString.Invariant($"<circle cx=\"{x:F2}\" cy=\"{y:F2}\" r=\"1.8\" fill=\"#173d58\"/>"));
        }
        File.WriteAllText(path, svg.Append("</svg>").ToString());
    }
    public static void Run(IHierarchicalClustering? hierarchy = null)
    {
        Listing[] listings = ReadListings();
        Directory.CreateDirectory("work/csharp");
        string[] booking = { "price", "accommodates", "minimum_nights", "review_scores_rating", "availability_365" };
        double[][] bookingPoints = Standardize(listings, booking, false);
        int[] first = Dbscan(bookingPoints, 0.5, 25);
        File.WriteAllLines("work/csharp/booking_labels.csv", new[] { "listing_id,label" }.Concat(listings.Select((listing, i) => $"{listing.Id},{first[i]}")));
        // The earlier exploration applied HDBSCAN to both all rows and the first pass's noise.
        if(hierarchy != null)
        {
            foreach(bool noiseOnly in new[] { false, true })
            {
                int[] indices = Enumerable.Range(0, listings.Length).Where(i => !noiseOnly || first[i] == -1).ToArray();
                if(indices.Length == 0) continue;
                int[] labels = hierarchy.Hdbscan(indices.Select(i => bookingPoints[i]).ToArray(), 12, 5);
                foreach(int label in labels.Distinct()) Plot($"work/csharp/booking_{noiseOnly}_{label}.svg", listings, indices.Where((_, i) => labels[i] == label).ToArray());
            }
        }
        string[] numeric = listings[0].Values.Keys.Where(name => name != "listing_id" && name != "longitude" && name != "latitude" && listings.Any(row => double.IsFinite(row.Values[name]))).ToArray();
        string[][] featureSets = { numeric, new[] { "price", "accommodates", "review_scores_rating" }, new[] { "price", "minimum_nights", "availability_365" }, new[] { "price", "number_of_reviews", "availability_365", "review_scores_rating" }, new[] { "price", "accommodates", "bedrooms", "beds", "review_scores_rating", "availability_365" } };
        List<(double Spread, string Name, int[] Indices)> candidates = new();
        for(int set = 0; set < featureSets.Length; set ++)
        foreach(bool logged in new[] { false, true })
        {
            double[][] points = Standardize(listings, featureSets[set], logged);
            foreach(double epsilon in featureSets[set].Length > 3 ? new[] { 0.4, 0.7, 1.0, 1.4 } : new[] { 0.05, 0.1, 0.2 })
            {
                int[] labels = Dbscan(points, epsilon, 8);
                foreach(int label in labels.Where(label => label >= 0).Distinct())
                {
                    int[] indices = Enumerable.Range(0, labels.Length).Where(i => labels[i] == label).ToArray();
                    if(indices.Length < 15 || indices.Length > 1200) continue;
                    candidates.Add((GeographicSpread(listings, indices), FormattableString.Invariant($"set_{set}_log_{logged}_eps_{epsilon}_cluster_{label}"), indices));
                }
            }
        }
        foreach(var candidate in candidates.OrderBy(candidate => candidate.Spread).Take(48)) Plot($"work/csharp/{candidate.Name}.svg", listings, candidate.Indices);
        Console.WriteLine("Inspect candidate maps for letters, establish their order, then export the final labels and word. This exploration does not claim a solved flag.");
    }
    public static void Main() => Run();
}
