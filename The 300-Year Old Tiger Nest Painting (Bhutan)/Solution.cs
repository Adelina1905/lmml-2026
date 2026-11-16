// C# conversion of main.py. Supply PNG decoding/encoding through IImageFiles below.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
namespace TigerNestPainting;
public static class Solution
{
    public interface IImageFiles
    {
        // Pixels are [row, column, channel]; preserve grayscale, color, and alpha channels.
        byte[,,] ReadPng(string path);
        void WritePng(string path, byte[,,] pixels);
    }
    public static double[,] EdgeCosts(byte[][,,] strips)
    {
        int count = strips.Length, height = strips[0].GetLength(0), width = strips[0].GetLength(1), channels = strips[0].GetLength(2);
        double[,] costs = new double[count, count];
        for(int first = 0; first < count; first ++)
        for(int second = 0; second < count; second ++)
        {
            if(first == second) { costs[first, second] = 1e100; continue; }
            for(int row = 0; row < height; row ++)
            for(int channel = 0; channel < channels; channel ++)
            {
                double right = strips[first][row, width - 1, channel], left = strips[second][row, 0, channel];
                if(width == 1) { costs[first, second] += Math.Pow(right - left, 2); continue; }
                double rightGradient = right - strips[first][row, width - 2, channel], leftGradient = strips[second][row, 1, channel] - left;
                costs[first, second] += Math.Pow(right + 0.5 * rightGradient - left, 2) + Math.Pow(right - left + 0.5 * leftGradient, 2);
            }
        }
        return costs;
    }
    // Hungarian assignment: each strip gets one successor, initially forming separate cycles.
    private static int[] Assign(double[,] costs)
    {
        int n = costs.GetLength(0);
        double[] rowPotential = new double[n + 1], colPotential = new double[n + 1];
        int[] matchedRow = new int[n + 1], previousColumn = new int[n + 1];
        for(int row = 1; row <= n; row ++)
        {
            matchedRow[0] = row;
            double[] minimum = Enumerable.Repeat(double.PositiveInfinity, n + 1).ToArray();
            bool[] used = new bool[n + 1];
            int column = 0;
            do
            {
                used[column] = true;
                int currentRow = matchedRow[column], nextColumn = 0;
                double delta = double.PositiveInfinity;
                for(int candidate = 1; candidate <= n; candidate ++)
                {
                    if(used[candidate]) continue;
                    double reduced = costs[currentRow - 1, candidate - 1] - rowPotential[currentRow] - colPotential[candidate];
                    if(reduced < minimum[candidate]) { minimum[candidate] = reduced; previousColumn[candidate] = column; }
                    if(minimum[candidate] < delta) { delta = minimum[candidate]; nextColumn = candidate; }
                }
                for(int candidate = 0; candidate <= n; candidate ++)
                {
                    if(used[candidate]) { rowPotential[matchedRow[candidate]] += delta; colPotential[candidate] -= delta; }
                    else minimum[candidate] -= delta;
                }
                column = nextColumn;
            } while(matchedRow[column] != 0);
            do { int previous = previousColumn[column]; matchedRow[column] = matchedRow[previous]; column = previous; } while(column != 0);
        }
        int[] successor = new int[n];
        for(int column = 1; column <= n; column ++) successor[matchedRow[column] - 1] = column - 1;
        return successor;
    }
    public static int[] RecoverOrder(double[,] costs)
    {
        int count = costs.GetLength(0);
        if(count == 1) return new[] { 0 };
        double[,] augmented = new double[count + 1, count + 1];
        for(int i = 0; i < count; i ++)
        for(int j = 0; j < count; j ++) augmented[i, j] = costs[i, j];
        // The extra dummy strip gives the final path free left and right endpoints.
        int[] successor = Assign(augmented);
        HashSet<int> remaining = Enumerable.Range(0, count + 1).ToHashSet();
        List<List<int>> cycles = new();
        while(remaining.Count > 0)
        {
            List<int> cycle = new();
            int current = remaining.Min();
            while(remaining.Remove(current)) { cycle.Add(current); current = successor[current]; }
            cycles.Add(cycle);
        }
        while(cycles.Count > 1)
        {
            double best = double.PositiveInfinity;
            int aBest = -1, bBest = -1, xBest = -1, yBest = -1;
            for(int a = 0; a < cycles.Count; a ++)
            for(int b = 0; b < a; b ++)
            foreach(int x in cycles[a])
            foreach(int y in cycles[b])
            {
                double delta = augmented[x, successor[y]] + augmented[y, successor[x]] - augmented[x, successor[x]] - augmented[y, successor[y]];
                if(delta < best) { best = delta; aBest = a; bBest = b; xBest = x; yBest = y; }
            }
            (successor[xBest], successor[yBest]) = (successor[yBest], successor[xBest]);
            cycles[bBest].AddRange(cycles[aBest]);
            cycles.RemoveAt(aBest);
        }
        List<int> order = new();
        for(int current = successor[count]; current != count; current = successor[current])
        {
            if(order.Contains(current)) throw new InvalidDataException("Repeated strip in reconstruction.");
            order.Add(current);
        }
        if(order.Count != count) throw new InvalidDataException("Missing strip in reconstruction.");
        return order.ToArray();
    }
    public static void Run(IImageFiles images)
    {
        string[] files = Directory.GetFiles("input").Where(path => Path.GetExtension(path).Equals(".png", StringComparison.OrdinalIgnoreCase)).OrderBy(path => path, StringComparer.Ordinal).ToArray();
        if(files.Length == 0) throw new InvalidDataException("No PNG strips in input/.");
        var strips = files.Select(images.ReadPng).ToArray();
        int height = strips[0].GetLength(0), width = strips[0].GetLength(1), channels = strips[0].GetLength(2);
        if(strips.Any(strip => strip.GetLength(0) != height || strip.GetLength(1) != width || strip.GetLength(2) != channels)) throw new InvalidDataException("Strip dimensions differ.");
        int[] order = RecoverOrder(EdgeCosts(strips));
        byte[,,] reconstructed = new byte[height, width * order.Length, channels];
        for(int index = 0; index < order.Length; index ++)
        for(int row = 0; row < height; row ++)
        for(int col = 0; col < width; col ++)
        for(int channel = 0; channel < channels; channel ++) reconstructed[row, index * width + col, channel] = strips[order[index]][row, col, channel];
        images.WritePng("reconstructed_image.png", reconstructed);
        File.WriteAllText("reconstructed_order.json", JsonSerializer.Serialize(order.Select(index => Path.GetFileName(files[index])), new JsonSerializerOptions { WriteIndented = true }));
    }
}
