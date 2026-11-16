// C# logic conversion of main.py: fit an unrestricted CART tree, then read split initials in preorder.
// This educational CART implementation does not reproduce scikit-learn's random generator or tie rules.
// Matching hidden flags exactly requires matching scikit-learn 1.7.2's fitted structure.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.VisualBasic.FileIO;
namespace WeakLayer;
public static class Solution
{
    private sealed record Row(double[] Features, string Label);
    private sealed record Node(int Feature, double Threshold, Node? Left, Node? Right);
    private static double Impurity(Dictionary<string, int> counts, int total) => total == 0 ? 0 : total - counts.Values.Sum(count => (double)count * count) / total;
    private static Node Fit(Row[] rows, Random random)
    {
        if(rows.Length < 2 || rows.All(row => row.Label == rows[0].Label)) return new Node(-1, 0, null, null);
        int[] features = Enumerable.Range(0, rows[0].Features.Length).ToArray();
        for(int i = features.Length - 1; i > 0; i --) { int j = random.Next(i + 1); (features[i], features[j]) = (features[j], features[i]); }
        int bestFeature = -1;
        double bestThreshold = 0, bestImpurity = double.PositiveInfinity;
        foreach(int feature in features)
        {
            Row[] ordered = rows.OrderBy(row => row.Features[feature]).ToArray();
            Dictionary<string, int> left = rows.Select(row => row.Label).Distinct().ToDictionary(label => label, _ => 0);
            Dictionary<string, int> right = rows.GroupBy(row => row.Label).ToDictionary(group => group.Key, group => group.Count());
            for(int cut = 0; cut < ordered.Length - 1; cut ++)
            {
                left[ordered[cut].Label] += 1;
                right[ordered[cut].Label] -= 1;
                double current = ordered[cut].Features[feature], next = ordered[cut + 1].Features[feature];
                if(current == next) continue;
                double impurity = Impurity(left, cut + 1) + Impurity(right, ordered.Length - cut - 1);
                if(impurity < bestImpurity) { bestImpurity = impurity; bestFeature = feature; bestThreshold = current / 2 + next / 2; }
            }
        }
        if(bestFeature < 0) return new Node(-1, 0, null, null);
        Row[] lower = rows.Where(row => row.Features[bestFeature] <= bestThreshold).ToArray(), upper = rows.Where(row => row.Features[bestFeature] > bestThreshold).ToArray();
        return new Node(bestFeature, bestThreshold, Fit(lower, random), Fit(upper, random));
    }
    public static void Main()
    {
        using TextFieldParser csv = new("test-input.csv");
        csv.SetDelimiters(",");
        string[] header = csv.ReadFields()!;
        int labelColumn = Array.IndexOf(header, "slab_released");
        if(labelColumn < 0) throw new InvalidDataException("Missing slab_released target.");
        int[] featureColumns = Enumerable.Range(0, header.Length).Where(column => column != labelColumn).ToArray();
        string[] names = featureColumns.Select(column => header[column]).ToArray();
        List<Row> rows = new();
        while(!csv.EndOfData)
        {
            string[] cells = csv.ReadFields()!;
            rows.Add(new Row(featureColumns.Select(column => double.Parse(cells[column], CultureInfo.InvariantCulture)).ToArray(), cells[labelColumn]));
        }
        Node root = Fit(rows.ToArray(), new Random(42));
        Stack<Node> pending = new();
        pending.Push(root);
        StringBuilder word = new();
        // Push right first so the stack visits root, left subtree, right subtree.
        while(pending.Count > 0)
        {
            Node node = pending.Pop();
            if(node.Feature < 0) continue;
            word.Append(char.ToUpperInvariant(names[node.Feature][0]));
            pending.Push(node.Right!);
            pending.Push(node.Left!);
        }
        File.WriteAllText("test-output.txt", $"SIGMOID_{word}\n");
    }
}
