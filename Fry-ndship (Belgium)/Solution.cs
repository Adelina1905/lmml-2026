// C# conversion of solve.py. SVG scatter plots replace matplotlib PNG figures.
// The word is read visually from the residual plot, as in the Python solution.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using Microsoft.VisualBasic.FileIO;
namespace FryNdship;
public static class Solution
{
    private static readonly CultureInfo Culture = CultureInfo.InvariantCulture;
    private sealed record Field(string Id, double Length, double Width, double Ditch, double Hedge, double Edge);
    private static Field[] ReadFields(string path)
    {
        using TextFieldParser csv = new(path);
        csv.SetDelimiters(",");
        string[] header = csv.ReadFields()!;
        List<Field> fields = new();
        while(!csv.EndOfData)
        {
            string[] row = csv.ReadFields()!;
            string Text(string column) => row[Array.IndexOf(header, column)];
            double Read(string column) => double.Parse(Text(column), Culture);
            double length = Read("long_side_m"), area = Read("area_ha"), edge = Read("edge_m");
            if(length <= 0 || area <= 0 || edge <= 0) throw new InvalidDataException("Field dimensions must be positive.");
            fields.Add(new Field(Text("field_id"), length, 10000 * area / length, Read("ditch_m"), Read("hedge_m"), edge));
        }
        if(fields.Select(field => field.Id).Distinct().Count() != fields.Count) throw new InvalidDataException("Duplicate field identifiers.");
        return fields.ToArray();
    }
    // QR least squares keeps the regression calculation visible without a numerical package.
    private static double[] Fit(double[][] x, double[] y)
    {
        int rows = x.Length, columns = x[0].Length;
        double[][] q = new double[columns][];
        double[,] r = new double[columns, columns];
        for(int column = 0; column < columns; column ++)
        {
            double[] vector = x.Select(row => row[column]).ToArray();
            for(int previous = 0; previous < column; previous ++)
            {
                r[previous, column] = Enumerable.Range(0, rows).Sum(i => q[previous][i] * vector[i]);
                for(int i = 0; i < rows; i ++) vector[i] -= r[previous, column] * q[previous][i];
            }
            r[column, column] = Math.Sqrt(vector.Sum(value => value * value));
            if(r[column, column] < 1e-12) throw new InvalidDataException("Regression features are linearly dependent.");
            q[column] = vector.Select(value => value / r[column, column]).ToArray();
        }
        double[] coefficients = q.Select(column => column.Zip(y, (left, right) => left * right).Sum()).ToArray();
        for(int row = columns - 1; row >= 0; row --)
        {
            for(int column = row + 1; column < columns; column ++) coefficients[row] -= r[row, column] * coefficients[column];
            coefficients[row] /= r[row, row];
        }
        return coefficients;
    }
    private static double[] Predict(double[][] x, double[] coefficients) => x.Select(row => row.Zip(coefficients, (value, weight) => value * weight).Sum()).ToArray();
    private static object Metrics(double[] actual, double[] predicted)
    {
        double mean = actual.Average(), squaredError = actual.Zip(predicted, (a, p) => Math.Pow(a - p, 2)).Sum();
        return new { r2 = 1 - squaredError / actual.Sum(value => Math.Pow(value - mean, 2)), rmse_m = Math.Sqrt(squaredError / actual.Length), mae_m = actual.Zip(predicted, (a, p) => Math.Abs(a - p)).Average() };
    }
    private static void Scatter(string path, string title, double[] x, double[] y)
    {
        double left = x.Min(), right = x.Max(), bottom = y.Min(), top = y.Max();
        StringBuilder svg = new("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1400 460\"><rect width=\"1400\" height=\"460\" fill=\"white\"/>");
        svg.Append($"<text x=\"50\" y=\"30\" font-size=\"22\">{System.Security.SecurityElement.Escape(title)}</text><path d=\"M50 55V410H1370\" fill=\"none\" stroke=\"black\"/>");
        for(int i = 0; i < x.Length; i ++)
        {
            double px = 60 + 1300 * (x[i] - left) / Math.Max(right - left, 1e-12), py = 400 - 330 * (y[i] - bottom) / Math.Max(top - bottom, 1e-12);
            svg.Append(FormattableString.Invariant($"<circle cx=\"{px:F2}\" cy=\"{py:F2}\" r=\"1.4\" fill=\"#176b91\"/>"));
        }
        svg.Append("<text x=\"500\" y=\"445\">Predicted edge length (m); vertical axis: measured minus predicted (m)</text></svg>");
        File.WriteAllText(path, svg.ToString());
    }
    public static void Main()
    {
        Field[] fields = ReadFields("potato_fields.csv");
        double[] y = fields.Select(field => field.Edge).ToArray();
        double[][] x = fields.Select(field => new[] { 1.0, field.Length, field.Width, field.Ditch }).ToArray();
        double[] coefficients = Fit(x, y), prediction = Predict(x, coefficients);
        Scatter("supervision_residuals.csharp.svg", "Fry-ndship: residuals from declared dimensions", prediction, y.Zip(prediction, (actual, fitted) => actual - fitted).ToArray());
        // A separate holdout measures prediction quality; its .NET shuffle differs from NumPy's.
        int[] shuffled = Enumerable.Range(0, fields.Length).ToArray();
        Random random = new(2026);
        for(int i = shuffled.Length - 1; i > 0; i --) { int j = random.Next(i + 1); (shuffled[i], shuffled[j]) = (shuffled[j], shuffled[i]); }
        int[] train = shuffled.Take((int)(fields.Length * 0.8)).ToArray(), test = shuffled.Skip((int)(fields.Length * 0.8)).ToArray();
        double[] validationWeights = Fit(train.Select(i => x[i]).ToArray(), train.Select(i => y[i]).ToArray());
        File.WriteAllText("metrics.csharp.json", JsonSerializer.Serialize(new { coefficients, in_sample = Metrics(y, prediction), holdout = Metrics(test.Select(i => y[i]).ToArray(), Predict(test.Select(i => x[i]).ToArray(), validationWeights)), word_read_visually = "SUPERVISION" }, new JsonSerializerOptions { WriteIndented = true }));
        // Omitting ditch or adding the on-site hedge reveals the other comparison words.
        foreach(bool includeHedge in new[] { false, true })
        {
            double[][] comparison = fields.Select(field => includeHedge ? new[] { 1.0, field.Length, field.Width, field.Ditch, field.Hedge } : new[] { 1.0, field.Length, field.Width }).ToArray();
            double[] fitted = Predict(comparison, Fit(comparison, y));
            Scatter(includeHedge ? "prediction_residuals.csharp.svg" : "regression_residuals.csharp.svg", includeHedge ? "With on-site hedge" : "Without declared ditch", fitted, y.Zip(fitted, (actual, estimate) => actual - estimate).ToArray());
        }
        Console.WriteLine("Read the plotted letters: SIGMOID_SUPERVISION");
    }
}
