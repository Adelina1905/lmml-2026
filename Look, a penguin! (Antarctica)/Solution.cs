// C# conversion of solve.py. Detect corruption first, then recognize letters without changing that mask.
// SVG charts replace matplotlib PNGs. Run from this task folder.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
namespace LookAPenguin;
public static class Solution
{
    private static readonly Dictionary<char, string> Font = new()
    {
        ['A'] = ".#./#.#/###/#.#/#.#", ['B'] = "##./#.#/##./#.#/##.", ['C'] = "###/#../#../#../###", ['D'] = "##./#.#/#.#/#.#/##.",
        ['E'] = "###/#../###/#../###", ['F'] = "###/#../###/#../#..", ['G'] = "###/#../#.#/#.#/###", ['H'] = "#.#/#.#/###/#.#/#.#",
        ['I'] = "###/.#./.#./.#./###", ['J'] = "..#/..#/..#/#.#/###", ['K'] = "#.#/#.#/##./#.#/#.#", ['L'] = "#../#../#../#../###",
        ['M'] = "#.#/###/###/#.#/#.#", ['N'] = "#.#/###/###/###/#.#", ['n'] = "###/###/#.#/#.#/#.#", ['O'] = "###/#.#/#.#/#.#/###",
        ['P'] = "##./#.#/##./#../#..", ['Q'] = "###/#.#/#.#/###/..#", ['R'] = "##./#.#/##./#.#/#.#", ['S'] = "###/#../###/..#/###",
        ['T'] = "###/.#./.#./.#./.#.", ['U'] = "#.#/#.#/#.#/#.#/###", ['V'] = "#.#/#.#/#.#/#.#/.#.", ['W'] = "#.#/#.#/#.#/###/###",
        ['X'] = "#.#/#.#/.#./#.#/#.#", ['Y'] = "#.#/#.#/.#./.#./.#.", ['Z'] = "###/..#/.#./#../###"
    };
    private sealed class Report
    {
        public string Id = "", Header = "";
        public DateTime Stamp;
        public string[] Groups = Array.Empty<string>(), ExtraGroups = Array.Empty<string>();
        public Dictionary<char, string> Main = new(), Extra = new();
        public Dictionary<string, int> Values = new();
        public List<string> Strong = new(), Contextual = new(), Reasons = new();
        public (int Day, int Hour) Point;
        public bool InLetter;
    }
    private sealed record Glyph(char Letter, int X, int Y, string Bitmap);
    private static bool Digits(string value) => value.Length > 0 && value.All(char.IsDigit);
    private static int Temperature(string group) => int.Parse(group[2..], CultureInfo.InvariantCulture) * (group[1] == '1' ? -1 : 1);
    private static int Pressure(string group) { int value = int.Parse(group[1..], CultureInfo.InvariantCulture); return value + (value < 5000 ? 10000 : 0); }
    private static double Median(IEnumerable<int> source)
    {
        int[] values = source.OrderBy(value => value).ToArray();
        return (values[(values.Length - 1) / 2] + values[values.Length / 2]) / 2.0;
    }
    private static Report[] Parse(string path)
    {
        List<Report> reports = new();
        foreach(string line in File.ReadLines(path).Where(line => !string.IsNullOrWhiteSpace(line)))
        {
            string[] fields = line.TrimEnd('=').Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries), groups = fields.Skip(3).ToArray();
            DateTime stamp = DateTime.ParseExact(fields[1] + " " + fields[2], "yyyy-MM-dd HH'Z'", CultureInfo.InvariantCulture);
            int boundary = Array.IndexOf(groups, "333");
            if(boundary < 0) boundary = groups.Length;
            string? wind = groups[4];
            Report report = new() { Id = fields[0], Stamp = stamp, Groups = groups, Header = groups[3], ExtraGroups = groups.Skip(boundary + 1).ToArray() };
            foreach(string group in groups.Skip(5).Take(Math.Max(0, boundary - 5))) report.Main[group[0]] = group;
            foreach(string group in report.ExtraGroups.Where(group => group[0] == '1' || group[0] == '2')) report.Extra[group[0]] = group;
            // A missing wind group can shift the temperature into the wind position.
            if(!report.Main.ContainsKey('1') && wind.StartsWith("1", StringComparison.Ordinal) && report.Main.ContainsKey('2') && report.Main.TryGetValue('8', out string? clouds) && char.IsDigit(clouds[1]) && clouds[1] > wind[0]) { report.Main['1'] = wind; wind = null; }
            foreach(var item in new[] { (Name: "temperature", Prefix: '1'), (Name: "dewpoint", Prefix: '2'), (Name: "station_pressure", Prefix: '3'), (Name: "sea_pressure", Prefix: '4') })
            {
                if(report.Main.TryGetValue(item.Prefix, out string? group) && Digits(group)) report.Values[item.Name] = item.Prefix <= '2' ? Temperature(group) : Pressure(group);
            }
            if(wind != null)
            {
                foreach(var item in new[] { (Name: "cloud_cover", Value: wind[..1]), (Name: "direction", Value: wind.Substring(1, 2)), (Name: "speed", Value: wind[3..]) }) if(Digits(item.Value)) report.Values[item.Name] = int.Parse(item.Value, CultureInfo.InvariantCulture);
            }
            if(int.Parse(report.Id, CultureInfo.InvariantCulture) != reports.Count + 1 || reports.Count > 0 && stamp - reports[^1].Stamp != TimeSpan.FromHours(3)) throw new InvalidDataException("Nonconsecutive report IDs or observation times.");
            reports.Add(report);
        }
        return reports.ToArray();
    }
    private static (Glyph[] Letters, int Days) Detect(Report[] reports)
    {
        for(int i = 0; i < reports.Length; i ++)
        {
            Report r = reports[i];
            var v = r.Values;
            if(!Digits(r.Groups[1]) || r.Groups[1] != r.Stamp.ToString("ddHH", CultureInfo.InvariantCulture) + "1") r.Strong.Add("invalid SYNOP observation-time group");
            if(r.Groups.Skip(1).Any(group => group != "333" && !Regex.IsMatch(group, @"^[0-9/]{5}$"))) r.Strong.Add("illegal group character or length");
            if(v.GetValueOrDefault("direction") > 36) r.Strong.Add("wind direction exceeds code 36");
            if(v.ContainsKey("temperature") && v.ContainsKey("dewpoint"))
            {
                int depression = v["temperature"] - v["dewpoint"];
                if(depression < 0) r.Strong.Add("dew point exceeds air temperature");
                else if(depression < 4) r.Contextual.Add("isolated near-saturation value within an otherwise drier sequence");
            }
            if(v.ContainsKey("station_pressure") && v.ContainsKey("sea_pressure") && (v["sea_pressure"] - v["station_pressure"] < 50 || v["sea_pressure"] - v["station_pressure"] > 62)) r.Strong.Add("station-to-sea-level correction outside the empirical 5.0-6.2 hPa band");
            if(r.Extra.ContainsKey('1') && r.Extra.ContainsKey('2') && Temperature(r.Extra['1']) < Temperature(r.Extra['2'])) r.Strong.Add("12-hour maximum is below the minimum");
            if(r.Main.TryGetValue('8', out string? cloud) && char.IsDigit(cloud[1]) && v.TryGetValue("cloud_cover", out int cover) && cover < 9 && cloud[1] - '0' > cover) r.Strong.Add("low/middle cloud amount exceeds total cloud cover");
            int[] gusts = r.ExtraGroups.Where(group => (group.StartsWith("910", StringComparison.Ordinal) || group.StartsWith("911", StringComparison.Ordinal)) && Digits(group)).Select(group => int.Parse(group[3..], CultureInfo.InvariantCulture)).ToArray();
            if(v.ContainsKey("speed") && gusts.Any(gust => gust < v["speed"])) r.Strong.Add("reported gust is below sustained wind");
            if(v.GetValueOrDefault("speed") > 60) r.Strong.Add("isolated wind-speed spike above 60 m/s");
            if(i > 0 && i + 1 < reports.Length && reports[i - 1].Values.ContainsKey("temperature") && v.ContainsKey("temperature") && reports[i + 1].Values.ContainsKey("temperature") && Math.Min(Math.Abs(v["temperature"] - reports[i - 1].Values["temperature"]), Math.Abs(v["temperature"] - reports[i + 1].Values["temperature"])) > 100) r.Strong.Add("isolated temperature excursion exceeding 10 C on both sides");
            if(r.Main.TryGetValue('7', out string? weatherGroup) && Digits(weatherGroup))
            {
                int weather = int.Parse(weatherGroup.Substring(1, 2), CultureInfo.InvariantCulture);
                if(weather >= 50 && v.TryGetValue("cloud_cover", out int cloudCover) && cloudCover == 0) r.Contextual.Add("precipitation-weather code with zero cloud cover");
                if(weather >= 70 && weather <= 79 && Digits(r.Header[3..]) && int.Parse(r.Header[3..], CultureInfo.InvariantCulture) > 80) r.Contextual.Add("snow-weather code with unusually high visibility");
            }
            if(v.GetValueOrDefault("direction") != 0)
            {
                int[] neighbors = Enumerable.Range(Math.Max(0, i - 2), Math.Min(reports.Length, i + 3) - Math.Max(0, i - 2)).Where(j => j != i && reports[j].Values.GetValueOrDefault("direction") != 0).Select(j => reports[j].Values["direction"]).ToArray();
                if(neighbors.Length > 0 && Math.Abs(v["direction"] - Median(neighbors)) >= 10) r.Contextual.Add("wind-direction departure of at least 100 degrees from local median");
            }
        }
        Dictionary<int, int> residuals = new();
        for(int i = 1; i < reports.Length; i ++)
        {
            Report r = reports[i];
            if(!r.Main.TryGetValue('5', out string? group) || !r.Values.ContainsKey("station_pressure") || !reports[i - 1].Values.ContainsKey("station_pressure")) continue;
            int kind = group[1] - '0', amount = int.Parse(group[2..], CultureInfo.InvariantCulture) * (kind < 4 ? 1 : kind > 4 ? -1 : 0);
            residuals[i] = r.Values["station_pressure"] - reports[i - 1].Values["station_pressure"] - amount;
        }
        // Opposite errors on adjacent intervals implicate their shared observation only.
        HashSet<int> explained = new();
        foreach(var pair in residuals)
        {
            if(Math.Abs(pair.Value) > 3 && residuals.TryGetValue(pair.Key + 1, out int next) && Math.Abs(pair.Value + next) <= 2) { reports[pair.Key].Strong.Add("isolated pressure change inconsistent with tendencies on both sides"); explained.Add(pair.Key); explained.Add(pair.Key + 1); }
        }
        foreach(var pair in residuals) if(Math.Abs(pair.Value) > 3 && !explained.Contains(pair.Key)) reports[pair.Key].Strong.Add("pressure-tendency sign or amount contradicts observed pressure change");
        DateTime start = reports[0].Stamp.Date;
        foreach(Report r in reports) r.Point = ((r.Stamp.Date - start).Days, r.Stamp.Hour / 3);
        HashSet<(int Day, int Hour)> basePoints = reports.Where(r => r.Strong.Count > 0 || r.Contextual.Any(reason => !reason.Contains("wind-direction", StringComparison.Ordinal))).Select(r => r.Point).ToHashSet();
        for(int i = 0; i < reports.Length; i ++)
        {
            Report r = reports[i];
            r.Reasons = r.Strong.Concat(r.Contextual.Where(reason => !reason.Contains("wind-direction", StringComparison.Ordinal))).ToList();
            if(!r.Contextual.Any(reason => reason.Contains("wind-direction", StringComparison.Ordinal))) continue;
            int support = 0;
            for(int dx = -1; dx <= 1; dx ++)
            for(int dy = -1; dy <= 1; dy ++) if((dx != 0 || dy != 0) && basePoints.Contains((r.Point.Day + dx, r.Point.Hour + dy))) support ++;
            bool isolated = i > 0 && i + 1 < reports.Length && reports[i - 1].Values.TryGetValue("direction", out int before) && reports[i + 1].Values.TryGetValue("direction", out int after) && before <= 36 && after <= 36 && Math.Min(Math.Abs(r.Values["direction"] - before), Math.Abs(r.Values["direction"] - after)) >= 10;
            if(r.Values.GetValueOrDefault("speed") >= 7 && isolated) r.Reasons.Add("isolated direction jump of at least 100 degrees with wind >= 7 m/s");
            else if(support >= 3) r.Reasons.Add("local direction outlier supported by at least three neighboring anomalous observations");
        }
        HashSet<(int Day, int Hour)> points = reports.Where(r => r.Reasons.Count > 0).Select(r => r.Point).ToHashSet(), hard = reports.Where(r => r.Strong.Count > 0).Select(r => r.Point).ToHashSet();
        int days = (reports[^1].Stamp.Date - start).Days + 1;
        List<Glyph> letters = new();
        for(int x = 0; x < days - 2; x ++)
        for(int y = 0; y < 4; y ++)
        {
            string bitmap = string.Join("/", Enumerable.Range(0, 5).Select(dy => new string(Enumerable.Range(0, 3).Select(dx => points.Contains((x + dx, y + dy)) ? '#' : '.').ToArray())));
            bool hasHardEvidence = Enumerable.Range(0, 5).Any(dy => Enumerable.Range(0, 3).Any(dx => hard.Contains((x + dx, y + dy))));
            foreach(var template in Font) if(bitmap == template.Value && hasHardEvidence) letters.Add(new Glyph(template.Key, x, y, bitmap));
        }
        HashSet<(int Day, int Hour)> lettering = new();
        foreach(Glyph glyph in letters)
        {
            string[] rows = glyph.Bitmap.Split('/');
            for(int dy = 0; dy < 5; dy ++)
            for(int dx = 0; dx < 3; dx ++) if(rows[dy][dx] == '#') lettering.Add((glyph.X + dx, glyph.Y + dy));
        }
        foreach(Report r in reports) r.InLetter = lettering.Contains(r.Point);
        return (letters.ToArray(), days);
    }
    private static void SaveCharts(Report[] reports, Glyph[] letters, int days)
    {
        const int cell = 12;
        StringBuilder svg = new($"<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {days * cell + 100} 190\"><rect width=\"100%\" height=\"100%\" fill=\"white\"/><text x=\"50\" y=\"22\">Flagged SYNOP reports: days across, hours down</text>");
        for(int hour = 0; hour < 8; hour ++) svg.Append($"<text x=\"5\" y=\"{55 + hour * cell}\">{hour * 3:00}Z</text>");
        foreach(Report r in reports.Where(r => r.Reasons.Count > 0)) svg.Append($"<circle cx=\"{55 + r.Point.Day * cell}\" cy=\"{50 + r.Point.Hour * cell}\" r=\"3\" fill=\"#142c46\"/>");
        for(int day = 0; day < days; day += 7) svg.Append($"<text x=\"{55 + day * cell}\" y=\"160\" font-size=\"10\">{reports[0].Stamp.AddDays(day).ToString("MMM dd", CultureInfo.InvariantCulture)}</text>");
        File.WriteAllText("chart.csharp.svg", svg.Append("</svg>").ToString());
        StringBuilder crops = new($"<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {Math.Max(1, letters.Length) * 100} 150\"><rect width=\"100%\" height=\"100%\" fill=\"white\"/>");
        for(int i = 0; i < letters.Length; i ++)
        {
            Glyph glyph = letters[i];
            string[] rows = glyph.Bitmap.Split('/');
            for(int row = 0; row < 5; row ++)
            for(int col = 0; col < 3; col ++) if(rows[row][col] == '#') crops.Append($"<circle cx=\"{i * 100 + 25 + col * 20}\" cy=\"{25 + row * 20}\" r=\"6\" fill=\"#142c46\"/>");
            crops.Append($"<text x=\"{i * 100 + 20}\" y=\"135\">Day {glyph.X}</text>");
        }
        File.WriteAllText("letters.csharp.svg", crops.Append("</svg>").ToString());
    }
    public static void Main()
    {
        Report[] reports = Parse("messages.txt");
        var result = Detect(reports);
        using(StreamWriter csv = new("corrupted.csharp.csv"))
        {
            csv.WriteLine("id,date,hour_utc,in_letter,reasons");
            foreach(Report r in reports.Where(r => r.Reasons.Count > 0)) csv.WriteLine($"{r.Id},{r.Stamp:yyyy-MM-dd},{r.Stamp.Hour:00},{(r.InLetter ? 1 : 0)},\"{string.Join("; ", r.Reasons).Replace("\"", "\"\"")}\"");
        }
        SaveCharts(reports, result.Letters, result.Days);
        string word = new string(result.Letters.Select(glyph => glyph.Letter).ToArray()).ToUpperInvariant();
        Console.WriteLine($"Reports: {reports.Length}; corrupted: {reports.Count(r => r.Reasons.Count > 0)}; recovered flag: SIGMOID_{word}");
    }
}
