// C# conversion of solve.py. Run from this task folder; uses only the .NET libraries.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json.Nodes;
namespace WaterlineReplay;
public static class Solution
{
    public static void Main(string[] args)
    {
        string path = args.Length > 0 ? args[0] : "test-input.json";
        JsonArray events = JsonNode.Parse(File.ReadAllText(path))!["events"]!.AsArray();
        var runs = events.Select(item => item!).GroupBy(item => Text(item, "run_id")).ToDictionary(group => group.Key, group => group.ToArray());
        // A restored tool result must agree with the assistant's yes/no answer.
        string badRun = runs.Single(pair => IsWrong(pair.Value)).Key;
        HashSet<string> goodSources = runs.Where(pair => pair.Key != badRun).SelectMany(pair => Sources(pair.Value)).ToHashSet();
        string culprit = Sources(runs[badRun]).Except(goodSources).Single();
        JsonNode memory = runs[badRun].Single(item => Text(item, "event") == "memory.result" && Text(item["item"]!, "source_id") == culprit);
        Console.WriteLine($"Expired note {culprit}: valid until {Text(memory["item"]!, "valid_until")}, used at {Text(memory, "timestamp")}");
        File.WriteAllText(Path.Combine(Path.GetDirectoryName(Path.GetFullPath(path))!, "answer.txt"), culprit + Environment.NewLine);
    }
    private static bool IsWrong(JsonNode[] events)
    {
        string status = Text(events.First(item => Text(item, "event") == "tool.result")["result"]!, "status");
        string answer = Text(events.First(item => Text(item, "event") == "assistant.response"), "text");
        return (status == "RESTORED") == answer.StartsWith("no", StringComparison.OrdinalIgnoreCase);
    }
    private static IEnumerable<string> Sources(JsonNode[] events) => events.First(item => Text(item, "event") == "llm.request")["messages"]!.AsArray().Where(message => message?["source_id"] != null).Select(message => Text(message!, "source_id"));
    private static string Text(JsonNode item, string key) => item[key]!.GetValue<string>();
}
