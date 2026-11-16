// C# conversion of optimize_routing.py, main.py, and the deterministic gateway in model_utils.py.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
namespace TerangaTokenDiet;
public static class Solution
{
    public sealed record Choice(string tier, string style, bool fewshot, bool context);
    public sealed record Call(string Answer, int Tokens);
    private const string Cheap = "You are the Teranga FM quick-reply bot. Answer fast.";
    private const string Large = "You are the Teranga FM flagship assistant, an exhaustively trained large language model serving farmers and residents across the Peanut Basin. Always be thorough, polite, and comprehensive in every reply you give, and make sure the person asking never feels rushed.";
    private const string Verbose = "Please answer the following question in a warm, detailed, and complete manner, making sure to explain your reasoning, add helpful context, and double check that nothing important has been left out before you respond.";
    private const string Examples = "Example Q: What is the capital of Mali?\nExample A: Bamako is the capital of Mali.\nExample Q: What currency does Nigeria use?\nExample A: Nigeria uses the Nigerian naira.";
    // Python's Unicode word class includes letters and numbers; make that intention explicit here.
    private static int CountTokens(string text) => Regex.Matches(text, @"[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]").Count;
    public static Call BuildCall(JsonElement question, Choice choice)
    {
        string? snippet = question.TryGetProperty("context_snippet", out JsonElement value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;
        List<string> prompt = new() { choice.tier == "cheap" ? Cheap : Large, choice.style == "terse" ? "Answer:" : Verbose };
        if(choice.fewshot) prompt.Add(Examples);
        if(choice.context && !string.IsNullOrEmpty(snippet)) prompt.Add($"Context: {snippet}");
        prompt.Add($"Q: {question.GetProperty("question").GetString()}");
        bool assisted = choice.tier == "large" || choice.context && !string.IsNullOrEmpty(snippet);
        string[] included = question.GetProperty("slots").EnumerateArray().Where(slot => assisted || !slot.GetProperty("large_only").GetBoolean()).Select(slot => slot.GetProperty("phrasings")[0].GetString()!).ToArray();
        string facts = included.Length > 0 ? string.Join("; ", included) + "." : "I'm not sure, please ask a human agent.";
        string answer = choice.style == "verbose" ? "Thanks so much for reaching out to the Teranga FM helpline! " + facts + " We hope this fully answers your question, feel free to call back any time with more!" : facts;
        return new Call(answer, CountTokens(string.Join("\n", prompt)) + CountTokens("A: " + answer));
    }
    private static bool Passes(JsonElement question, string answer) => question.GetProperty("slots").EnumerateArray().All(slot => slot.GetProperty("phrasings").EnumerateArray().Any(phrase => answer.ToLowerInvariant().Contains(phrase.GetString()!.ToLowerInvariant(), StringComparison.Ordinal)));
    public static void Main()
    {
        using JsonDocument bank = JsonDocument.Parse(File.ReadAllText("dataset/fact_bank.json"));
        Dictionary<string, Choice> routing = new();
        List<object> answers = new();
        int total = 0;
        foreach(JsonProperty question in bank.RootElement.EnumerateObject())
        {
            Choice? best = null;
            Call? bestCall = null;
            // Questions are independent, so choose the cheapest passing option for each one.
            foreach(string tier in new[] { "cheap", "large" })
            foreach(string style in new[] { "verbose", "terse" })
            foreach(bool fewshot in new[] { false, true })
            foreach(bool context in new[] { false, true })
            {
                Choice candidate = new(tier, style, fewshot, context);
                Call call = BuildCall(question.Value, candidate);
                if(Passes(question.Value, call.Answer) && (bestCall == null || call.Tokens < bestCall.Tokens)) { best = candidate; bestCall = call; }
            }
            if(best == null || bestCall == null) throw new InvalidOperationException($"No passing route for {question.Name}");
            routing[question.Name] = best;
            answers.Add(new { id = question.Name, answer = bestCall.Answer });
            total += bestCall.Tokens;
        }
        if(total > 1200) throw new InvalidOperationException($"Token budget exceeded: {total}");
        JsonSerializerOptions options = new() { WriteIndented = true };
        File.WriteAllText("routing_config.json", JsonSerializer.Serialize(routing, options));
        File.WriteAllText("answers.json", JsonSerializer.Serialize(answers, options));
        Console.WriteLine($"Quality: {routing.Count}/{routing.Count}; tokens: {total}/1200");
    }
}
