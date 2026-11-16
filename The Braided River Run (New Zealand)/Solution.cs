// C# conversion of main.py and the map rules it uses from environment.py.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
namespace BraidedRiverRun;
public static class Solution
{
    private readonly record struct State(int Row, int Col, int Bridges, int Phase);
    private static readonly (char Action, int Row, int Col)[] Moves = { ('U', -1, 0), ('D', 1, 0), ('L', 0, -1), ('R', 0, 1), ('W', 0, 0) };
    private sealed class Valley
    {
        public string[] Grid = Array.Empty<string>(), Heights = Array.Empty<string>();
        public int Period = 10, Low = 3, SwampCost = 3, ClimbCost = 1, InitialMask;
        public (int Row, int Col) Start, Goal;
        public Dictionary<char, int> BridgeBits = new(), WinchMasks = new();
        public bool InBounds(int row, int col) => row >= 0 && row < Grid.Length && col >= 0 && col < Grid[0].Length;
        public int Duration(int row, int col, int nextRow, int nextCol) => row == nextRow && col == nextCol ? 1 : (Grid[nextRow][nextCol] == ',' ? SwampCost : 1) + ClimbCost * Math.Max(0, Heights[nextRow][nextCol] - Heights[row][col]);
    }
    private static Valley LoadMap(string path)
    {
        Valley valley = new();
        List<string> grid = new(), heights = new();
        List<char> opened = new();
        Dictionary<char, char[]> winches = new();
        string section = "header";
        foreach(string raw in File.ReadLines(path))
        {
            if(string.IsNullOrWhiteSpace(raw)) continue;
            if(section != "header")
            {
                if(raw.Trim() == "heights") { section = "heights"; continue; }
                if(section == "map") grid.Add(raw); else heights.Add(raw);
                continue;
            }
            string[] parts = raw.Split('#')[0].Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if(parts.Length == 0) continue;
            switch(parts[0])
            {
                case "map": section = "map"; break;
                case "tide_period": valley.Period = int.Parse(parts[1]); break;
                case "tide_low": valley.Low = int.Parse(parts[1]); break;
                case "swamp_cost": valley.SwampCost = int.Parse(parts[1]); break;
                case "climb_cost": valley.ClimbCost = int.Parse(parts[1]); break;
                case "open": opened.AddRange(parts.Skip(1).Select(value => value[0])); break;
                case "winch": winches[parts[1][0]] = parts.Skip(2).Select(value => value[0]).ToArray(); break;
                default: throw new InvalidDataException($"Unknown header: {parts[0]}");
            }
        }
        if(grid.Count == 0 || grid.Any(row => row.Length != grid[0].Length)) throw new InvalidDataException("Invalid map dimensions.");
        valley.Grid = grid.ToArray();
        valley.Heights = heights.Count > 0 ? heights.ToArray() : grid.Select(row => new string('0', row.Length)).ToArray();
        char[] bridges = string.Concat(grid).Where(symbol => char.IsUpper(symbol) && symbol != 'S' && symbol != 'G').Distinct().OrderBy(symbol => symbol).ToArray();
        if(bridges.Length > 30) throw new NotSupportedException("Use a wider bridge mask for more than 30 bridges.");
        for(int i = 0; i < bridges.Length; i ++) valley.BridgeBits[bridges[i]] = 1 << i;
        valley.InitialMask = opened.Sum(bridge => valley.BridgeBits[bridge]);
        foreach(var pair in winches) valley.WinchMasks[pair.Key] = pair.Value.Sum(bridge => valley.BridgeBits[bridge]);
        for(int row = 0; row < grid.Count; row ++)
        for(int col = 0; col < grid[row].Length; col ++)
        {
            if(grid[row][col] == 'S') valley.Start = (row, col);
            if(grid[row][col] == 'G') valley.Goal = (row, col);
        }
        return valley;
    }
    private static (string Route, int Minutes) Solve(Valley valley)
    {
        State start = new(valley.Start.Row, valley.Start.Col, valley.InitialMask, 0);
        Dictionary<State, int> best = new() { [start] = 0 };
        Dictionary<State, (State Previous, char Action)> previous = new();
        PriorityQueue<State, int> queue = new();
        queue.Enqueue(start, 0);
        // Equal position, bridge mask, and tide phase have identical possible futures.
        while(queue.TryDequeue(out State state, out int time))
        {
            if(time != best[state]) continue;
            if((state.Row, state.Col) == valley.Goal)
            {
                StringBuilder reversed = new();
                State cursor = state;
                while(cursor != start) { var step = previous[cursor]; reversed.Append(step.Action); cursor = step.Previous; }
                return (new string(reversed.ToString().Reverse().ToArray()), time);
            }
            foreach(var move in Moves)
            {
                int row = state.Row + move.Row, col = state.Col + move.Col;
                if(!valley.InBounds(row, col)) continue;
                char symbol = valley.Grid[row][col];
                if(symbol == '#' || symbol == '=') continue;
                int arrival = time + valley.Duration(state.Row, state.Col, row, col);
                if(symbol == '~' && arrival % valley.Period >= valley.Low) continue;
                if(valley.BridgeBits.TryGetValue(symbol, out int bit) && (state.Bridges & bit) == 0) continue;
                int mask = move.Action != 'W' && valley.WinchMasks.TryGetValue(symbol, out int toggled) ? state.Bridges ^ toggled : state.Bridges;
                State next = new(row, col, mask, arrival % valley.Period);
                if(arrival >= best.GetValueOrDefault(next, int.MaxValue)) continue;
                best[next] = arrival;
                previous[next] = (state, move.Action);
                queue.Enqueue(next, arrival);
            }
        }
        throw new InvalidOperationException("No legal route reaches the goal.");
    }
    public static void Main(string[] args)
    {
        var result = Solve(LoadMap(args.Length > 0 ? args[0] : "test-input.txt"));
        File.WriteAllText(args.Length > 1 ? args[1] : "solution.txt", $"# {result.Minutes} minutes, {result.Route.Length} actions\n{result.Route}\n");
        Console.WriteLine($"{result.Minutes} minutes");
    }
}
