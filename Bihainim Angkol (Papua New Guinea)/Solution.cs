// C# conversion of main.py: shortest paths plus the existing readable fishing policy.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
namespace BihainimAngkol;
public static class Solution
{
    private static readonly string[] Grid = { "....TT.", "####~##", "....ss.", "....ss.", ".......", "..H...." };
    private static readonly (string Name, int Row, int Col)[] Moves = { ("NORTH", -1, 0), ("SOUTH", 1, 0), ("EAST", 0, 1), ("WEST", 0, -1) };
    private static readonly (int Row, int Col) Beach = (5, 2), Reef = (0, 4);
    private static readonly (int Row, int Col)[] Seagrass = { (2, 4), (2, 5), (3, 4), (3, 5) };
    private static readonly Dictionary<(int, int), int> HomeDistance = DistancesTo(Beach), ReefDistance = DistancesTo(Reef);
    private static readonly Dictionary<(int, int), Dictionary<(int, int), int>> GrassDistance = Seagrass.ToDictionary(cell => cell, DistancesTo);
    private static bool IsWater(int row, int col) => row >= 0 && row < Grid.Length && col >= 0 && col < Grid[0].Length && Grid[row][col] != '#';
    private static Dictionary<(int, int), int> DistancesTo((int Row, int Col) goal)
    {
        Dictionary<(int, int), int> distances = new() { [goal] = 0 };
        Queue<(int Row, int Col)> queue = new();
        queue.Enqueue(goal);
        while(queue.Count > 0)
        {
            var cell = queue.Dequeue();
            foreach(var move in Moves)
            {
                var next = (cell.Row + move.Row, cell.Col + move.Col);
                if(!IsWater(next.Item1, next.Item2) || distances.ContainsKey(next)) continue;
                distances[next] = distances[cell] + 1;
                queue.Enqueue(next);
            }
        }
        return distances;
    }
    private static string MoveToward(int row, int col, Dictionary<(int, int), int> distances) => Moves.OrderBy(move => distances.GetValueOrDefault((row + move.Row, col + move.Col), 99)).First().Name;
    public static string ChooseAction(int row, int col, int hold, int stepsLeft)
    {
        var here = (row, col);
        string goHome = here == Beach ? "WAIT" : MoveToward(row, col, HomeDistance);
        if(hold == 6 || stepsLeft <= HomeDistance[here]) return goHome;
        if(Grid[row][col] == 'T') return "FISH";
        // Go to the 90% reef only when there is time for a bite and the return trip.
        if(stepsLeft >= ReefDistance[here] + 1 + HomeDistance[Reef]) return MoveToward(row, col, ReefDistance);
        if(Grid[row][col] == 's') return "FISH";
        var grass = Seagrass.OrderBy(cell => GrassDistance[cell][here]).First();
        if(stepsLeft >= GrassDistance[grass][here] + 1 + HomeDistance[grass]) return MoveToward(row, col, GrassDistance[grass]);
        return goHome;
    }
    public static void Main()
    {
        using StreamWriter output = new("test-output.csv");
        output.WriteLine("row,col,hold,steps_left,action");
        string[] lines = File.ReadAllLines("test-input.csv");
        string[] header = lines[0].TrimStart('\uFEFF').Split(',');
        foreach(string line in lines.Skip(1).Where(line => !string.IsNullOrWhiteSpace(line)))
        {
            string[] cells = line.Split(',');
            int Read(string name) => int.Parse(cells[Array.IndexOf(header, name)]);
            int row = Read("row"), col = Read("col"), hold = Read("hold"), left = Read("steps_left");
            output.WriteLine($"{row},{col},{hold},{left},{ChooseAction(row, col, hold, left)}");
        }
    }
}
