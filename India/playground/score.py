import os
import pandas as pd

truth = pd.read_csv("data/train-output.csv")
scores = pd.read_csv("output/all_scores.csv")
mine = pd.read_csv("output/output.csv")

# Get exactly the filenames used in the experiment
used = set(
    os.path.basename(p)
    for p in paths.values()
)

def as_set(df):
    return {
        tuple(sorted((a, b)))
        for a, b in zip(df["image_a"], df["image_b"])
        if a in used and b in used and a != b
    }

T = as_set(truth)
M = as_set(mine)

correct = T & M

recall = len(correct) / max(len(T), 1)
precision = len(correct) / max(len(M), 1)

print("True pairs:", len(T))
print("Found pairs:", len(M))
print("Correct pairs:", len(correct))

print("Recall:", recall)
print("Precision:", precision)

print("Missed:", sorted(T - M))
print("Wrong:", sorted(M - T))