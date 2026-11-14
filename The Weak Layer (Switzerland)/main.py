from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def recover_word(data, random_state = 42):
    features = data.drop(columns = ["slab_released"])
    model = DecisionTreeClassifier(random_state = random_state)
    model.fit(features, data["slab_released"])
    tree = model.tree_
    letters = []
    pending = [0]
    while(pending):
        node = pending.pop()
        feature = tree.feature[node]
        if(feature < 0):
            continue
        letters.append(features.columns[feature][0].upper())
        pending.extend((tree.children_right[node], tree.children_left[node]))
    return "".join(letters)

def main():
    data = pd.read_csv("test-input.csv")
    word = recover_word(data)
    Path("test-output.txt").write_text(f"SIGMOID_{word}\n", encoding = "utf-8")

if(__name__ == "__main__"):
    main()
