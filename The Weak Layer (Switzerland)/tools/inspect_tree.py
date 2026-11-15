from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

def main():
    folder = Path(__file__).resolve().parents[1]
    data = pd.read_csv(folder / "data" / "train-input.csv")
    features = data.drop(columns = ["slab_released"])
    labels = data["slab_released"]
    model = DecisionTreeClassifier(random_state = 42)
    model.fit(features, labels)
    tree_text = export_text(model, feature_names = list(features.columns), max_depth = model.get_depth(), decimals = 6, show_weights = True)
    results = folder / "results"
    results.mkdir(exist_ok = True)
    (results / "practice-tree.txt").write_text(tree_text, encoding = "utf-8")
    print(f"Training rows: {len(data)}; input features: {features.shape[1]}")
    print(f"Random state: {model.random_state}; all other classifier parameters: defaults")
    print(f"Tree nodes: {model.tree_.node_count}; leaves: {model.get_n_leaves()}; depth: {model.get_depth()}")
    print(f"Training accuracy: {model.score(features, labels):.6%}")
    print("Full tree exported to results/practice-tree.txt")
    print(tree_text)

if(__name__ == "__main__"):
    main()
