# Reports only WHICH door index opened (no counts, no labels).
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))  # usage: python whichdoor.py first_model.npz model.npz
import open_vault as ov
z = np.load(ov.HERE / "vault.npz")
Xs = ov.load_sealed()
for f in sys.argv[1:]:
    m = np.load(f); S = Xs @ np.asarray(m["W"], float).T + np.asarray(m["b"], float)
    opened = []
    for d in range(len(z["tables"])):
        sub = dict(p=z["p"], n=z["n"], k=z["k"], tables=z["tables"][d:d+1], tags=z["tags"][d:d+1], words=z["words"][d:d+1])
        np.savez("/tmp/v1.npz", **sub)
        w = ov.try_vault(S, "/tmp/v1.npz", verbose=False)
        if w: opened.append((d, w[0]))
    print(f, opened)
