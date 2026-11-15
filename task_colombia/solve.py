"""LMML 2026 - Looks Good to Me! (Bogota Sound Check)

Finds the altered reviews in reviews.json using the trusted reference reviews,
then follows the archive note to recover the flag hidden in the archive seals.

Run from inside this folder:  python solve.py
Outputs: FLAG.txt, results/report.txt, results/altered_records.csv, results/*.png
"""
import csv
import hashlib
import json
import os
import unicodedata

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


data = json.load(open(os.path.join(HERE, "reviews.json"), encoding="utf-8"))
T, S = data["trusted_reference"], data["samples"]
log(f"trusted reviews: {len(T)}   suspect samples: {len(S)}")
log("archive note:", data["archive_note"])
assert all(r["sample_index"] == i for i, r in enumerate(S))

# ---------------------------------------------------------------------------
# Stage 1 - text alterations: characters that never occur in the trusted text
# ---------------------------------------------------------------------------
log("\n=== Stage 1: characters never seen in the trusted reviews ===")
trusted_chars = set("".join(r["summary"] + r["review_text"] for r in T))
text_altered = {}
for r in S:
    full = r["summary"] + " | " + r["review_text"]
    bad = [(i, ch) for i, ch in enumerate(full) if ch not in trusted_chars and ch != "|"]
    if bad:
        text_altered[r["sample_index"]] = bad
        i, ch = bad[0]
        ctx = full[max(0, i - 25):i + 25].replace("\n", " | ")
        log(f"  sample {r['sample_index']:4d}  rating {r['rating']}  "
            f"U+{ord(ch):04X} {unicodedata.name(ch)!s:28s}  ...{ctx!r}")
log(f"-> {len(text_altered)} samples contain look-alike / invisible characters")

# ---------------------------------------------------------------------------
# Stage 2 - label alterations: rating contradicts the text
# A rating model is trained ONLY on the trusted reviews (clean labels).
# ---------------------------------------------------------------------------
log("\n=== Stage 2: ratings that contradict the review text ===")


def clean_text(r):
    # drop the foreign characters so they don't influence the rating model
    t = r["summary"] + " . " + r["review_text"]
    return "".join(ch for ch in t if ch in trusted_chars)


XT = [clean_text(r) for r in T]
XS = [clean_text(r) for r in S]
yT = np.array([r["rating"] for r in T])
yS = np.array([r["rating"] for r in S])

vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2).fit(XT)
FT, FS = vec.transform(XT), vec.transform(XS)
model = LogisticRegression(C=8, max_iter=5000)
stars = np.arange(1, 6)

# honest validation on trusted data (10-fold out-of-fold predictions)
oof = cross_val_predict(model, FT, yT, cv=StratifiedKFold(10, shuffle=True, random_state=0),
                        method="predict_proba")
E_T = oof @ stars
log(f"trusted 10-fold CV: accuracy {np.mean(oof.argmax(1) + 1 == yT):.3f}, "
    f"MAE of expected rating {np.mean(np.abs(E_T - yT)):.3f}")
log("trusted CV mean expected rating per true rating: " +
    ", ".join(f"{k}->{E_T[yT == k].mean():.2f}" for k in range(1, 6)))

model.fit(FT, yT)
E_S = model.predict_proba(FS) @ stars  # expected rating predicted from the text

# Survey every rating class: how many reviews sit on the 'wrong side'?
log("\nSuspect reviews per given rating, sorted by the model's expected rating:")
for k in range(1, 6):
    e = np.sort(E_S[yS == k])
    log(f"  rating {k}: n={len(e):4d}  lowest 3 E={np.round(e[:3], 2)}  highest 3 E={np.round(e[-3:], 2)}")


def natural_break(values, direction):
    """Indices above the biggest gap in the sorted list (top side for 'high', bottom for 'low')."""
    v = np.sort(values)[::-1] if direction == "high" else np.sort(values)
    gaps = np.abs(np.diff(v[:15]))
    j = int(np.argmax(gaps))
    return v[j], v[j + 1], gaps[j]


label_altered = {}
# rating 3 (neutral) but text reads clearly positive
idx3 = np.where(yS == 3)[0]
a, b, g = natural_break(E_S[idx3], "high")
cut3 = (a + b) / 2
log(f"\nRating-3 reviews: largest gap in expected rating is between {a:.2f} and {b:.2f} "
    f"(gap {g:.2f}) -> threshold E >= {cut3:.2f}")
for i in idx3[np.argsort(-E_S[idx3])]:
    if E_S[i] >= cut3:
        label_altered[int(i)] = f"rated 3, text reads positive (E={E_S[i]:.2f})"
        log(f"  sample {i:4d}  E={E_S[i]:.2f}  {S[i]['summary'][:60]!r}")
# rating 4 (positive) but text reads neutral / negative
idx4 = np.where(yS == 4)[0]
a, b, g = natural_break(E_S[idx4], "low")
cut4 = (a + b) / 2
log(f"\nRating-4 reviews: largest gap in expected rating is between {a:.2f} and {b:.2f} "
    f"(gap {g:.2f}) -> threshold E <= {cut4:.2f}")
for i in idx4[np.argsort(E_S[idx4])]:
    if E_S[i] <= cut4:
        label_altered[int(i)] = f"rated 4, text reads neutral/negative (E={E_S[i]:.2f})"
        log(f"  sample {i:4d}  E={E_S[i]:.2f}  {S[i]['summary'][:60]!r}")
log("\nRatings 1, 2 and 5: no review sits on the wrong side (no 1/2-star text reads positive,"
    " no 5-star text reads negative) -> nothing flagged there.")
log(f"-> {len(label_altered)} samples with a contradicting rating "
    f"(sample 371 even says '4 stars is my tops' but is rated 3)")

# ---------------------------------------------------------------------------
# Stage 3 - follow the archive note
# ---------------------------------------------------------------------------
log("\n=== Stage 3: open the seals ===")
altered = sorted(set(text_altered) | set(label_altered))
key = ",".join(str(i) for i in altered)
mask = hashlib.sha256(key.encode("utf-8")).digest()
log(f"altered records ({len(altered)}): {key}")
log(f"SHA-256 mask: {mask.hex()}")

hits = []
for src in ("trusted_reference", "samples"):
    for pos, r in enumerate(data[src]):
        plain = bytes(x ^ y for x, y in zip(bytes.fromhex(r["archive_tag"]), mask)).rstrip(b"\0")
        if plain and all(32 <= c < 127 for c in plain):
            hits.append((src, pos, plain.decode()))
for src, pos, msg in hits:
    log(f"readable seal: {src}[{pos}] -> {msg}")
assert len(hits) == 1 and hits[0][2].startswith("SIGMOID_"), "no unique readable seal"
flag = hits[0][2]
log(f"\nFLAG: {flag}")

# Sanity checks: each half alone, or any single record removed, opens nothing
def readable_count(ids):
    m = hashlib.sha256(",".join(map(str, sorted(ids))).encode()).digest()
    n = 0
    for src in ("trusted_reference", "samples"):
        for r in data[src]:
            p = bytes(x ^ y for x, y in zip(bytes.fromhex(r["archive_tag"]), m)).rstrip(b"\0")
            n += bool(p) and all(32 <= c < 127 for c in p)
    return n


log("\nControls (number of readable seals):")
log(f"  only text-altered set   : {readable_count(text_altered)}")
log(f"  only label-altered set  : {readable_count(label_altered)}")
drop = [readable_count([x for x in altered if x != a]) for a in altered]
log(f"  full set minus any one record (24 tries): max {max(drop)} readable")

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
with open(os.path.join(HERE, "FLAG.txt"), "w") as f:
    f.write(flag + "\n")
with open(os.path.join(RES, "altered_records.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["sample_index", "rating", "alteration", "evidence", "summary"])
    for i in altered:
        if i in text_altered:
            ch = text_altered[i][0][1]
            w.writerow([i, S[i]["rating"], "text", f"U+{ord(ch):04X} {unicodedata.name(ch)}", S[i]["summary"]])
        else:
            w.writerow([i, S[i]["rating"], "rating", label_altered[i], S[i]["summary"]])

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, k, cut, side in ((axes[0], 3, cut3, ">="), (axes[1], 4, cut4, "<=")):
        ii = np.where(yS == k)[0]
        order = ii[np.argsort(E_S[ii])]
        col = ["#d62728" if int(i) in label_altered else "#7f7f7f" for i in order]
        ax.bar(range(len(order)), E_S[order], color=col)
        ax.axhline(cut, ls="--", c="k", lw=1)
        ax.set_title(f"Suspect reviews rated {k}: expected rating from text\n(red = flagged, line = threshold {side} {cut:.2f})")
        ax.set_xlabel("reviews sorted by expected rating")
        ax.set_ylabel("expected rating (trusted model)")
        ax.set_ylim(1, 5)
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "rating_contradictions.png"), dpi=120)
    plt.close()
    log("\nplot saved: results/rating_contradictions.png")
except ImportError:
    log("matplotlib not installed: plot skipped")

with open(os.path.join(RES, "report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG) + "\n")
