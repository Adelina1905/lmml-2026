"""LMML 2026 - task_spain "Feliz Navidad!"

Rebuilds the office's Christmas board from the pilots' receipts.

Steps
1. Join every receipt with the office price guide (on Product).
2. Price per unit = Total_paid_eur / Quantity.
3. Office rule: keep a receipt only if Fair_min_eur <= price per unit <= Fair_max_eur
   (cheaper = fake, more expensive = tourist trap).
4. Board = Pilot_no (21 columns, 1..21) x List_no (6 rows, 1 = Jamon at the top
   ... 6 = Barquillos at the bottom). A cell is filled if that pilot brought that
   souvenir for a fair price at least once.
5. Read the word on the board -> flag.

Run from inside this folder:  python solve.py
Outputs go to results/ and FLAG.txt.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RES = HERE / "results"
RES.mkdir(exist_ok=True)

N_PILOTS = 21
N_ITEMS = 6

# ---------------------------------------------------------------- load + rule
guide = pd.read_csv(DATA / "office_price_guide.csv")
purchases = pd.read_csv(DATA / "purchases.csv")

df = purchases.merge(guide, on="Product", how="left", validate="many_to_one")
assert df["List_no"].notna().all(), "every receipt must match a product in the guide"

df["unit_price"] = df["Total_paid_eur"] / df["Quantity"]
df["verdict"] = np.select(
    [df["unit_price"] < df["Fair_min_eur"], df["unit_price"] > df["Fair_max_eur"]],
    ["fake (too cheap)", "tourist trap (too expensive)"],
    default="fair",
)
fair = df[df["verdict"] == "fair"]

# ---------------------------------------------------------------- the board
counts = (
    pd.crosstab(fair["List_no"], fair["Pilot_no"])
    .reindex(index=range(1, N_ITEMS + 1), columns=range(1, N_PILOTS + 1), fill_value=0)
)
board = (counts > 0).astype(int)
board.to_csv(RES / "board_matrix.csv")

ascii_rows = ["".join("#" if v else "." for v in board.loc[r]) for r in board.index]

# Split into glyphs: a glyph is a run of non-empty pilot columns.
col_has_ink = board.sum(axis=0).to_numpy() > 0
glyph_spans, start = [], None
for i, ink in enumerate(list(col_has_ink) + [False]):
    if ink and start is None:
        start = i
    elif not ink and start is not None:
        glyph_spans.append((start, i - 1))
        start = None

# 6-row bitmaps of the glyphs we read on the board (rows = List_no 1..6).
# Lowercase-shaped glyphs start lower (c, u, i); the task says to submit them uppercase.
GLYPHS = {
    "c": ["...", "...", "###", "#..", "#..", "###"],
    "R": ["##.", "#.#", "#.#", "##.", "#.#", "#.#"],
    "u": ["...", "...", "#.#", "#.#", "#.#", "###"],
    "C": ["###", "#..", "#..", "#..", "#..", "###"],
    "i": [".", "#", ".", "#", "#", "#"],
    "O": ["###", "#.#", "#.#", "#.#", "#.#", "###"],
}
read = []
for a, b in glyph_spans:
    bitmap = [row[a : b + 1] for row in ascii_rows]
    match = [k for k, v in GLYPHS.items() if v == bitmap]
    assert len(match) == 1, f"unrecognised glyph in pilot columns {a+1}-{b+1}: {bitmap}"
    read.append(match[0])
word = "".join(read).upper()
flag = f"SIGMOID_{word}"

# ---------------------------------------------------------------- figure
INK = "#1f3a5f"
SURFACE = "#ffffff"
fig, ax = plt.subplots(figsize=(13, 4.8))
ax.imshow(board.to_numpy(), cmap=matplotlib.colors.ListedColormap([SURFACE, INK]),
          aspect="equal", vmin=0, vmax=1)
for x in np.arange(-0.5, N_PILOTS, 1):
    ax.axvline(x, color="#d0d4da", lw=0.8)
for y in np.arange(-0.5, N_ITEMS, 1):
    ax.axhline(y, color="#d0d4da", lw=0.8)
ax.set_xticks(range(N_PILOTS))
ax.set_xticklabels(range(1, N_PILOTS + 1), fontsize=9)
ax.set_yticks(range(N_ITEMS))
ax.set_yticklabels([f"{n}. {p}" for n, p in zip(guide["List_no"], guide["Product"])], fontsize=10)
ax.set_xlabel("Pilot_no (one column per pilot)")
ax.set_ylabel("Office list (List_no)")
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.set_title(
    f"Rebuilt Christmas board: fair-price souvenirs only ({len(fair)} of {len(df)} receipts)\n"
    f"word on the board: {''.join(read)}  ->  flag {flag}",
    fontsize=12, loc="left",
)
fig.tight_layout()
fig.savefig(RES / "board.png", dpi=150)
plt.close(fig)

# Control figure: what the board would look like with the wrong filters.
fig, axes = plt.subplots(3, 1, figsize=(10, 7.5))
controls = {
    "Fakes only (unit price below the fair range)": df["verdict"] == "fake (too cheap)",
    "Tourist traps only (unit price above the fair range)": df["verdict"] == "tourist trap (too expensive)",
    "Wrong rule: Total_paid_eur inside the range (ignores Quantity)":
        df["Total_paid_eur"].between(df["Fair_min_eur"], df["Fair_max_eur"]),
}
for ax, (title, mask) in zip(axes, controls.items()):
    sub = df[mask]
    m = (pd.crosstab(sub["List_no"], sub["Pilot_no"])
         .reindex(index=range(1, N_ITEMS + 1), columns=range(1, N_PILOTS + 1), fill_value=0) > 0)
    ax.imshow(m.to_numpy(), cmap=matplotlib.colors.ListedColormap([SURFACE, "#8a94a3"]),
              aspect="equal", vmin=0, vmax=1)
    ax.set_title(f"{title}: {int(mask.sum())} receipts", fontsize=10, loc="left")
    ax.set_xticks(range(N_PILOTS)); ax.set_xticklabels(range(1, N_PILOTS + 1), fontsize=7)
    ax.set_yticks(range(N_ITEMS)); ax.set_yticklabels(guide["Product"], fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
fig.tight_layout()
fig.savefig(RES / "controls_wrong_filters.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------- report
lines = [
    "LMML 2026 - task_spain report",
    "=" * 40,
    f"receipts: {len(df)}",
    "verdicts:",
    df["verdict"].value_counts().to_string(),
    "",
    "fair receipts per product:",
    fair["Product"].value_counts().reindex(guide["Product"]).fillna(0).astype(int).to_string(),
    "",
    "receipts within 1 cent of a range boundary: "
    + str(int(((df["unit_price"] - df["Fair_min_eur"]).abs().lt(0.01)
               | (df["unit_price"] - df["Fair_max_eur"]).abs().lt(0.01)).sum())),
    "",
    "board (rows = List_no 1..6, columns = Pilot_no 1..21, # = fair purchase):",
    *[f"{r} {guide.loc[guide.List_no == r, 'Product'].item():<11}{row}"
      for r, row in zip(board.index, ascii_rows)],
    "",
    "glyphs (pilot columns): "
    + ", ".join(f"{g} ({a+1}-{b+1})" for g, (a, b) in zip(read, glyph_spans)),
    f"word: {word}",
    f"FLAG: {flag}",
]
report = "\n".join(lines)
(RES / "report.txt").write_text(report + "\n", encoding="utf-8")
(HERE / "FLAG.txt").write_text(flag + "\n", encoding="utf-8")
print(report)
