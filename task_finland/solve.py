"""LMML 2026 - task_finland: clean the route log, rebuild the network, rank towns by degree."""
import unicodedata
import sys
from pathlib import Path
import pandas as pd
import networkx as nx

HERE = Path(__file__).resolve().parent
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "routes.csv"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

# Swedish (and short) names -> official Finnish name (all keys already lower-case, diacritics stripped)
SWEDISH_TO_FINNISH = {
    "abo": "turku", "alavo": "alavus", "birkala": "pirkkala", "bjorneborg": "pori",
    "borga": "porvoo", "brahestad": "raahe", "esbo": "espoo", "fredrikshamn": "hamina",
    "grankulla": "kauniainen", "hango": "hanko", "helsingfors": "helsinki",
    "hogfors": "karkkila", "hyvinge": "hyvinkaa", "idensalmi": "iisalmi",
    "ikalis": "ikaalinen", "ilmola": "ilmajoki", "inga": "inkoo", "jakobstad": "pietarsaari",
    "kajana": "kajaani", "karleby": "kokkola", "kasko": "kaskinen", "kervo": "kerava",
    "keuru": "keuruu", "kimitoon": "kemionsaari", "korsholm": "mustasaari",
    "kristinestad": "kristiinankaupunki", "kronoby": "kruunupyy", "kumo": "kokemaki",
    "kyrkslatt": "kirkkonummi", "lahtis": "lahti", "lappo": "lapua", "larsmo": "luoto",
    "lojo": "lohja", "lovisa": "loviisa", "lundo": "lieto", "malax": "maalahti",
    "nykarleby": "uusikaarlepyy", "nystad": "uusikaupunki", "narpes": "narpio",
    "nadendal": "naantali", "pargas": "parainen", "raseborg": "raasepori", "reso": "raisio",
    "raumo": "rauma", "s:t karins": "kaarina", "s:t michel": "mikkeli", "sibbo": "sipoo",
    "sjundea": "siuntio", "tammerfors": "tampere", "tavastehus": "hameenlinna",
    "tornea": "tornio", "traskanda": "jarvenpaa", "tusby": "tuusula", "uleaborg": "oulu",
    "ulvsby": "ulvila", "vanda": "vantaa", "vasa": "vaasa", "vichtis": "vihti",
    "villmanstrand": "lappeenranta", "virmo": "mynamaki", "vora": "voyri",
}

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def clean(name: str) -> str:
    s = strip_accents(str(name)).strip().strip(".").strip().lower()
    s = " ".join(s.split())
    return SWEDISH_TO_FINNISH.get(s, s)

raw = pd.read_csv(CSV, dtype=str, keep_default_na=False)
lines = []
p = lines.append
p(f"raw rows: {len(raw)}")
p(f"raw distinct town spellings: {pd.concat([raw.from_town, raw.to_town]).nunique()}")

# naive (wrong) analysis the junior analyst did
naive = pd.concat([raw.from_town, raw.to_town]).value_counts()
p("\nNaive group-by on raw log (top 8):\n" + naive.head(8).to_string())

df = raw.copy()
dups = df.duplicated().sum()
df = df.drop_duplicates()
p(f"\nexact duplicate rows removed: {dups}")
df["a"] = df.from_town.map(clean)
df["b"] = df.to_town.map(clean)
loops = (df.a == df.b).sum()
p(f"self-loops after cleaning (A->A) removed: {loops}")
df = df[df.a != df.b]

G = nx.Graph()
G.add_edges_from(zip(df.a, df.b))  # undirected, simple graph: each town pair counted once
p(f"\nlog entries after cleaning: {len(df)}")
p(f"towns: {G.number_of_nodes()}  connections: {G.number_of_edges()}")

deg = pd.Series(dict(G.degree())).sort_values(ascending=False)
p("\nTop 12 towns by distinct connections:\n" + deg.head(12).to_string())
top7 = deg.head(7)
top8 = deg.head(8)
p(f"\ntop-8 degrees all distinct: {top8.nunique() == 8}")
word = "".join(t[0] for t in top7.index).upper()
flag = f"SIGMOID_{word}"
p(f"\nFLAG: {flag}")

assert G.number_of_nodes() == 116, G.number_of_nodes()
assert G.number_of_edges() == 448, G.number_of_edges()
assert top8.nunique() == 8

deg.rename("connections").rename_axis("town").to_csv(OUT / "degree_ranking.csv")
pairs = set(zip(df.from_town, df.a)) | set(zip(df.to_town, df.b))
pd.DataFrame(sorted(pairs), columns=["raw_name", "clean_name"]).to_csv(OUT / "name_mapping.csv", index=False)
p(f"raw spellings mapped: {len(pairs)} -> {len(set(c for _, c in pairs))} towns")

# per-town: raw log appearances vs distinct connections (shows why the group-by misleads)
appear = pd.concat([df.a, df.b]).value_counts()
cmp = pd.DataFrame({"log_appearances": appear, "distinct_connections": deg}).loc[deg.head(10).index]
p("\nTop 10: log appearances vs distinct connections\n" + cmp.to_string())

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = deg.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#c0392b" if i >= len(top) - 7 else "#7f8c8d" for i in range(len(top))]
    ax.barh([t.title() for t in top.index], top.values, color=colors)
    for y, v in enumerate(top.values):
        ax.text(v + 0.3, y, str(v), va="center", fontsize=8)
    ax.set_xlabel("distinct connections (degree)")
    ax.set_title(f"Cleaned network: 116 towns, 448 connections  ->  {flag}")
    fig.tight_layout()
    fig.savefig(OUT / "top_towns.png", dpi=150)
except ImportError:
    pass
(OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
(HERE / "FLAG.txt").write_text(flag + "\n", encoding="utf-8")
print("\n".join(lines))
