"""
Call My BLEUff - catch the panellists who write for BLEU instead of for the listener.

    python solve.py

Reads dataset/references.csv and dataset/panel.csv, writes everything to results/
and the flag to FLAG.txt. Pure Python + pandas + matplotlib.

Idea
----
BLEU only counts n-grams a description shares with the references. It has three
blind spots, and each kind of bluff walks through one of them:

  1. BLEU has no recall. Cutting a description short (dropping the venue's facts,
     even its name) costs nothing in precision.                -> CUT SHORT
  2. The brevity penalty is computed once for the whole deck, so a few short cards
     can be paid for by padding other cards with extra reference-like sentences
     ("It is family friendly."), restated facts or stuttered words. -> PADDED
  3. Matching is local: n-grams are counted wherever they are, so fragments of the
     references glued together in the wrong order still score.     -> WORD SALAD
     (salad loses facts and the venue's name, so check 1 catches most of it)

The LISTENER CHECK asks, for every card, what a listener would notice:
  A. coverage : does it name the venue and give the facts most references agree on?
  B. padding  : is a fact said twice, does it open with "It/They" before naming
                the venue, or does a word / phrase stutter ("more than more than")?
  C. cut off  : does it stop mid-phrase ("... rating of 3 out.") or is it a
                reference with its tail chopped off?
A card passes if it trips none of them. A deck's LISTENER SCORE is the share of
its 40 cards that pass. The honest panellist is the one with the highest score.
Every check is calibrated on the references themselves (honest by definition).
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bleu import corpus_bleu, segment_stats, tokenize

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
REPORT: list[str] = []


def say(*parts) -> None:
    line = " ".join(str(p) for p in parts)
    print(line)
    REPORT.append(line)


# =============================================================================
# 1. Facts a description can state about a venue (E2E-style venue cards)
# =============================================================================
ENTITIES = [
    "Alimentum", "Aromi", "Bibimbap House", "Blue Spice", "Browns Cambridge", "Clowns", "Cocum",
    "Cotto", "Fitzbillies", "Giraffe", "Green Man", "Loch Fyne", "Midsummer House", "Strada",
    "Taste of Cambridge", "Cambridge Blue", "Cricketers", "Dumpling Tree", "Eagle", "Golden Curry",
    "Golden Palace", "Mill", "Olive Grove", "Phoenix", "Plough", "Punter", "Rice Boat", "Twenty Two",
    "Vaults", "Waterman", "Wrestlers", "Travellers Rest Beefeater", "Wildwood", "Zizzi",
    "All Bar One", "Avalon", "Burger King", "Café Adriatic", "Café Brazil", "Café Rouge",
    "Café Sicilia", "Clare Hall", "Crowne Plaza Hotel", "Express by Holiday Inn",
    "Rainbow Vegetarian Café", "Raja Indian Cuisine", "Ranch", "Bakers", "Portland Arms",
    "Sorrento", "Yippee Noodle Bar",
]
ENT_RE = {e: re.compile(r"\b" + re.escape(e.lower()) + r"\b") for e in ENTITIES}
ENT_STRIP = re.compile("|".join(re.escape(e.lower()) for e in sorted(ENTITIES, key=len, reverse=True)))

SLOTS = {
    "food": {
        "chinese": r"\bchinese\b",
        "english": r"\benglish\b|\bbritish\b",
        "fast": r"\bfast[- ]?foods?\b|\bburgers?\b|\bfries\b|\bhamburger",
        "french": r"\bfrench\b",
        "indian": r"\bindian\b",
        "italian": r"\bitalian\b|\bpasta\b|\bpizza\b",
        "japanese": r"\bjapanese|\bsushi\b",
    },
    "area": {
        "riverside": r"\briver",
        "centre": r"\bcity cent(?:re|er)\b|\bcentre of\b|\bcenter of\b",
    },
    "price": {
        "cheap": r"\bcheap|less than (?:£|l)?\s?20|under (?:£|l)?\s?20|below (?:£|l)?\s?20|lower than (?:£)?\s?20"
                 r"|£?20 or less|\blow[- ]?(?:er )?pric|\blow[- ]cost|inexpensive|affordabl|\bbudget|low price range"
                 r"|price range (?:is )?low|prices? (?:are|is) low|\bvalue\b|\bdiscount",
        "moderate": r"moderate|20\s?-\s?(?:£)?25|20 to (?:£)?25|20 and (?:£)?25|\bmid[- ]?(?:range|pric|level)"
                    r"|medium[- ]pric|average[- ]pric|average price|prices? (?:are|is) average|reasonabl",
        "high": r"expensive|more than (?:£)?\s?30|over (?:£)?\s?30|above (?:£)?\s?30|£\s?30 (?:plus|and over)"
                r"|30 (?:pounds|euros)|\bhigh(?:ly)?[- ]?(?:er )?pric|high price range|price range (?:is )?high"
                r"|prices? (?:are|is) high|pricey|costly|high[- ]end|higher end|upscale|high cost",
    },
    "rating": {
        "low": r"\b1 (?:out )?of 5|\b1 out\b|one out of five|\blow(?:ly)?[- ]rat|low (?:customer|reviews?|ratings?)"
               r"|\bpoor|\b1[- ]star|one[- ]star|rated low|badly rated|\bbad (?:reviews?|ratings?)"
               r"|ratings? (?:are|is) low|rate[sd]? (?:it |the place )?(?:a )?1\b",
        "average": r"\b3 (?:out )?of 5|three out of five|average (?:customer )?ratings?|average[- ]rated|rated average"
                   r"|\b3[- ]star|three[- ]star|average customer|average reviews?|ratings? (?:are|is) average"
                   r"|moderate(?:ly)? rat|rated (?:in )?(?:the )?average",
        "high": r"\b5 (?:out )?of 5|five out of five|\bhigh(?:ly)?[- ]rat|high customer|five[- ]star|\b5[- ]star"
                r"|high ratings?|excellent (?:customer )?(?:rating|reviews?)|top[- ]rated|rated high"
                r"|ratings? (?:are|is) high|great (?:customer )?(?:rating|reviews?)|highly recommended",
    },
    "family": {
        "no": r"\bnot (?:a |very |so |too |really )?(?:family|kid|child)|\bnon[- ]?(?:family|kid|child)|adults?[- ]only"
              r"|isn't (?:a )?(?:family|kid|child)|no (?:kids|children|families)"
              r"|not (?:suitable|recommended|good|ideal|for) (?:for )?(?:the )?(?:children|kids|famil)"
              r"|n't (?:family|kid|child)|n't cater|leave the kids|no children",
        "yes": r"\b(?:famil(?:y|ies)|kids?|child(?:ren)?)\b",
    },
}
SLOT_RE = {s: {v: re.compile(p) for v, p in vals.items()} for s, vals in SLOTS.items()}
DANGLING = set("of out the a an and with by to in near at for is are its has this than called named".split())
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def entities(text: str) -> set[str]:
    t = text.lower()
    return {e for e, rx in ENT_RE.items() if rx.search(t)}


def slot_values(text: str) -> dict[str, set[str]]:
    t = ENT_STRIP.sub(" <ent> ", text.lower())   # 'Raja Indian Cuisine' is a place, not food
    out = {}
    for slot, vals in SLOT_RE.items():
        hit = {v for v, rx in vals.items() if rx.search(t)}
        if slot == "family" and "no" in hit:
            hit.discard("yes")
        out[slot] = hit
    return out


def facts_stated(text: str) -> set[tuple]:
    f = {("venue", e) for e in entities(text)}
    for slot, vals in slot_values(text).items():
        f |= {(slot, v) for v in vals}
    return f


def consensus(refs: list[str], agree: int) -> tuple[set[tuple], str | None]:
    """Facts at least `agree` references state, and the venue's own name."""
    counts = Counter(f for r in refs for f in facts_stated(r))
    facts = {f for f, c in counts.items() if c >= agree}
    ent_count, first = Counter(), Counter()
    for r in refs:
        es = entities(r)
        ent_count.update(es)
        if es:
            first[min(es, key=lambda e: r.lower().find(e.lower()))] += 1
    name = None
    if ent_count:
        top = max(ent_count.values())
        name = max((e for e in ent_count if ent_count[e] == top), key=lambda e: first[e])
    return facts, name


def sentences(text: str) -> list[str]:
    return [s for s in SENT_SPLIT.split(text.strip()) if s]


# =============================================================================
# 2. The listener check for one card
# =============================================================================
def listener_check(text: str, refs: list[str], agree: int = 3, min_recall: float = 0.7) -> dict:
    tok = tokenize(text)
    ref_tok = [tokenize(r) for r in refs]
    facts, name = consensus(refs, agree)
    said = facts_stated(text)
    recall = len(facts & said) / len(facts) if facts else 1.0
    no_name = name is not None and name not in entities(text)

    # padding
    per_sentence = Counter()
    for s in sentences(text):
        for f in facts_stated(s):
            if f[0] != "venue":
                per_sentence[f] += 1
    repeated_fact = any(c >= 2 for c in per_sentence.values())
    pronoun_opening = bool(re.match(r"\s*(it|they|its|it's|their)\b", text, re.I))
    stutter = any(tok[i:i + n] == tok[i + n:i + 2 * n]
                  for n in (1, 2, 3, 4) for i in range(len(tok) - 2 * n + 1) if tok[i].isalpha())

    # cut off
    body = tok[:-1] if tok and tok[-1] in ".!?" else tok
    chopped_reference = any(3 <= len(body) < len(r) and r[:len(body)] == body for r in ref_tok)
    dangling_end = bool(body) and body[-1] in DANGLING

    a = recall < min_recall or no_name
    b = repeated_fact or pronoun_opening or stutter
    c = chopped_reference or dangling_end
    return {
        "recall": recall, "no_name": no_name, "repeated_fact": repeated_fact,
        "pronoun_opening": pronoun_opening, "stutter": stutter,
        "chopped_reference": chopped_reference, "dangling_end": dangling_end,
        "A_coverage": a, "B_padding": b, "C_cut_off": c, "passes": not (a or b or c),
    }


def decode(seats: list[int]) -> str:
    word = ""
    for i in range(0, len(seats), 3):
        d1, d2, d3 = (s - 1 for s in seats[i:i + 3])
        word += chr(ord("A") + 9 * d1 + 3 * d2 + d3)
    return word


# =============================================================================
# 3. Run
# =============================================================================
def main() -> None:
    refs = pd.read_csv(DATA / "references.csv")
    panel = pd.read_csv(DATA / "panel.csv")
    ref_map = refs.groupby(["round", "card"]).reference.apply(list).to_dict()
    say("references:", refs.shape, " panel:", panel.shape)

    # --- house scorer ---------------------------------------------------------
    rows = []
    for (rnd, seat), deck in panel.groupby(["round", "seat"]):
        res = corpus_bleu(list(deck.text), [ref_map[(rnd, c)] for c in deck.card])
        rows.append({"round": rnd, "seat": seat, "bleu": res["bleu"], "brevity_penalty": res["brevity_penalty"],
                     "p1": res["precisions"][0], "p4": res["precisions"][3],
                     "output_length": res["output_length"], "reference_length": res["reference_length"]})
    house = pd.DataFrame(rows)
    bleu_seats = list(house.loc[house.groupby("round").bleu.idxmax()].sort_values("round").seat)
    say("\nHouse scorer (BLEU) crowns:", bleu_seats, "->", "SIGMOID_" + decode(bleu_seats), "(not the flag)")

    # --- calibration: how often does an honest text trip each check? -----------
    cal = []
    for (rnd, card), R in ref_map.items():
        for i, t in enumerate(R):
            cal.append(listener_check(t, R[:i] + R[i + 1:]))
    cal = pd.DataFrame(cal)
    cols = ["A_coverage", "B_padding", "C_cut_off", "no_name", "repeated_fact", "pronoun_opening",
            "stutter", "chopped_reference", "dangling_end"]
    say("\nCalibration on the 4,800 references (each tested against the other four of its card):")
    for c in cols:
        say(f"  {c:18s} trips on {100 * cal[c].mean():5.2f}% of references")
    say(f"  any check          trips on {100 * (~cal.passes).mean():5.2f}% of references"
        "  (the references are raw crowd-sourced texts and many of them leave facts out)")

    # --- every panel card -------------------------------------------------------
    cards = []
    for _, row in panel.iterrows():
        R = ref_map[(row["round"], row.card)]
        d = listener_check(row.text, R)
        st = segment_stats(row.text, R)
        d.update({"round": row["round"], "card": row.card, "seat": row.seat, "text": row.text,
                  "out_len": st["out_len"], "ref_len": st["ref_len"],
                  "p1": st["matched"][0] / max(1, st["total"][0]),
                  "p4": st["matched"][3] / max(1, st["total"][3])})
        cards.append(d)
    cards = pd.DataFrame(cards)
    cards.to_csv(OUT / "card_checks.csv", index=False)

    decks = (cards.groupby(["round", "seat"])
             .agg(listener_score=("passes", "mean"), flagged=("passes", lambda x: int((~x).sum())),
                  A_coverage=("A_coverage", "sum"), B_padding=("B_padding", "sum"), C_cut_off=("C_cut_off", "sum"),
                  mean_recall=("recall", "mean"))
             .reset_index().merge(house, on=["round", "seat"]))
    decks.to_csv(OUT / "deck_scores.csv", index=False)

    say("\nListener score per deck (share of the 40 cards that pass all checks):")
    piv = decks.pivot(index="round", columns="seat", values="listener_score")
    flg = decks.pivot(index="round", columns="seat", values="flagged")
    blu = decks.pivot(index="round", columns="seat", values="bleu")
    honest, lines = [], []
    say(" round | listener score s1 s2 s3 | flagged cards s1 s2 s3 | BLEU s1 s2 s3 | honest | margin (cards)")
    for rnd in piv.index:
        best = int(piv.loc[rnd].idxmax())
        f = sorted(flg.loc[rnd])
        honest.append(best)
        say(f"  {rnd:3d}  |  {piv.loc[rnd,1]:.3f} {piv.loc[rnd,2]:.3f} {piv.loc[rnd,3]:.3f}  |  "
            f"{flg.loc[rnd,1]:3d} {flg.loc[rnd,2]:3d} {flg.loc[rnd,3]:3d}  |  "
            f"{blu.loc[rnd,1]:.3f} {blu.loc[rnd,2]:.3f} {blu.loc[rnd,3]:.3f}  |  seat {best} | {f[1] - f[0]}")
    word = decode(honest)
    hon = decks[decks.seat == decks["round"].map(dict(zip(piv.index, honest)))]
    blf = decks.drop(hon.index)
    say(f"\nHonest decks: listener score {hon.listener_score.min():.3f}-{hon.listener_score.max():.3f}, "
        f"flagged cards {hon.flagged.min()}-{hon.flagged.max()}")
    say(f"Bluffing decks: listener score {blf.listener_score.min():.3f}-{blf.listener_score.max():.3f}, "
        f"flagged cards {blf.flagged.min()}-{blf.flagged.max()}")
    say(f"Smallest gap between the honest deck and the next deck: {min(sorted(flg.loc[r])[1] - sorted(flg.loc[r])[0] for r in flg.index)} cards")
    say("Honest seats:", honest)
    say("Base-3 digits:", [s - 1 for s in honest])
    say("FLAG: SIGMOID_" + word)
    (HERE / "FLAG.txt").write_text("SIGMOID_" + word + "\n")

    # --- robustness: different thresholds give the same honest seats ------------
    say("\nRobustness (agree = how many references must state a fact; min_recall = coverage threshold):")
    for agree in (2, 3, 4):
        for mr in (0.6, 0.7, 0.8):
            ok = []
            for rnd in range(1, 25):
                sub = panel[panel["round"] == rnd]
                sc = {s: sum(listener_check(t, ref_map[(rnd, c)], agree, mr)["passes"]
                             for t, c in zip(sub[sub.seat == s].text, sub[sub.seat == s].card)) for s in (1, 2, 3)}
                ok.append(max(sc, key=sc.get))
            say(f"  agree={agree} min_recall={mr}: SIGMOID_{decode(ok)}  {'same' if ok == honest else 'DIFFERENT'}")

    # --- how the bluffs game BLEU: evidence -------------------------------------
    cards["deck"] = cards["round"].map(dict(zip(piv.index, honest)))
    cards["who"] = (cards.seat == cards.deck).map({True: "honest panellist", False: "bluffer"})

    def kind(r):
        if r.C_cut_off or (r.A_coverage and r.out_len < r.ref_len):
            return "cut short / dropped facts"
        if r.B_padding:
            return "padded / repeated"
        if r.A_coverage:
            return "lost facts (word salad)"
        return "passes"
    cards["kind"] = cards.apply(kind, axis=1)
    cards.loc[cards.who == "honest panellist", "kind"] = cards.loc[cards.who == "honest panellist", "kind"].where(
        cards.loc[cards.who == "honest panellist", "kind"] == "passes", "flagged (honest)")
    ev = (cards.groupby(["who", "kind"])
          .agg(cards=("text", "size"), mean_fact_recall=("recall", "mean"),
               length_vs_closest_ref=("out_len", "sum"), ref=("ref_len", "sum"), unigram_precision=("p1", "mean"),
               fourgram_precision=("p4", "mean")).reset_index())
    ev["length_vs_closest_ref"] = ev.length_vs_closest_ref / ev.ref
    ev = ev.drop(columns="ref")
    ev.round(3).to_csv(OUT / "bluff_mechanisms.csv", index=False)
    say("\nHow the flagged cards play BLEU (segment statistics from bleu.py):")
    say(ev.round(3).to_string(index=False))

    # the corpus-level brevity penalty hides the short cards
    agree_bleu = sum(int(a == b) for a, b in zip(bleu_seats, honest))
    lowest = sum(int(hon.set_index("round").bleu[r] == blu.loc[r].min()) for r in range(1, 25))
    say(f"\nBLEU crowned the honest panellist in only {agree_bleu} of 24 rounds; in {lowest} rounds the honest "
        "deck has the LOWEST BLEU of the three.")
    n_cut = int((cards.kind == "cut short / dropped facts").sum())
    say(f"\nBrevity penalty: the {n_cut} cut-short cards are ~30% shorter than their closest reference, yet the median "
        f"brevity penalty of a bluffing deck is {blf.brevity_penalty.median():.3f} (honest decks: "
        f"{hon.brevity_penalty.median():.3f}; 1.0 = no penalty). BLEU sums lengths over the whole deck before "
        "comparing, so the padded cards (~20% longer) and the untouched cards pay for the short ones.")

    # examples
    say("\nExamples of each check firing:")
    for col, label in [("chopped_reference", "chopped reference"), ("dangling_end", "dangling end"),
                       ("no_name", "venue never named"), ("pronoun_opening", "opens with It/They"),
                       ("repeated_fact", "fact said twice"), ("stutter", "stutter")]:
        ex = cards[(cards.who == "bluffer") & cards[col]].head(2)
        for _, r in ex.iterrows():
            say(f"  [{label}] r{r['round']} seat {r.seat} card {r.card}: {r.text}")
    ex = cards[(cards.who == "bluffer") & cards.A_coverage & ~cards.no_name & (cards.recall < 0.5)].head(3)
    for _, r in ex.iterrows():
        say(f"  [low coverage {r.recall:.2f}] r{r['round']} seat {r.seat} card {r.card}: {r.text}")

    # --- figures ------------------------------------------------------------------
    colors = {1: "#3b6fb6", 2: "#d9822b", 3: "#4c9a5b"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for seat in (1, 2, 3):
        g = decks[decks.seat == seat]
        axes[0].scatter(g["round"], g.bleu, s=32, color=colors[seat], label=f"seat {seat}")
        axes[1].scatter(g["round"], g.listener_score, s=32, color=colors[seat], label=f"seat {seat}")
    for ax in axes:
        for rnd, s in zip(range(1, 25), honest):
            ax.axvspan(rnd - 0.4, rnd + 0.4, color="#eeeeee", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    hb = decks.merge(pd.DataFrame({"round": range(1, 25), "seat": honest}))
    axes[0].scatter(hb["round"], hb.bleu, s=140, facecolors="none", edgecolors="black", label="honest (listener check)")
    axes[1].scatter(hb["round"], hb.listener_score, s=140, facecolors="none", edgecolors="black",
                    label="honest (listener check)")
    axes[0].set_ylabel("deck BLEU (house scorer)")
    axes[0].set_title("BLEU does not find the honest panellist ...")
    axes[1].set_ylabel("listener score\n(share of cards passing)")
    axes[1].set_title("... the listener check puts one deck clear of the other two in every round")
    axes[1].set_xlabel("round")
    axes[1].set_xticks(range(1, 25))
    axes[1].legend(frameon=True, ncol=2, loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "bleu_vs_listener.png", dpi=130)

    fig, ax = plt.subplots(figsize=(13, 4.2))
    x, labels = 0, []
    for rnd in range(1, 25):
        for seat in (1, 2, 3):
            r = decks[(decks["round"] == rnd) & (decks.seat == seat)].iloc[0]
            bottom = 0
            for col, c in [("A_coverage", "#c0504d"), ("B_padding", "#8064a2"), ("C_cut_off", "#f2a541")]:
                ax.bar(x, r[col], bottom=bottom, color=c, width=0.8, label=col if x == 0 else None)
                bottom += r[col]
            if seat == honest[rnd - 1]:
                ax.text(x, bottom + 0.5, "H", ha="center", fontsize=7, fontweight="bold")
            x += 1
        x += 1
    ax.set_xticks([4 * i + 1 for i in range(24)])
    ax.set_xticklabels(range(1, 25))
    ax.set_xlabel("round (three bars = seats 1, 2, 3; H = honest)")
    ax.set_ylabel("cards tripping each check\n(a card can trip several)")
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "flags_per_deck.png", dpi=130)

    say("\nfigures: results/bleu_vs_listener.png, results/flags_per_deck.png")
    (OUT / "report.txt").write_text("\n".join(REPORT) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
