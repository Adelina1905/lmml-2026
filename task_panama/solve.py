#!/usr/bin/env python3
"""CanalQA audit: find benchmark items that leaked into the fine-tuning corpus,
exclude them, and recompute the assistant's honest accuracy.

Run from inside this folder:
    python solve.py            # writes report.json and results/
    python open_seal.py report.json

Standard library only.
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "canalqa_audit.json"
RESULTS = HERE / "results"

# --------------------------------------------------------------------------
# 1. Normalisation helpers
# --------------------------------------------------------------------------

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm_entity(s: str) -> str:
    s = strip_accents(s).lower().strip().rstrip(".")
    return re.sub(r"\s+", " ", s)


# Words with no meaning for matching an attribute or an answer.
STOP = {"the", "of", "a", "an", "el", "la", "los", "las", "de", "del", "canal"}

# Spelled-out numbers seen in the corpus (English and Spanish).
NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "uno": "1", "una": "1", "dos": "2", "tres": "3", "cuatro": "4", "cinco": "5",
    "seis": "6", "siete": "7", "ocho": "8", "nueve": "9", "diez": "10",
    "nineteen ninety-nine": "1999",
}

# Unit abbreviations / synonyms -> one canonical token.
UNIT_SYNONYMS = {
    "m": "meter", "meters": "meter", "metre": "meter", "metres": "meter", "metros": "meter",
    "km": "kilometer", "kilometers": "kilometer", "kilometros": "kilometer",
    "sq": "square", "cuadrados": "square",
}

# Attribute words that carry no extra meaning ("overall length" == "length",
# "count of chambers" == "number of chambers").
ATTR_SYNONYMS = {"overall": None, "count": "number", "lock": None}


def singular(tok: str) -> str:
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def translate_tokens(text: str, glossary: dict) -> list:
    """Lower-case, strip accents, translate Spanish words with the lab glossary."""
    gl = {strip_accents(k).lower(): v.lower() for k, v in glossary.items()}
    out = []
    for raw in re.findall(r"[\w\-\.,]+", text.lower()):
        tok = raw.strip(".,")
        key = strip_accents(tok)
        if key in gl:
            out.extend(gl[key].split())
        else:
            out.append(key)
    return out


def norm_answer(ans: str, glossary: dict) -> frozenset:
    s = strip_accents(ans).lower().strip()
    for w, n in NUMBER_WORDS.items():
        s = re.sub(rf"\b{re.escape(w)}\b", n, s)
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)          # Spanish decimal comma
    toks = []
    for t in translate_tokens(s, glossary):
        t = UNIT_SYNONYMS.get(t, t)
        if t in STOP:
            continue
        toks.append(singular(UNIT_SYNONYMS.get(singular(t), singular(t))))
    return frozenset(toks)


def norm_attr(attr: str, glossary: dict) -> frozenset:
    toks = []
    for t in translate_tokens(attr, glossary):
        if t in STOP:
            continue
        t = singular(t)
        t = ATTR_SYNONYMS.get(t, t)
        if t:
            toks.append(t)
    return frozenset(toks)


# --------------------------------------------------------------------------
# 2. Parse the benchmark questions and the leaked Q&A records in the corpus
# --------------------------------------------------------------------------

# Benchmark question: "... the <attribute> of <entity>?" / "."
BENCH_Q = re.compile(r"\bthe (?P<attr>[\w\s\-]+?) of (?P<ent>.+?)[\?\.]$", re.I)

# Corpus templates holding a question + an answer.
EN_A = re.compile(r"the entry asks: '(?P<q>.+?)' The logged response reads '(?P<a>.+?)' for (?P<ent>.+?)\.$")
EN_B = re.compile(r"Question on record: '(?P<q>.+?)' Answer on record: '(?P<a>.+?)'\. Verified against the (?P<ent>.+?) registry entry\.$")
ES_C = re.compile(r"Registro del canal LOG-\d+\. (?P<q>.+?) Respuesta registrada: '(?P<a>.+?)'\. Verificado en la entrada de (?P<ent_es>.+?) \((?P<ent>.+?)\)\.$")

# Attribute inside a Spanish question: "¿Cuál es <attr> de <E>?" / "Indique <attr> de <E>."
ES_ATTR = re.compile(r"(?:¿cu[aá]l es|indique)\s+(?P<attr>.+?)\s+de[l]?\s+(?P<ent_es>.+?)[\?\.]$", re.I)


def parse_record(doc: dict):
    text = doc["text"]
    for pat, lang in ((EN_A, "en"), (EN_B, "en"), (ES_C, "es")):
        m = pat.search(text)
        if not m:
            continue
        q, a, ent = m.group("q"), m.group("a"), m.group("ent")
        if lang == "en":
            mq = BENCH_Q.search(q.strip())
            attr = mq.group("attr") if mq else q
        else:
            q2 = re.sub(r"^seg[uú]n el registro del canal,\s*", "", q.strip(), flags=re.I)
            # anchor on the Spanish entity name given by the template, so that
            # "el año de finalización de Represa de Gatún" keeps the whole attribute
            ent_es = re.escape(m.group("ent_es"))
            mq = re.search(rf"(?:¿cu[aá]l es|indique)\s+(?P<attr>.+?)\s+de\s+{ent_es}[\?\.]$", q2, re.I)
            attr = mq.group("attr") if mq else q2
        return {"doc_id": doc["doc_id"], "lang": doc["lang"], "question": q, "answer": a,
                "entity": ent, "attr": attr}
    return None


def main():
    d = json.loads(BUNDLE.read_text(encoding="utf-8"))
    gl = d["glossary_es_en"]
    bench = d["benchmark"]
    preds = {p["id"]: p for p in d["model_predictions"]}
    RESULTS.mkdir(exist_ok=True)
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    # sanity: reported accuracy reproduces from model_predictions
    n_correct = sum(p["correct"] for p in preds.values())
    out(f"Benchmark items: {len(bench)} | correct: {n_correct} | "
        f"recomputed accuracy {n_correct/len(bench):.4f} (reported {d['reported_accuracy']})")

    # --- leaked records ---------------------------------------------------
    records, unparsed_qa = [], []
    for doc in d["corpus"]:
        r = parse_record(doc)
        if r:
            records.append(r)
        elif re.search(r"\?|respuesta|answer|response", doc["text"], re.I):
            unparsed_qa.append(doc["doc_id"])
    out(f"Corpus docs: {len(d['corpus'])} | Q&A-style records parsed: {len(records)} "
        f"({Counter(r['lang'] for r in records)}) | unparsed Q&A-like docs: {unparsed_qa}")

    # --- match every benchmark item against every record ------------------
    rows = []
    for b in bench:
        mb = BENCH_Q.search(b["question"].strip())
        b_ent, b_attr = norm_entity(mb.group("ent")), norm_attr(mb.group("attr"), gl)
        b_ans = norm_answer(b["gold"], gl)
        hits = {"exact": [], "variant": [], "same_entity_other_attr": [], "same_attr_other_entity": []}
        for r in records:
            same_ent = norm_entity(r["entity"]) == b_ent
            same_attr = norm_attr(r["attr"], gl) == b_attr
            same_ans = norm_answer(r["answer"], gl) == b_ans
            if same_ent and same_attr:
                hits["exact" if same_ans else "variant"].append(r)
            elif same_ent:
                hits["same_entity_other_attr"].append(r)
            elif same_attr:
                hits["same_attr_other_entity"].append(r)
        rows.append((b, preds[b["id"]], hits))

    def fmt(r):
        return f"{r['doc_id']}[{r['lang']}] '{r['question']}' -> '{r['answer']}'"

    out("\n=== A. Exact leaks: same entity + same attribute + same answer (after normalising units/language) ===")
    exact = [x for x in rows if x[2]["exact"]]
    for b, p, h in exact:
        out(f"{b['id']} gold='{b['gold']}' pred='{p['pred']}' correct={p['correct']}  <= {fmt(h['exact'][0])}")

    out("\n=== B. Borderline: same entity + same attribute, answer worded differently from gold ===")
    variant = [x for x in rows if x[2]["variant"] and not x[2]["exact"]]
    for b, p, h in variant:
        out(f"{b['id']} Q='{b['question']}' gold='{b['gold']}' pred='{p['pred']}' correct={p['correct']}")
        for r in h["variant"]:
            out(f"      <= {fmt(r)}")

    out("\n=== C. Decoys (NOT leaks): record about same entity but a different attribute ===")
    for b, p, h in rows:
        if not h["exact"] and not h["variant"] and h["same_entity_other_attr"]:
            out(f"{b['id']} '{b['question']}' gold='{b['gold']}' | records: " +
                "; ".join(f"{r['doc_id']} {r['attr']}='{r['answer']}'" for r in h["same_entity_other_attr"]))

    out("\n=== D. Decoys (NOT leaks): same attribute but a different entity ===")
    for b, p, h in rows:
        if not h["exact"] and not h["variant"] and h["same_attr_other_entity"]:
            out(f"{b['id']} '{b['question']}' | records: " +
                "; ".join(f"{r['doc_id']} {r['entity']}" for r in h["same_attr_other_entity"]))

    # records that answer no benchmark question
    used = {r["doc_id"] for _, _, h in rows for r in h["exact"] + h["variant"]}
    out(f"\nQ&A records that match no benchmark item (other canal facts): "
        f"{len(records) - len(used)} of {len(records)}")

    # --- decision ------------------------------------------------------------
    # Borderline items are EXCLUDED: the corpus holds the very same question about
    # the same entity, with the same fact in a different wording
    # ('Atlantic terminus' ~ 'Atlantic terminal', 'Balboa district' ~ 'Balboa township').
    # The model was trained on the question->answer pair, so its correct answer
    # cannot be credited to it.
    excluded = sorted({b["id"] for b, _, _ in exact} | {b["id"] for b, _, _ in variant})
    remaining = [(b, p) for b, p, _ in rows if b["id"] not in excluded]
    rem_correct = sum(p["correct"] for _, p in remaining)
    honest = rem_correct / len(remaining)

    exc_correct = sum(preds[i]["correct"] for i in excluded)
    out("\n=== Result ===")
    out(f"Excluded items: {len(excluded)} (exact {len(exact)} + borderline {len(variant)}); "
        f"model correct on {exc_correct}/{len(excluded)} of them = {exc_correct/len(excluded):.2%}")
    out(f"Remaining items: {len(remaining)}; correct {rem_correct}; "
        f"honest accuracy = {rem_correct}/{len(remaining)} = {honest:.4f} ({honest:.2%})")
    out(f"Reported 73.75% -> honest {honest:.2%} (inflation {0.7375 - honest:+.2%} points)")
    out(f"Excluded ids: {excluded}")

    report = {"excluded_ids": excluded, "honest_accuracy": round(honest, 4)}
    (HERE / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESULTS / "audit_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # per-item table for the write-up
    with open(RESULTS / "items.csv", "w", encoding="utf-8") as f:
        f.write("id,status,correct,gold,pred,leak_doc\n")
        for b, p, h in rows:
            st = "excluded_exact" if h["exact"] else ("excluded_borderline" if h["variant"] else "kept")
            src = (h["exact"] or h["variant"] or [{"doc_id": ""}])[0]["doc_id"]
            f.write(f"{b['id']},{st},{p['correct']},\"{b['gold']}\",\"{p['pred']}\",{src}\n")
    print("\nwrote report.json, results/audit_report.txt, results/items.csv")


if __name__ == "__main__":
    main()
