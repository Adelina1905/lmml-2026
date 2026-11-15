"""Reproduce the minimum-token routing using the supplied, unchanged simulator."""
import itertools
import json
from pathlib import Path
import model_utils as mu

def main():
    bank = mu.load_fact_bank()
    config = {}
    for qid, question in bank.items():
        candidates = []
        for tier, style, fewshot, context in itertools.product(mu.TIERS, mu.STYLES, (False, True), (False, True)):
            choice = dict(tier=tier, style=style, fewshot=fewshot, context=context)
            call = mu.build_call(question, **choice)
            if(mu.question_passes(question["slots"], call["answer"])): candidates.append((call["tokens"], choice))
        if(not candidates): raise RuntimeError(f"No passing configuration for {qid}")
        config[qid] = min(candidates, key=lambda candidate: candidate[0])[1]
    result = mu.run_config(config, bank)
    if(not result["quality_gate"] or not result["budget_gate"]): raise RuntimeError("Optimized routing fails a grading gate")
    path = Path(__file__).parent / "routing_config.json"
    path.write_text("{\n" + ",\n".join("  " + json.dumps(qid) + ": " + json.dumps(choice) for qid, choice in config.items()) + "\n}\n", encoding="utf-8")
    saved_result = mu.run_config(mu.load_json(path), bank)
    if(saved_result != result): raise RuntimeError("Saved configuration changed the result")
    print(f"Wrote {path.name}; evaluated {len(bank) * 16} configurations")
    print(f"Quality: {sum(info['pass'] for info in result['per_question'].values())}/{len(bank)} -> PASS")
    print(f"Tokens: {result['total_tokens']} / {mu.TOKEN_BUDGET} -> PASS")

if(__name__ == "__main__"): main()
