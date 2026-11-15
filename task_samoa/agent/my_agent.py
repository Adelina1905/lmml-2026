"""Dispatch on the champion's observable profile to a per-profile agent configuration."""
import importlib.util, os, json
_here = os.path.dirname(os.path.abspath(__file__))
def _load(name, params, tag):
    spec = importlib.util.spec_from_file_location(f"{name}_{tag}", os.path.join(_here, name + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); m.P.update(params); return m
CFG = json.loads(os.environ.get("MIX_CFG", json.dumps({"0": ["agent_s2", {"lam": 1.5}], "1": ["agent_v8", {"rush_ahead": 2, "rush_behind": -2, "intruder": 3.0, "reserve": 3}], "2": ["agent_s2", {"lam": 1.5}]})))
AG = {int(k): _load(v[0], v[1], k) for k, v in CFG.items()}
CUR = {"m": None}
def reset_agent():
    for m in AG.values(): m.reset_agent()
    CUR["m"] = None
def choose_action(s):
    m = AG[s["opponent_profile"]]
    if s["actions_remaining"] >= 255: m.reset_agent()
    return m.choose_action(s)
