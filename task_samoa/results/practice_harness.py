"""Parallel practice harness: plays agent vs champion on practice groups/seeds."""
import sys, json, importlib.util, time, collections, multiprocessing as mp
ENGINE = "/home/claude/work/engine"
sys.path.insert(0, ENGINE)

def load(path):
    spec = importlib.util.spec_from_file_location("ag_"+str(abs(hash(path))), path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def worker(args):
    agent_path, jobs = args
    from client import ArenaClient
    m = load(agent_path)
    out = []
    with ArenaClient() as env:
        for (arena, seed) in jobs:
            if hasattr(m, "reset_agent"): m.reset_agent()
            r = env.reset(arena, seed); seat = r["observation"]["seat"]
            t=time.time()
            while not r["done"]:
                a = m.choose_action(r["observation"])
                assert a in r["observation"]["legal_actions"], (a, r["observation"]["legal_actions"])
                r = env.step(a)
            w = r["winner"]
            out.append((arena, seed, seat, r["scores"][seat], r["scores"][1-seat],
                        1 if w == seat else (0 if w is None else -1), time.time()-t))
    return out

def run(agent_path, groups=range(32), seeds=range(4), base=100000, procs=2):
    jobs = [(a, base + s*37 + a) for a in groups for s in seeds]
    chunks = [jobs[i::procs*4] for i in range(procs*4)]
    with mp.Pool(procs) as p:
        res = [x for part in p.map(worker, [(agent_path, c) for c in chunks]) for x in part]
    return res

def report(res):
    n=len(res); w=sum(r[5]==1 for r in res); d=sum(r[5]==0 for r in res)
    print(f"games {n} wins {w} ({w/n:.1%}) draws {d}  avg margin {sum(r[3]-r[4] for r in res)/n:.1f}  avg time/game {sum(r[6] for r in res)/n:.2f}s")
    by=collections.defaultdict(list)
    for r in res: by[r[0]].append(r)
    print(" ".join(f"{a}:{sum(x[5]==1 for x in v)}/{len(v)}" for a,v in sorted(by.items())))
    for key,name in [(lambda r:r[2],"seat"),(lambda r:(r[0]//2)%3,"profile")]:
        g=collections.defaultdict(list)
        for r in res: g[key(r)].append(r)
        print(name, {k:f"{sum(x[5]==1 for x in v)}/{len(v)} m={sum(x[3]-x[4] for x in v)/len(v):.0f}" for k,v in sorted(g.items())})

if __name__ == "__main__":
    a = sys.argv[1]; s = int(sys.argv[2]) if len(sys.argv)>2 else 4
    base = int(sys.argv[3]) if len(sys.argv)>3 else 100000
    t=time.time(); res = run(a, seeds=range(s), base=base); report(res); print(f"wall {time.time()-t:.0f}s")
    json.dump(res, open(a.replace('.py','')+f"_res_{base}.json","w"))
