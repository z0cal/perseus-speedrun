"""Route mix of a Decode-8 plan's compiled refreshes, and its deliberate refreshes against the record's.

Compiles the 14 stages chained (as PlannedRun.prepare: a graph dir with blocks 0-12 and stage 13) and counts the
refresh sites per route (0 = dense, 512 = SparseB, 1 = SPRU) and kind; then compares every deliberate refresh's
route with the record's (the code's own refreshes, e.g. the K/V push: a plan may not move one onto another route).
compile_check.py's 0 errors does not catch that: the planner and the compiler read the same remapped landing.

    python route_mix.py GRAPH_DIR PLAN_DIR          (from work/perseus, PYTHONPATH=python)"""
import collections
import sys

from perseus import _pc

REC = ("../graphs_combo/rec", "bootstrap_placements/gpt2_decode_python_n32_kc16_cm")
BLOCKS = [f"block_{b}" for b in range(14)]


def sites(g, p):
    ps = _pc.Program.compile(g, p, blocks=BLOCKS, keep_going=True)
    return {b: ps[b].bts_sites for b in BLOCKS}


def main():
    new, rec = sites(*sys.argv[1:3]), sites(*REC)
    for name, s in (("record", rec), ("plan", new)):
        mix = collections.Counter((x[2], x[1]) for b in BLOCKS[:13] for x in s[b])
        by_route = collections.Counter(x[2] for b in BLOCKS[:13] for x in s[b])
        print(f"{name}: blocks 0-12 {sum(by_route.values())} refreshes, per route {dict(sorted(by_route.items()))}, "
              f"(route, kind) {dict(sorted(mix.items()))}; stage 13 {len(s['block_13'])}")
    bad = 0
    for b in BLOCKS:
        r = {x[0]: x for x in rec[b] if x[1] == "deliberate"}
        for x in (x for x in new[b] if x[1] == "deliberate"):
            if x[0] in r and (x[2], x[4]) != (r[x[0]][2], r[x[0]][4]):
                bad += 1
                print(f"  {b} {x[0]}: deliberate on route {x[2]} (real {x[4]}), the record's on {r[x[0]][2]} (real {r[x[0]][4]})")
    print(f"deliberate refreshes off the record's route: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
