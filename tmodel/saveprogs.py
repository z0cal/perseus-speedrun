import sys, os
from perseus import _pc
B = [f"block_{b}" for b in range(14)]
PL = {
 "record": ("../../subagent/r4/graphs_combo/r4", "bootstrap_placements/gpt2_decode_python_n32a4_kc16r_cm"),
 "em9": ("../lndefer/graphs_combo/lnd_e3", "bootstrap_placements/lnd_fin_e3_em9_cm"),
 "d27sb": ("../lndefer/graphs_combo/lnd_e3", "bootstrap_placements/lnd_fin_e3_d27sb_cm"),
}
out = sys.argv[1]
for name in sys.argv[2:] or PL:
    g, p = PL[name]
    ps = _pc.Program.compile(g, p, blocks=B, keep_going=True)
    os.makedirs(f"{out}/{name}", exist_ok=True)
    for b in B:
        ps[b].save(f"{out}/{name}/{b}.json")
    print(name, "saved")
