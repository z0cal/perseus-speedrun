import sys
from perseus import _pc
from perseus.serve import gpt2_decode as served
plan, A, B = sys.argv[1:4]
cfg = served._config(plan, "n32")
blocks = [f"block_{b}" for b in range(14)]
pa = _pc.Program.compile(A, plan, config=cfg, blocks=blocks, keep_going=True)
pb = _pc.Program.compile(B, plan, config=cfg, blocks=blocks, keep_going=True)
for lab in blocks:
    ra, rb = list(pa[lab].pt_refs), list(pb[lab].pt_refs)
    d = [(i, x, y) for i, (x, y) in enumerate(zip(ra, rb)) if tuple(x) != tuple(y)]
    sa, sb = list(pa[lab].bts_site_levels), list(pb[lab].bts_site_levels)
    ds = [(x, y) for x, y in zip(sa, sb) if tuple(x) != tuple(y)]
    if d or ds or len(ra) != len(rb):
        print(lab, f"pt_refs {len(ra)} vs {len(rb)}, {len(d)} differ:", d[:12], "| site levels differ:", ds)
