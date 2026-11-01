"""Sanity: key-level diff of two approximation configs (which fields of which section/layer differ)."""
import json, sys, collections
a, b = (json.load(open(p)) for p in sys.argv[1:3])
def walk(x, y, path, out):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x: out.append((path + (k,), "only-in-B")); continue
            if k not in y: out.append((path + (k,), "only-in-A")); continue
            walk(x[k], y[k], path + (k,), out)
    elif x != y:
        out.append((path, "differs"))
out = []
walk(a, b, (), out)
agg = collections.defaultdict(list)
for p, kind in out:
    sec = p[0]
    layer = p[1] if len(p) > 1 else ""
    field = ".".join(map(str, p[2:])) if len(p) > 2 else ""
    agg[(sec, field, kind)].append(layer)
for (sec, field, kind), layers in sorted(agg.items()):
    ls = sorted(set(layers), key=lambda s: (len(s), s))
    print(f"{sec:10s} {field:28s} {kind:10s} {len(ls):2d} layers: {', '.join(l.replace('transformer.h.', 'h') for l in ls)[:150]}")
print("top-level sections A:", sorted(a), " B:", sorted(b))
