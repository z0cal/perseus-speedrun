import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_trace import depth
for d in list(range(40, 66, 2)) + list(range(84, 98, 2)) + list(range(110, 130, 2)) + list(range(230, 256, 2)):
    c = [1.0 if (i % 2 == 0 or i == 1) else 0.0 for i in range(d + 1)]
    print("even+x deg", d, "depth/ctmul/scmul", depth(c))
for d in range(5, 32):
    print("dense deg", d, depth([1.0] * (d + 1)))
