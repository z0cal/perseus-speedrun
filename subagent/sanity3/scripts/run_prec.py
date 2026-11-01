import json, math, sys
import numpy as np
sys.path.insert(0, '.')
from emchain import *
R = '/home/user/perseus-speedrun/fits/'
SHIP16 = [-5.73829476916553172e-01, 2.63718536771466207e-02, -9.15574236933522245e-01, -3.08975417541565676e-02,
          2.85601052580403802e-01, 4.83129148738224799e-03, -2.74350673185783482e-02, -3.16919293037672828e-04,
          1.31295181914509542e-03, 1.15825536926191591e-05, -3.79010152666429525e-05, -2.71035793019274615e-07,
          7.33952552563662122e-07, 4.41686895092293209e-09, -1.03169742989480305e-08]
SHIP24 = [-5.01431878575762102e-01, -2.60978240793176830e-02, -2.75969415548868335e-01, -3.76057943042461193e-02,
    6.98674483745098551e-01, 2.06638885145157200e-02, -1.93915485831718332e-01, -3.59503488102385325e-03,
    2.34902992390395096e-02, 3.23162556614403976e-04, -1.63628724411827371e-03, -1.80046855470203313e-05,
    7.47055232583680419e-05, 6.86710430736672347e-07, -2.41801471221991641e-06, -1.91108603715104241e-08,
    5.85030407613337840e-08, 4.05831352269635484e-10, -1.09946346874119116e-09, -6.79891244700694627e-12,
    1.65253168377321969e-11]
cfgs = [
    ('ship K16 d14 r5', SHIP16, 5, 16, False),
    ('d27 K16 r4', json.load(open(R + 'full_d27_r4.json')), 4, 16, False),
    ('even z11 K16 r4', json.load(open(R + 'even_k16_z11_r4.json')), 4, 16, True),
    ('even z13 K16 r4', json.load(open(R + 'even_k16_z13_r4.json')), 4, 16, True),
    ('ship K24 d20 r5', SHIP24, 5, 24, False),
    ('d27 K24 r4', json.load(open(R + 'k24_full_d27_r4.json')), 4, 24, False),
    ('even z13 K24 r4 e6', json.load(open(R + 'even_k24_z13_r4_e6.json')), 4, 24, True),
    ('even z11 K24(K16 tbl?)', json.load(open(R + 'even_k16_z11_r4.json')), 4, 24, True),
]
def main():
  es = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
  print(f"{'cfg':24s} {'k,m':>5} {'#mul':>4} " + ' '.join(f"{'2^-'+str(e):>6}" for e in es))
  for name, c, da, K, even in cfgs:
      row = []
      for e in es:
          u = grid(2.0 ** -e, K)
          ev = Ev()
          out, km = chain(c, da, K, u, even, ev)
          row.append(-math.log2(np.abs(out - target(u)).max()))
      print(f"{name:24s} {str(km):>5} {ev.nmult:4d} " + ' '.join(f"{b:6.1f}" for b in row))

if __name__ == "__main__":
    main()
