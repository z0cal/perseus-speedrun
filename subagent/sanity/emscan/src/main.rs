//! Sanity: EvalMod levels consumed by each series, for every input top a route could feed it (n32 chain), and the
//! output limb count: shipped vs d27 must differ by exactly 2 primes everywhere, with the same output nl.
use pc_core::bts_setup::EvalMod;
use pc_core::scale::ScaleTables;
use pc_shadow::evalmod::{build_evalmod, PsPlan};
use pc_shadow::fides::FMeta;

fn main() {
    let chain = pc_core::params::Chain::generate(&pc_core::params::ChainSpec::n32()).expect("n32");
    let st = ScaleTables::precompute(&chain).expect("tables");
    println!("chain q.len() = {}", chain.q.len());
    let mut bad = 0;
    for (name, k) in [("K16", 0), ("K24", 1)] {
        for top in (25..=53).filter(|t| t % 2 == 1) {
            for nl in [1i32, 2] {
                let nf = if nl == 2 { st.sf(top).unwrap() * st.sf(top).unwrap() } else { st.sf(top).unwrap() };
                let input = FMeta { top, nl, nf };
                let mut outs = vec![];
                for em in [EvalMod::Shipped, EvalMod::D27] {
                    let c = if k == 0 { em.dense() } else { em.sparse_k24() };
                    let r = build_evalmod(&st, input, &PsPlan::from_coeffs(c).unwrap(), em.da_its() as i32);
                    outs.push(r.map(|(_, o)| (o.top, o.nl)).map_err(|e| format!("{e}")));
                }
                match (&outs[0], &outs[1]) {
                    (Ok(a), Ok(b)) => {
                        let ok = a.0 - b.0 == -2 && a.1 == b.1;
                        if !ok { bad += 1; }
                        if !ok || (top % 4 == 1) {
                            println!("{name} in top {top} nl {nl}: shipped -> top {} nl {}, d27 -> top {} nl {}  ({} primes){}", a.0, a.1, b.0, b.1, top - a.0, if ok { "" } else { "  MISMATCH" });
                        }
                    }
                    (a, b) => println!("{name} in top {top} nl {nl}: shipped {:?} d27 {:?}", a.as_ref().map(|_| ()), b.as_ref().map(|_| ())),
                }
            }
        }
    }
    println!("mismatches: {bad}");
}
