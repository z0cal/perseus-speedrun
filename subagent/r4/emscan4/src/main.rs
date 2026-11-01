//! r4: EvalMod levels of the shipped and d27 series on record 4's lean chain n32a4 (and n32 for reference), with
//! build_evalmod (the routine the GPU's EvalModPlan::program calls): for every input top a route can feed it, the
//! output top, the primes consumed, and whether every consumed prime sits in a 2-prime (composite) level group.
use pc_core::bts_setup::EvalMod;
use pc_core::scale::ScaleTables;
use pc_shadow::evalmod::{build_evalmod, PsPlan};
use pc_shadow::fides::FMeta;

fn main() {
    for (cname, spec) in [("n32", pc_core::params::ChainSpec::n32()), ("n32a4", pc_core::params::ChainSpec::n32a4())] {
        let chain = pc_core::params::Chain::generate(&spec).expect("chain");
        let st = ScaleTables::precompute(&chain).expect("tables");
        // group of each limb id
        let mut gsize = vec![0usize; chain.q.len()];
        let mut top = 0;
        for &g in &chain.groups { for i in top..top + g { gsize[i] = g; } top += g; }
        println!("{cname}: q.len() = {}, groups (bottom first) = {:?}, lone_top = {:?}", chain.q.len(), chain.groups, chain.lone_top());
        let n_q = chain.q.len() as i32;
        let mut bad = 0; let mut non_pair = 0;
        for (kname, k) in [("K16", 0), ("K24", 1)] {
            for t in 0..n_q {
                if gsize[t as usize] != 2 { continue; }
                for nl in [1i32, 2] {
                    let Ok(sf) = st.sf(t) else { continue };
                    let nf = if nl == 2 { sf * sf } else { sf };
                    let input = FMeta { top: t, nl, nf };
                    let mut outs = vec![];
                    for em in [EvalMod::Shipped, EvalMod::D27] {
                        let c = if k == 0 { em.dense() } else { em.sparse_k24() };
                        let r = build_evalmod(&st, input, &PsPlan::from_coeffs(c).unwrap(), em.da_its() as i32);
                        outs.push(r.ok().map(|(p, o)| (o.top, o.nl, p.ops.len())));
                    }
                    if let (Some(a), Some(b)) = (outs[0], outs[1]) {
                        let pairs_ok = ((a.0 + 1)..=t).all(|i| gsize[i as usize] == 2);
                        if !pairs_ok { non_pair += 1; }
                        let ok = b.0 - a.0 == 2 && a.1 == b.1;
                        if !ok { bad += 1; }
                        // tops a route actually feeds: the raise lands at the chain top (n_q - 1) or a raise drop below it
                        if (!ok || !pairs_ok) || (t >= n_q - 1 - 2 * 6 && (t == n_q - 1 || t % 4 == (n_q - 1) % 4)) {
                            println!("  {kname} in top {t} nl {nl}: shipped -> top {} ({} primes, {} ops), d27 -> top {} ({} primes, {} ops){}{}",
                                a.0, t - a.0, a.2, b.0, t - b.0, b.2, if ok { "" } else { "  MISMATCH" }, if pairs_ok { "" } else { "  NON-PAIR LEVEL CONSUMED" });
                        }
                    }
                }
            }
        }
        println!("{cname}: mismatches {bad}, scans consuming a non-pair level {non_pair}");
    }
}
