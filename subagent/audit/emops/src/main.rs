// Audit: op mix (by EmOp kind and level) of an EvalMod program, old vs new tables. usage: emops COEFFS.json DA
use pc_core::scale::ScaleTables;
use pc_shadow::evalmod::{build_evalmod, EmOp, PsPlan};
use pc_shadow::fides::FMeta;
use std::collections::BTreeMap;
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let c: Vec<f64> = serde_json::from_str(&std::fs::read_to_string(&a[0]).unwrap()).unwrap();
    let da: i32 = a[1].parse().unwrap();
    let chain = pc_core::params::Chain::generate(&pc_core::params::ChainSpec::n32()).unwrap();
    let st = ScaleTables::precompute(&chain).unwrap();
    let top = 45;
    let input = FMeta { top, nl: 2, nf: st.sf(top).unwrap() * st.sf(top).unwrap() };
    let ps = PsPlan::from_coeffs(&c).unwrap();
    let (prog, out) = build_evalmod(&st, input, &ps, da).unwrap();
    let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
    let mut mult_limbs = 0usize;
    for op in prog.fused() {
        let s = format!("{op:?}");
        let k = s.split(|ch: char| !ch.is_alphanumeric()).next().unwrap().to_string();
        *kinds.entry(k.clone()).or_default() += 1;
        match &op {
            EmOp::Mult { n, .. } => mult_limbs += *n,
            EmOp::MultAcc { n, pairs, .. } => mult_limbs += *n * pairs.len(),
            _ => {}
        }
    }
    println!("{}: k={} m={} levels(limbs)={} raw_ops={} fused_ops={} kinds={:?} sum_limbs_over_mults={}", a[0], ps.k, ps.m, top - out.top, prog.ops.len(), prog.fused().len(), kinds, mult_limbs);
}
