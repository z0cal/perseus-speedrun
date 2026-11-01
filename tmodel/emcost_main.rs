//! EvalMod op census per series: relins (Mult/MultAcc) with limb counts, tensor products, cheap limb-ops, rescales.
//! usage: emcost [COEFFS.json DA ...]  (no args: shipped/d27 K16/K24). Prints one JSON line per table.
use pc_core::bts_setup::EvalMod;
use pc_core::scale::ScaleTables;
use pc_shadow::evalmod::{build_evalmod, EmOp, PsPlan};
use pc_shadow::fides::FMeta;

fn census(st: &ScaleTables, name: &str, c: &[f64], da: i32, top: i32) {
    let input = FMeta { top, nl: 2, nf: st.sf(top).unwrap() * st.sf(top).unwrap() };
    let ps = PsPlan::from_coeffs(c).expect("PS plan");
    let (prog, out) = build_evalmod(st, input, &ps, da).expect("build");
    let ops = prog.fused();
    let (mut relin, mut relin_n, mut tens_n, mut cheap_n, mut resc, mut resc_n) = (0, 0usize, 0usize, 0usize, 0, 0usize);
    let mut relin_ns = vec![];
    let mut kinds = std::collections::BTreeMap::<&str, usize>::new();
    for op in &ops {
        let k = match op {
            EmOp::Copy { n, .. } => { cheap_n += 2 * n; "copy" }
            EmOp::Add { n, .. } => { cheap_n += 2 * n; "add" }
            EmOp::MulConst { n, .. } => { cheap_n += 2 * n; "mulconst" }
            EmOp::AddConst { n, .. } => { cheap_n += n; "addconst" }
            EmOp::Rescale { n, .. } => { resc += 1; resc_n += n; "rescale" }
            EmOp::Mult { n, .. } => { relin += 1; relin_n += n; tens_n += 4 * n; relin_ns.push(*n); "mult" }
            EmOp::MultAcc { n, pairs, .. } => { relin += 1; relin_n += n; tens_n += 4 * n * pairs.len(); relin_ns.push(*n); "multacc" }
            EmOp::WSum { n, srcs, .. } => { cheap_n += 2 * n * srcs.len(); "wsum" }
        };
        *kinds.entry(k).or_default() += 1;
    }
    let oplist: Vec<String> = ops.iter().map(|op| match op {
        EmOp::Copy { n, .. } => format!("[\"copy\",{n},1]"),
        EmOp::Add { n, .. } => format!("[\"add\",{n},1]"),
        EmOp::MulConst { n, .. } => format!("[\"mulconst\",{n},1]"),
        EmOp::AddConst { n, .. } => format!("[\"addconst\",{n},1]"),
        EmOp::Rescale { n, .. } => format!("[\"rescale\",{n},1]"),
        EmOp::Mult { n, .. } => format!("[\"mult\",{n},1]"),
        EmOp::MultAcc { n, pairs, .. } => format!("[\"multacc\",{n},{}]", pairs.len()),
        EmOp::WSum { n, srcs, .. } => format!("[\"wsum\",{n},{}]", srcs.len()),
    }).collect();
    println!("{{\"name\":\"{name}\",\"deg\":{},\"da\":{da},\"top\":{top},\"levels\":{},\"k\":{},\"m\":{},\"ops\":{},\"relin\":{relin},\"relin_n\":{relin_n},\"relin_ns\":{:?},\"tens_n\":{tens_n},\"cheap_n\":{cheap_n},\"rescale\":{resc},\"rescale_n\":{resc_n},\"kinds\":{:?},\"oplist\":[{}]}}",
        c.len() - 1, top - out.top, ps.k, ps.m, ops.len(), relin_ns, kinds, oplist.join(","));
}

fn main() {
    let spec = if std::env::var("CHAIN").as_deref() == Ok("n32a4") { pc_core::params::ChainSpec::n32a4() } else { pc_core::params::ChainSpec::n32() };
    let chain = pc_core::params::Chain::generate(&spec).expect("chain");
    eprintln!("chain q {} p {} digits {:?}", chain.q.len(), chain.p.len(), chain.digits.iter().map(|d| d.len()).collect::<Vec<_>>());
    let st = ScaleTables::precompute(&chain).expect("tables");
    let args: Vec<String> = std::env::args().skip(1).collect();
    let top = std::env::var("TOP").ok().and_then(|s| s.parse().ok()).unwrap_or(45);
    if args.is_empty() {
        for em in [EvalMod::Shipped, EvalMod::D27] {
            let da = em.da_its() as i32;
            census(&st, &format!("{em:?}_K16"), em.dense(), da, top);
            census(&st, &format!("{em:?}_K24"), em.sparse_k24(), da, top);
        }
    }
    for p in args.chunks(2) {
        let c: Vec<f64> = serde_json::from_str(&std::fs::read_to_string(&p[0]).unwrap()).unwrap();
        census(&st, &p[0], &c, p[1].parse().unwrap(), top);
    }
}
