//! EvalMod op census (tmodel/emcost_main.rs format) for the shipped tables, a coefficient file, and the EVEN form:
//! x = u/K on input; even: x <- x - 1/(4K) (AddConst), x <- x^2, x <- x + x, x <- x - 1 (z = 2 (u/K - 1/(4K))^2 - 1),
//! then the Chebyshev series in z, then the double angles.
//! usage: CHAIN=n32a4 TOP=42 emcost_even [NAME COEFFS.json DA EVEN_K(0 = no prepass) ...]
use pc_core::scale::ScaleTables;
use pc_shadow::evalmod::{build_chebyshev, build_double_angle, Builder, EmOp, PsPlan};
use pc_shadow::fides::FMeta;

fn build(st: &ScaleTables, c: &[f64], da: i32, top: i32, even_k: f64) -> (Vec<EmOp>, FMeta, PsPlan) {
    let nl: i32 = std::env::var("NL").ok().and_then(|s| s.parse().ok()).unwrap_or(2); let input = FMeta { top, nl, nf: if nl == 2 { st.sf(top).unwrap() * st.sf(top).unwrap() } else { st.sf(top).unwrap() } };
    let ps = PsPlan::from_coeffs(c).expect("PS plan");
    let mut b = Builder::new(st);
    let x = b.input(input);
    if even_k > 0.0 {
        b.add_scalar(x, -0.25 / even_k).unwrap();
        b.square(x).unwrap();
        b.add(x, x).unwrap();
        b.add_scalar(x, -1.0).unwrap();
    }
    build_chebyshev(&mut b, x, &ps).unwrap();
    build_double_angle(&mut b, x, da).unwrap();
    let out = b.metas[x];
    let prog = b.prog;
    (prog.fused(), out, ps)
}

fn census(st: &ScaleTables, name: &str, c: &[f64], da: i32, top: i32, even_k: f64) {
    let (ops, out, ps) = build(st, c, da, top, even_k);
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
    println!("{{\"name\":\"{name}\",\"deg\":{},\"da\":{da},\"top\":{top},\"levels\":{},\"out_top\":{},\"out_nl\":{},\"k\":{},\"m\":{},\"ops\":{},\"relin\":{relin},\"relin_n\":{relin_n},\"relin_ns\":{:?},\"tens_n\":{tens_n},\"cheap_n\":{cheap_n},\"rescale\":{resc},\"rescale_n\":{resc_n},\"kinds\":{:?},\"oplist\":[{}]}}",
        c.len() - 1, top - out.top, out.top, out.nl, ps.k, ps.m, ops.len(), relin_ns, kinds, oplist.join(","));
}

fn main() {
    let spec = if std::env::var("CHAIN").as_deref() == Ok("n32a4") { pc_core::params::ChainSpec::n32a4() } else { pc_core::params::ChainSpec::n32() };
    let chain = pc_core::params::Chain::generate(&spec).expect("chain");
    let st = ScaleTables::precompute(&chain).expect("tables");
    let top = std::env::var("TOP").ok().and_then(|s| s.parse().ok()).unwrap_or(42);
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        census(&st, "Shipped_K16", pc_core::bts_setup::evalmod_coeffs(), pc_core::bts_setup::EVALMOD_DA_ITS as i32, top, 0.0);
        census(&st, "Shipped_K24", &pc_core::bts_setup::SPARSE_B_CHEB_K24, pc_core::bts_setup::SPARSE_B_DA_ITS as i32, top, 0.0);
    }
    for p in args.chunks(4) {
        let c: Vec<f64> = serde_json::from_str(&std::fs::read_to_string(&p[1]).unwrap()).unwrap();
        census(&st, &p[0], &c, p[2].parse().unwrap(), top, p[3].parse().unwrap());
    }
}
