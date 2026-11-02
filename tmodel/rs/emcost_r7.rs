//! tmodel's emcost on this tree: EvalMod op census (lean builder, fused, as the runtime programs it) for the shipped
//! K = 16 series and the even one, at every input top they build at. usage: emcost CHAIN > n32sp_em.jsonl
use pc_shadow::evalmod::{build_evalmod_opts, EmOp, PsPlan};
use pc_shadow::fides::FMeta;

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "n32sp".into());
    let chain = pc_core::params::Chain::generate(&pc_core::params::ChainSpec::preset(&name).unwrap()).unwrap();
    let st = pc_core::scale::ScaleTables::precompute(&chain).unwrap();
    for (sname, ps) in [("Shipped_K16", PsPlan::evalmod()), ("Even_K16", PsPlan::evalmod_even())] {
        for top in 0..=st.l as i32 {
            if st.drop[top as usize] == 0 {
                continue;
            }
            let sf = st.sf(top).unwrap();
            let Ok((prog, out)) = build_evalmod_opts(&st, FMeta { top, nl: 2, nf: sf * sf }, &ps, 5, true) else { continue };
            let ops = prog.fused();
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
            println!("{{\"name\":\"{sname}\",\"top\":{top},\"out_top\":{},\"oplist\":[{}]}}", out.top, oplist.join(","));
        }
    }
}
