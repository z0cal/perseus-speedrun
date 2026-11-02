//! The EvalMod programs the runtime executes (EvalModPlan::program: fused, rewritten, relin-rescales scheduled),
//! per series and input top, with the fused relinearization-rescale pairs and the level-key digits marked.
//! usage: em_prog CHAIN > em_prog.jsonl
use pc_gpu::bts::evalmod::EvalModPlan;
use pc_shadow::evalmod::{EmOp, PsPlan};
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
            let Ok((prog, out, nbuf)) = EvalModPlan::<u32>::program(&st, FMeta { top, nl: 2, nf: sf * sf }, &ps, 5) else { continue };
            let ops = &prog.ops;
            let mut list = Vec::new();
            let mut i = 0;
            while i < ops.len() {
                let op = &ops[i];
                // a Mult/MultAcc whose next op is the Rescale of its result (n >= 3): RelinRescale (fused)
                let fused = match (op, ops.get(i + 1), ops.get(i + 2)) {
                    (EmOp::Mult { dst, n, .. } | EmOp::MultAcc { dst, n, .. }, Some(EmOp::Rescale { x, n: n2, k: None }), _) if x == dst && n2 == n && *n >= 3 => 1,
                    (EmOp::Mult { dst, n, .. } | EmOp::MultAcc { dst, n, .. }, Some(EmOp::Add { dst: d, a, b, n: n2, .. }), Some(EmOp::Rescale { x, n: n3, k: None }))
                        if d == dst && a == dst && b != dst && n2 == n && x == dst && n3 == n && *n >= 3 => 2,
                    _ => 0,
                };
                let s = match op {
                    EmOp::Copy { n, .. } => format!("[\"copy\",{n},1]"),
                    EmOp::Add { n, .. } => format!("[\"add\",{n},1]"),
                    EmOp::MulConst { n, .. } => format!("[\"mulconst\",{n},1]"),
                    EmOp::AddConst { n, .. } => format!("[\"addconst\",{n},1]"),
                    EmOp::Rescale { n, k, .. } => format!("[\"rescale\",{n},{}]", if k.is_some() { 2 } else { 1 }),
                    EmOp::Mult { n, .. } => format!("[\"{}\",{n},1]", if fused > 0 { "multrs" } else { "mult" }),
                    EmOp::MultAcc { n, pairs, .. } => format!("[\"{}\",{n},{}]", if fused > 0 { "multaccrs" } else { "multacc" }, pairs.len()),
                    EmOp::WSum { n, srcs, .. } => format!("[\"wsum\",{n},{}]", srcs.len()),
                };
                list.push(s);
                if fused == 1 {
                    i += 2; // the rescale rides the relinearization
                } else if fused == 2 {
                    list.push(match &ops[i + 1] { EmOp::Add { n, .. } => format!("[\"add\",{n},1]"), _ => unreachable!() });
                    i += 3;
                } else {
                    i += 1;
                }
            }
            println!("{{\"name\":\"{sname}\",\"top\":{top},\"out_top\":{},\"buffers\":{nbuf},\"oplist\":[{}]}}", out.top, list.join(","));
        }
    }
}
