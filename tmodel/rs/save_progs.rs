//! saveprogs.py without _pc: compile a plan dir against a graph dir (blocks chained) and save every program.
//! usage: save_progs GRAPH_DIR PLAN_DIR OUT_DIR [program|siblings|lockstep]
use pc_program::compile::{batch::Schedule, compile_dir, CompileOpts};
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let schedule = match a.get(3).map(String::as_str).unwrap_or("program") {
        "siblings" => Schedule::Siblings,
        "lockstep" => Schedule::Lockstep,
        _ => Schedule::Program,
    };
    let opts = CompileOpts { keep_going: true, schedule, ..CompileOpts::default() };
    let dc = compile_dir(Path::new(&a[1]), Path::new(&a[0]), &opts).expect("compile");
    std::fs::create_dir_all(&a[2]).unwrap();
    for (label, r) in dc.programs {
        match r {
            Ok(p) => {
                println!("{label}: {} launches, {} sites, {} check errors", p.ops.len(), p.bts_sites.len(), p.check.errors.len());
                for e in p.check.errors.iter().take(3) {
                    println!("   {e}");
                }
                p.save(Path::new(&a[2]).join(format!("{label}.json"))).unwrap();
            }
            Err(e) => println!("{label}: ERROR {e}"),
        }
    }
}
