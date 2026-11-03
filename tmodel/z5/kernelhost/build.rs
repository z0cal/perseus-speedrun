// Compiles the real bts.cuh for the CPU: the launchers (<<<...>>> syntax) are cut from a copy, nothing else changes.
use std::path::PathBuf;
use std::process::Command;
fn main() {
    let kdir = "/home/zocal/pesquisa/perseus-speedrun/work/perseus-z5/crates/pc-kernels/kernels";
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed={kdir}/bts.cuh");
    println!("cargo:rerun-if-changed=host/host.cpp");
    println!("cargo:rerun-if-changed=shim/cuda_runtime.h");
    let src = std::fs::read_to_string(format!("{kdir}/bts.cuh")).unwrap();
    let launcher = src.find("int small_int1_l(").expect("first launcher");
    let cut = src[..launcher].rfind("template <class W, class MM>").expect("its template line");
    std::fs::write(out.join("bts_host.cuh"), format!("{}\n}}\n}}\n", &src[..cut])).unwrap();
    let o = out.join("host.o");
    let st = Command::new("g++")
        .args(["-O2", "-std=c++17", "-fPIC", "-Wno-unknown-pragmas", "-Ishim"])
        .arg(format!("-I{kdir}"))
        .arg(format!("-I{}", out.display()))
        .args(["-c", "host/host.cpp", "-o"])
        .arg(&o)
        .status()
        .unwrap();
    assert!(st.success(), "g++ failed");
    let st = Command::new("ar").arg("crs").arg(out.join("libkernelhost.a")).arg(&o).status().unwrap();
    assert!(st.success(), "ar failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=kernelhost");
    println!("cargo:rustc-link-lib=stdc++");
}
