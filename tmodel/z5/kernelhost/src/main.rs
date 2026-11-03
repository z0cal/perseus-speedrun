//! The real raise kernels (bts.cuh, compiled for the CPU through shim/) on ciphertexts under the production sparse secret
//! of the n32sp chain: the overflow integer I each lift produces, exact arithmetic on the outputs.
//! usage: kernelhost census TRIALS | scaled TRIALS | slots TRIALS
use pc_core::bts_setup::{evalmod_coeffs, EVALMOD_EVEN_CHEB, EVALMOD_K};
use pc_core::keys::{sample_sparse_ternary, Prng, SecretKey};
use pc_core::modmul::{invmod, mulmod};
use pc_core::scale::ScaleTables;
use pc_gpu::bts::raise::{raise_args, raise_tab, small_int_host, Lift};
use pc_kernels::{ModRaiseArgs, PrimeConst, SmallIntArgs, SmallIntTab};
use rayon::prelude::*;
use std::f64::consts::PI;

extern "C" {
    fn host_mod_raise(a: *const ModRaiseArgs) -> i32;
    fn host_small_int(a: *const SmallIntArgs) -> i32;
}

const N: usize = 1 << 16;

struct Cx {
    q: Vec<u64>,
    qd: i128,
    pr: Vec<PrimeConst>,
    st: ScaleTables,
}

impl Cx {
    fn new() -> Cx {
        let chain = pc_client::spec::chain_of("n32sp").unwrap();
        assert_eq!(chain.n, N);
        assert_eq!(chain.level_unit, 2);
        let q = chain.q.clone();
        let qd = q[0] as i128 * q[1] as i128;
        let pr = q.iter().map(|&p| PrimeConst::new(p, 32)).collect();
        let st = ScaleTables::precompute(&chain).unwrap();
        Cx { q, qd, pr, st }
    }
    /// The tables `RaisePlan::new` builds (production functions).
    fn margs(&self, top: usize) -> ModRaiseArgs {
        raise_args(&self.q, 2, top)
    }
    fn tab(&self, lift: Lift, top: usize, c: Option<f64>) -> SmallIntTab {
        raise_tab(&self.q, 2, top, c, lift).unwrap()
    }
}

/// One polynomial pair at the composite bottom, coefficient form: part p, row r at (p nrows + r) N.
struct Bottom {
    s: Vec<(usize, i64)>,
    m: Vec<i64>,
    /// rows 0 and 1 of c0, c1
    c: [[Vec<u32>; 2]; 2],
}

fn encrypt(cx: &Cx, seed: u32, m: Vec<i64>) -> Bottom {
    let mut pr = Prng::from_seed([seed, 2, 3, 4, 5, 6, 7, 8], seed);
    let s = sample_sparse_ternary(&mut pr, N, SecretKey::SPARSE_H).unwrap();
    let s: Vec<(usize, i64)> = s.iter().enumerate().filter(|(_, &v)| v != 0).map(|(i, &v)| (i, v as i64)).collect();
    assert_eq!(s.len(), 32);
    let mut c: [[Vec<u32>; 2]; 2] = Default::default();
    for r in 0..2 {
        let p = cx.q[r] as i64;
        let c1: Vec<u32> = (0..N).map(|_| pr.uniform_below(p as u64) as u32).collect();
        let c0: Vec<u32> = (0..N)
            .map(|j| {
                let mut x = 0i64;
                for &(i, v) in &s {
                    if j >= i { x += v * c1[j - i] as i64 } else { x -= v * c1[N + j - i] as i64 }
                }
                (m[j] - x).rem_euclid(p) as u32
            })
            .collect();
        c[0][r] = c0;
        c[1][r] = c1;
    }
    Bottom { s, m, c }
}

/// The real kernels: what `raise_from_bottom` launches for this lift. Returns the rows 0..=top of both parts.
fn raise(cx: &Cx, b: &Bottom, lift: Lift, top: usize, c: Option<f64>) -> (Vec<u32>, usize, SmallIntTab) {
    let nrows = top + 1;
    let mut buf = vec![0u32; 2 * nrows * N];
    for p in 0..2 {
        for r in 0..2 {
            buf[(p * nrows + r) * N..(p * nrows + r + 1) * N].copy_from_slice(&b.c[p][r]);
        }
    }
    let part_stride = (nrows * N) as u64;
    if let Some((r0, r1)) = lift.mod_raise_rows(2) {
        let mut a = cx.margs(top);
        a.x = buf.as_mut_ptr().cast();
        a.pr = cx.pr.as_ptr();
        a.ct_stride = 2 * part_stride;
        a.part_stride = part_stride;
        a.k = 1;
        a.r0 = r0 as i32;
        a.r1 = r1 as i32;
        a.nparts = 2;
        assert_eq!(unsafe { host_mod_raise(&a) }, 0);
    }
    let tab = cx.tab(lift, top, c);
    let a = SmallIntArgs {
        src: buf.as_ptr().cast(),
        out: buf.as_mut_ptr().cast(),
        tab: &tab,
        pr: cx.pr.as_ptr(),
        src_ct_stride: 2 * part_stride,
        src_part_stride: part_stride,
        ct_stride: 2 * part_stride,
        part_stride,
        k: 1,
        nparts: 2,
    };
    assert_eq!(unsafe { host_small_int(&a) }, 0);
    (buf, nrows, tab)
}

/// c0 + c1 s mod q_r, coefficient j, from the kernel's row r.
fn phase(cx: &Cx, b: &Bottom, buf: &[u32], nrows: usize, r: usize, j: usize) -> u64 {
    let p = cx.q[r] as i64;
    let at = |part: usize, i: usize| buf[(part * nrows + r) * N + i] as i64;
    let mut x = at(0, j);
    for &(i, v) in &b.s {
        if j >= i { x += v * at(1, j - i) } else { x -= v * at(1, N + j - i) }
    }
    x.rem_euclid(p) as u64
}

/// I per coefficient from the kernel's unscaled output: (c0' + c1' s - m) / Qd, exact. Rows 2 and 3 are primes that do not
/// divide Qd and must agree; rows 0 and 1 must see c0' + c1' s = m.
fn overflow(cx: &Cx, b: &Bottom, lift: Lift) -> Vec<i32> {
    let (buf, nrows, _) = raise(cx, b, lift, 3, None);
    let qdinv: Vec<u64> = (0..4).map(|r| if r < 2 { 0 } else { invmod((cx.qd % cx.q[r] as i128) as u64, cx.q[r]) }).collect();
    (0..N)
        .map(|j| {
            for r in 0..2 {
                let p = cx.q[r];
                assert_eq!(phase(cx, b, &buf, nrows, r, j), b.m[j].rem_euclid(p as i64) as u64, "row {r} coefficient {j} lost m");
            }
            let mut got = [0i64; 2];
            for (o, r) in [2usize, 3].into_iter().enumerate() {
                let p = cx.q[r];
                let d = (phase(cx, b, &buf, nrows, r, j) as i64 - b.m[j]).rem_euclid(p as i64) as u64;
                let i = mulmod(d, qdinv[r], p) as i64;
                got[o] = if i > p as i64 / 2 { i - p as i64 } else { i };
            }
            assert_eq!(got[0], got[1], "rows 2 and 3 disagree at coefficient {j}");
            got[0] as i32
        })
        .collect()
}

fn census(trials: usize) {
    let cx = Cx::new();
    for lift in [Lift::PerPrime, Lift::Centred] {
        let hist = (0..trials)
            .into_par_iter()
            .map(|t| {
                let mut g = Prng::from_seed([t as u32 + 11, 9, 9, 9, 9, 9, 9, 9], t as u32 + 1);
                let dmax = (cx.qd as f64) * (-4f64).exp2();
                let m: Vec<i64> = (0..N).map(|_| ((2.0 * g.uniform_f64() - 1.0) * dmax) as i64).collect();
                let b = encrypt(&cx, t as u32 + 1, m);
                let mut h = vec![0u64; 81];
                for i in overflow(&cx, &b, lift) {
                    h[(i + 40) as usize] += 1;
                }
                h
            })
            .reduce(|| vec![0u64; 81], |mut a, b| {
                a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                a
            });
        let tot: u64 = hist.iter().sum();
        let var: f64 = hist.iter().enumerate().map(|(k, &v)| ((k as f64 - 40.0).powi(2)) * v as f64).sum::<f64>() / tot as f64;
        let maxi = hist.iter().enumerate().filter(|(_, &v)| v > 0).map(|(k, _)| (k as i64 - 40).abs()).max().unwrap();
        println!("{lift:?}: {tot} coefficients, sigma {:.4}, max |I| {maxi}", var.sqrt());
        for k in 4..=16i64 {
            let c: u64 = hist.iter().enumerate().filter(|(i, _)| (*i as i64 - 40).abs() >= k).map(|(_, &v)| v).sum();
            println!("  P(|I| >= {k:2}) = {:.3e}  ({c})", c as f64 / tot as f64);
        }
    }
}

/// The production constant: the raised ciphertext must decrypt to c (m + I Qd) up to the rounding of the two parts, on
/// every row; and the Rust host twin of small_int must agree with the kernel coefficient by coefficient (checks the shim).
fn scaled(trials: usize) {
    let cx = Cx::new();
    let top = cx.q.len() - 1 - 2; // raise drop 1: the even series' raise (raise drop 0 + 1 level saved)
    let (boot_k, t) = (16.0f64, 12u32);
    let exact = (cx.st.sf(top as i32).unwrap() / cx.qd as f64) * (1.0 / (boot_k * N as f64));
    let c = exact * (t as f64).exp2();
    println!("top {top}, c = {c:e}, c Qd = 2^{:.2}", (c * cx.qd as f64).log2());
    for lift in [Lift::PerPrime, Lift::Centred] {
        let (mut worst, mut maxi) = (0f64, 0i32);
        for tr in 0..trials {
            let mut g = Prng::from_seed([tr as u32 + 21, 9, 9, 9, 9, 9, 9, 9], tr as u32 + 1);
            let dmax = (cx.qd as f64) * (-6f64).exp2();
            let m: Vec<i64> = (0..N).map(|_| ((2.0 * g.uniform_f64() - 1.0) * dmax) as i64).collect();
            let b = encrypt(&cx, tr as u32 + 1, m);
            let is = overflow(&cx, &b, lift);
            let (buf, nrows, tab) = raise(&cx, &b, lift, top, Some(c));
            // the host twin on the inputs the kernel read
            if lift == Lift::Centred {
                for j in 0..256 {
                    let rows = [b.c[0][0][j] as u64, b.c[0][1][j] as u64];
                    let tw = small_int_host(&tab, &rows, &cx.q[..=top]);
                    for (r, &w) in tw.iter().enumerate() {
                        assert_eq!(w, buf[r * N + j] as u64, "twin vs kernel: row {r} coefficient {j}");
                    }
                }
            }
            // CRT over rows 3, 4, 5 of c0' + c1' s, centred: the integer the raised ciphertext decrypts to
            let (p3, p4, p5) = (cx.q[3] as i128, cx.q[4] as i128, cx.q[5] as i128);
            let inv34 = invmod((p3 % p4) as u64, p4 as u64) as i128;
            let inv345 = invmod(((p3 * p4) % p5) as u64, p5 as u64) as i128;
            for j in 0..N {
                let (a3, a4, a5) = (phase(&cx, &b, &buf, nrows, 3, j) as i128, phase(&cx, &b, &buf, nrows, 4, j) as i128, phase(&cx, &b, &buf, nrows, 5, j) as i128);
                let mut v = a3;
                v += p3 * ((a4 - v).rem_euclid(p4) * inv34 % p4);
                v += p3 * p4 * ((a5 - v).rem_euclid(p5) * inv345 % p5);
                let big = p3 * p4 * p5;
                if v > big / 2 {
                    v -= big;
                }
                let want = c * (b.m[j] as f64 + is[j] as f64 * cx.qd as f64);
                worst = worst.max((v as f64 - want).abs());
                maxi = maxi.max(is[j].abs());
            }
        }
        println!("{lift:?}: {trials} polys, |decrypted - c (m + I Qd)| max {worst:.3} (bound (1 + h) / 2 = 16.5), max |I| {maxi}");
        assert!(worst < 17.5, "the raised ciphertext does not decrypt to c (m + I Qd)");
    }
}

fn cheb(c: &[f64], x: f64) -> f64 {
    let (mut b1, mut b2) = (0.0, 0.0);
    for &ck in c[1..].iter().rev() {
        let t = 2.0 * x * b1 - b2 + ck;
        b2 = b1;
        b1 = t;
    }
    x * b1 - b2 + c[0] / 2.0
}
fn da(mut x: f64) -> f64 {
    let mut a = (2.0 * PI).powf(-1.0 / 32.0);
    for _ in 0..5 {
        x = 2.0 * x * x - a * a;
        a *= a;
    }
    x
}

#[derive(Clone)]
enum Series {
    Shipped(Vec<f64>),
    Even(Vec<f64>),
}
impl Series {
    fn at(&self, u: f64) -> f64 {
        match self {
            Series::Shipped(c) => da(cheb(c, u / EVALMOD_K)),
            Series::Even(c) => {
                let w = u / EVALMOD_K - 0.25 / EVALMOD_K;
                da(cheb(c, 2.0 * w * w - 1.0))
            }
        }
    }
}

fn z7() -> Vec<f64> {
    let s = std::fs::read_to_string("/home/zocal/pesquisa/perseus-speedrun/tmodel/z5/profile_by_I.py").unwrap();
    let l = s.lines().find(|l| l.starts_with("Z7=[")).unwrap();
    l[4..l.len() - 1].split(',').map(|x| x.trim().parse().unwrap()).collect()
}

fn fft(re: &mut [f64], im: &mut [f64], cs: &[(f64, f64)]) {
    let n = re.len();
    let lg = n.trailing_zeros();
    for i in 0..n {
        let j = (i as u32).reverse_bits() as usize >> (32 - lg);
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let step = n / len;
        for st in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (wr, wi) = cs[k * step];
                let (a, b) = (st + k, st + k + len / 2);
                let (tr, ti) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len *= 2;
    }
}

fn quant(v: &mut Vec<f64>, q: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q).round() as usize]
}

/// What the error of EvalMod alone does to the slots: the kernel's I of each coefficient through the shipped series, the
/// transforms taken as exact. No key-switch or rescale noise in this model.
fn slots(trials: usize) {
    let cx = Cx::new();
    let topz = cx.q.len() - 1 - 2;
    let top0 = cx.q.len() - 1;
    let series: Vec<(&str, Series, f64)> = vec![
        ("recorde", Series::Shipped(evalmod_coeffs().to_vec()), cx.st.sf(top0 as i32).unwrap() / cx.qd as f64),
        ("z7", Series::Even(z7()), cx.st.sf(topz as i32).unwrap() / cx.qd as f64),
        ("z5", Series::Even(EVALMOD_EVEN_CHEB.to_vec()), cx.st.sf(topz as i32).unwrap() / cx.qd as f64),
    ];
    let cs: Vec<(f64, f64)> = (0..N / 2).map(|k| { let a = 2.0 * PI * k as f64 / N as f64; (a.cos(), a.sin()) }).collect();
    let tw: Vec<(f64, f64)> = (0..N).map(|j| { let a = PI * j as f64 / N as f64; (a.cos(), a.sin()) }).collect();
    println!("sf/Qd: {:?}; f = (sf 2^deg / Qd) 2^-CF v, deg = 2 (correction = CF - deg)", series.iter().map(|s| s.2).collect::<Vec<_>>());
    println!("(-log2) lift      serie    rota        : max por polinomio p05 / mediana / p95 ; rms mediano");
    for lift in [Lift::PerPrime, Lift::Centred] {
        for &(route, cf) in &[("densa", 6i32), ("stcf", 8), ("stcf", 13)] {
            let per: Vec<Vec<(f64, f64)>> = (0..trials)
                .into_par_iter()
                .map(|t| {
                    let mut g = Prng::from_seed([t as u32 + 31, cf as u32, 9, 9, 9, 9, 9, 9], t as u32 + 1);
                    // the slot values: uniform in [-1, 1]; f = m / Qd is 2^-CF (sf / Qd) times the value (StC-first) or
                    // its inverse embedding, of RMS 1 / sqrt(3 N) (dense: a Gaussian of that RMS)
                    let v: Vec<f64> = (0..N)
                        .map(|j| {
                            if route == "densa" {
                                let (a, b) = (g.uniform_f64().max(1e-300), g.uniform_f64());
                                (-2.0 * a.ln()).sqrt() * (2.0 * PI * b).cos() / (3.0 * N as f64).sqrt()
                            } else if j < N / 2 {
                                2.0 * g.uniform_f64() - 1.0
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    let base = series[0].2 * 4.0 * (-(cf as f64)).exp2();
                    let m: Vec<i64> = v.iter().map(|x| (x * base * cx.qd as f64) as i64).collect();
                    let b = encrypt(&cx, t as u32 + 1, m);
                    let is = overflow(&cx, &b, lift);
                    series
                        .iter()
                        .map(|(_, sr, sf)| {
                            let scale = sf * 4.0 * (-(cf as f64)).exp2();
                            let e: Vec<f64> = (0..N)
                                .map(|j| {
                                    let f = v[j] * scale;
                                    (sr.at(is[j] as f64 + f) - f) / scale
                                })
                                .collect();
                            if route == "densa" {
                                let mut re: Vec<f64> = (0..N).map(|j| e[j] * tw[j].0).collect();
                                let mut im: Vec<f64> = (0..N).map(|j| e[j] * tw[j].1).collect();
                                fft(&mut re, &mut im, &cs);
                                let mx = re.iter().fold(0f64, |a, &x| a.max(x.abs()));
                                let rms = (re.iter().map(|x| x * x).sum::<f64>() / N as f64).sqrt();
                                (mx, rms)
                            } else {
                                let mx = e[..N / 2].iter().fold(0f64, |a, &x| a.max(x.abs()));
                                let rms = (e[..N / 2].iter().map(|x| x * x).sum::<f64>() / (N / 2) as f64).sqrt();
                                (mx, rms)
                            }
                        })
                        .collect()
                })
                .collect();
            for (k, (name, _, _)) in series.iter().enumerate() {
                let mut bits: Vec<f64> = per.iter().map(|r| -r[k].0.log2()).collect();
                let mut rms: Vec<f64> = per.iter().map(|r| r[k].1).collect();
                println!(
                    "{lift:?}  {name:8} {route} CF {cf:2}: {:6.2} / {:6.2} / {:6.2} ; rms {:.3e}",
                    quant(&mut bits, 0.05),
                    quant(&mut bits, 0.5),
                    quant(&mut bits, 0.95),
                    quant(&mut rms, 0.5)
                );
            }
        }
    }
}

/// Worst |series(i + f) - f| over |f| <= 2^-cf, in bits, for each |I|.
fn tail() {
    let cx = Cx::new();
    let topz = cx.q.len() - 1 - 2;
    let _ = topz;
    let series: Vec<(&str, Series)> = vec![("recorde", Series::Shipped(evalmod_coeffs().to_vec())), ("z7", Series::Even(z7())), ("z5", Series::Even(EVALMOD_EVEN_CHEB.to_vec()))];
    println!("|I|  -log2 pior erro de coeficiente (|f| <= 2^-13) recorde z7 z5 ; z5 no slot a CF 13: erro absoluto com A = 1");
    for i in 0..=16i32 {
        let bits: Vec<f64> = series
            .iter()
            .map(|(_, sr)| {
                let mut w = 0f64;
                for sgn in [-1.0, 1.0] {
                    for k in -64..=64 {
                        let f = k as f64 / 64.0 * (-13f64).exp2();
                        w = w.max((sr.at(sgn * i as f64 + f) - f).abs());
                    }
                }
                -w.log2()
            })
            .collect();
        println!("{i:3}   {:6.2} {:6.2} {:6.2} ; {:.2e}", bits[0], bits[1], bits[2], (13.0 - bits[2]).exp2());
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a[0] == "tail" {
        return tail();
    }
    let trials: usize = a[1].parse().unwrap();
    match a[0].as_str() {
        "census" => census(trials),
        "scaled" => scaled(trials),
        "slots" => slots(trials),
        o => panic!("mode {o}"),
    }
}
