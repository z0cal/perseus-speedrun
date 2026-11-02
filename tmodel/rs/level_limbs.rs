//! level (virtual grid) -> primes dropped, for a chain preset. usage: level_limbs n32sp
fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "n32sp".into());
    let chain = pc_core::params::Chain::generate(&pc_core::params::ChainSpec::preset(&name).unwrap()).unwrap();
    let shp = pc_shadow::ChainShape::of(&chain);
    let mut m = serde_json::Map::new();
    for lv in 0..=120u32 {
        if let Some(d) = shp.primes_dropped(lv) {
            m.insert(lv.to_string(), serde_json::json!(d));
        }
    }
    println!("{}", serde_json::json!({"chain": name, "nq": chain.q.len(), "np": chain.p.len(), "digits": chain.digits.iter().map(|d| d.len()).collect::<Vec<_>>(), "dropped": m}));
}
