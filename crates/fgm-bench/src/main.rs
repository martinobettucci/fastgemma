use fgm_core::{KvCache, Model, Runner};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(String::as_str).unwrap_or("/home/user/models/g4e2b.fgm");
    let t0 = Instant::now();
    let model = Model::open(path).expect("open model");
    println!("loaded {} ({:.2} GB) in {:?}", path, model.total_bytes() as f64 / 1e9, t0.elapsed());
    let cfg = &model.cfg;
    println!("  H={} L={} heads={} kv={} hd={}/{} vocab={}",
        cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads,
        cfg.num_key_value_heads, cfg.head_dim, cfg.global_head_dim, cfg.vocab_size);

    let toks: Vec<u32> = args.get(2)
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![2, 2364, 573, 3287]);
    let max_ctx = 512;
    let mut kv = KvCache::new(cfg, max_ctx);
    println!("  kv cache {:.1} MB for {} ctx", kv.bytes() as f64 / 1e6, max_ctx);
    let mut r = Runner::new(&model, toks.len().max(8), max_ctx);

    let t = Instant::now();
    let logits = r.forward(&toks, 0, &mut kv);
    let el = t.elapsed();
    let mut top: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("prefill {} tok in {:?} ({:.1} tok/s)", toks.len(), el, toks.len() as f64 / el.as_secs_f64());
    println!("top5: {:?}", &top[..5]);
    if let Some(dump) = args.get(3) {
        let mut bytes = Vec::with_capacity(logits.len() * 4);
        for v in logits { bytes.extend_from_slice(&v.to_le_bytes()); }
        std::fs::write(dump, &bytes).expect("write logits");
        println!("wrote {} logits -> {}", logits.len(), dump);
    }
}
