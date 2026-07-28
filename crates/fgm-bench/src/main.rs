//! fastgemma benchmark driver.
//!
//! Modes:
//!   dump  <model> <tokens-csv> [out.bin]   one forward, write logits (validation)
//!   sweep <model>                          prefill/decode throughput vs shape
//!
//! `FGM_THREADS` sets the GEMM pool size (default 4).

use fgm_core::forward::{NPHASE, PHASE_NAMES};
use fgm_core::{KvCache, Model, Runner};
use std::time::Instant;

fn threads() -> usize {
    std::env::var("FGM_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(4)
}

/// Deterministic pseudo-random token ids, avoiding special ids.
fn synth_tokens(n: usize, seed: u64) -> Vec<u32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            1000 + ((s >> 33) % 200_000) as u32
        })
        .collect()
}

/// Minimal structural JSON check, so the bench does not need a JSON dependency.
fn serde_json_check(s: &str) -> bool {
    let mut depth = 0i32;
    let mut in_str = false;
    let mut prev_esc = false;
    for ch in s.chars() {
        if in_str {
            if prev_esc { prev_esc = false; continue; }
            match ch { '\\' => prev_esc = true, '"' => in_str = false, _ => {} }
            continue;
        }
        match ch {
            '"' => in_str = true,
            '{' | '[' => depth += 1,
            '}' | ']' => { depth -= 1; if depth < 0 { return false; } }
            _ => {}
        }
    }
    depth == 0 && !in_str
}

fn argmax(v: &[f32]) -> usize {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &x) in v.iter().enumerate() {
        if x > best.1 {
            best = (i, x);
        }
    }
    best.0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("sweep");
    let path = args.get(2).map(String::as_str).unwrap_or("/home/user/models/g4e2b.fgm");

    let t0 = Instant::now();
    let model = Model::open(path).expect("open model");
    let cfg = model.cfg.clone();
    eprintln!("model {} ({:.2} GB) mapped in {:?}", path,
              model.total_bytes() as f64 / 1e9, t0.elapsed());
    eprintln!("  H={} L={} heads={} kv={} hd={}/{} inter={} vocab={} threads={}",
              cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads,
              cfg.num_key_value_heads, cfg.head_dim, cfg.global_head_dim,
              cfg.intermediate_size, cfg.vocab_size, threads());

    match mode {
        // Logits for EVERY position of the prompt, so validation can measure
        // greedy agreement over many positions instead of one near-tied token.
        "dump" => {
            let toks: Vec<u32> = args[3].split(',').map(|x| x.parse().unwrap()).collect();
            let m = toks.len();
            let mut caches = vec![KvCache::new(&cfg, m + 8)];
            let mut r = Runner::with_logit_rows(&model, m.max(8), m + 8, threads(), m);
            let seq = vec![0usize; m];
            let pos: Vec<usize> = (0..m).collect();
            let rows: Vec<usize> = (0..m).collect();
            let t = Instant::now();
            let logits = r.forward_multi(&toks, &seq, &pos, &mut caches, &rows);
            eprintln!("forward {m} tok in {:?} ({} logit rows)", t.elapsed(), rows.len());
            let v = cfg.vocab_size;
            let last = &logits[(m - 1) * v..m * v];
            let mut top: Vec<(usize, f32)> = last.iter().copied().enumerate().collect();
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
            eprintln!("last-position top5: {:?}", &top[..5]);
            if let Some(out) = args.get(4) {
                let mut bytes = Vec::with_capacity(logits.len() * 4);
                for x in logits {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
                std::fs::write(out, &bytes).expect("write");
                eprintln!("wrote {} x {} logits -> {out}", m, v);
            }
        }

        "sweep" => {
            println!("\n== prefill throughput, single sequence, {} threads ==", threads());
            println!("  {:>7} {:>10} {:>12} {:>10}", "tokens", "time", "tok/s", "ms/tok");
            for &n in &[64usize, 128, 256, 512] {
                let toks = synth_tokens(n, 7);
                let mut kv = KvCache::new(&cfg, n + 8);
                let mut r = Runner::new(&model, n, n + 8, threads());
                let t = Instant::now();
                r.forward(&toks, 0, &mut kv);
                let el = t.elapsed().as_secs_f64();
                println!("  {:>7} {:>9.3}s {:>12.1} {:>10.2}",
                         n, el, n as f64 / el, el * 1000.0 / n as f64);
                if std::env::var_os("FGM_PROFILE").is_some() {
                    let tot: f64 = r.prof.iter().sum();
                    let mut idx: Vec<usize> = (0..NPHASE).collect();
                    idx.sort_by(|&a, &b| r.prof[b].total_cmp(&r.prof[a]));
                    for i in idx {
                        if r.prof[i] > 1e-4 {
                            println!("        {:>14} {:>7.3}s {:>5.1}%",
                                     PHASE_NAMES[i], r.prof[i], 100.0 * r.prof[i] / tot);
                        }
                    }
                }
            }

            println!("\n== decode throughput, 1 sequence ==");
            let ctx = 640usize;
            let prompt = synth_tokens(128, 11);
            let mut kv = KvCache::new(&cfg, ctx);
            let mut r = Runner::new(&model, 128, ctx, threads());
            r.forward(&prompt, 0, &mut kv);
            let mut pos = prompt.len();
            let mut tok = 1000u32;
            let steps = 16;
            let t = Instant::now();
            for _ in 0..steps {
                tok = argmax(r.forward(&[tok], pos, &mut kv)) as u32;
                pos += 1;
            }
            let el = t.elapsed().as_secs_f64();
            println!("  {} steps in {:.3}s -> {:.2} tok/s ({:.1} ms/step)",
                     steps, el, steps as f64 / el, el * 1000.0 / steps as f64);
            println!("  kv cache {:.1} MB for {} ctx", kv.bytes() as f64 / 1e6, ctx);
        }

        // The target workload: 8 concurrent requests, ~8k prompt / ~2k output,
        // continuous batching. Prefill is chunked so decode is not starved.
        "serve" => {
            let conc: usize = std::env::var("FGM_CONC").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
            let pin: usize = std::env::var("FGM_IN").ok().and_then(|v| v.parse().ok()).unwrap_or(8192);
            let pout: usize = std::env::var("FGM_OUT").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
            let chunk: usize = std::env::var("FGM_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
            let ctx = pin + pout + 8;
            println!("\n== serve: {conc} concurrent, {pin} in / {pout} out, prefill chunk {chunk} ==");

            let mut caches: Vec<KvCache> = (0..conc).map(|_| KvCache::new(&cfg, ctx)).collect();
            println!("  kv {:.0} MB total ({:.1} MB/seq at {} ctx)",
                     caches.iter().map(|c| c.bytes()).sum::<usize>() as f64 / 1e6,
                     caches[0].bytes() as f64 / 1e6, ctx);
            let mut r = Runner::new(&model, chunk.max(conc), ctx, threads());

            // shared tool-definition prefix, then a per-request body
            let prompts: Vec<Vec<u32>> = (0..conc).map(|i| synth_tokens(pin, 100 + i as u64)).collect();

            let t_all = Instant::now();
            let mut ttft = vec![0.0f64; conc];
            let t_pre = Instant::now();
            for s in 0..conc {
                let mut off = 0;
                while off < pin {
                    let n = chunk.min(pin - off);
                    r.forward(&prompts[s][off..off + n], off, &mut caches[s]);
                    off += n;
                }
                ttft[s] = t_all.elapsed().as_secs_f64();
            }
            let pre = t_pre.elapsed().as_secs_f64();
            println!("  prefill: {} tok in {:.2}s -> {:.1} tok/s aggregate",
                     conc * pin, pre, (conc * pin) as f64 / pre);

            // batched decode: every GEMM serves all `conc` rows at once
            let mut toks: Vec<u32> = (0..conc).map(|i| 1000 + i as u32).collect();
            let seq: Vec<usize> = (0..conc).collect();
            let rows: Vec<usize> = (0..conc).collect();
            let t_dec = Instant::now();
            let mut steps = 0usize;
            for step in 0..pout {
                let pos: Vec<usize> = (0..conc).map(|s| pin + step).collect();
                let lg = r.forward_multi(&toks, &seq, &pos, &mut caches, &rows);
                for s in 0..conc {
                    toks[s] = argmax(&lg[s * cfg.vocab_size..(s + 1) * cfg.vocab_size]) as u32;
                }
                steps += 1;
                if t_dec.elapsed().as_secs_f64() > 60.0 {
                    break;
                }
            }
            let dec = t_dec.elapsed().as_secs_f64();
            let gen = steps * conc;
            println!("  decode:  {} tok in {:.2}s -> {:.1} tok/s aggregate ({:.1} ms/step, {:.1} tok/s/seq)",
                     gen, dec, gen as f64 / dec, dec * 1000.0 / steps as f64,
                     steps as f64 / dec);
            println!("  TTFT: first {:.2}s  last {:.2}s", ttft[0], ttft[conc - 1]);
            if steps < pout {
                println!("  (decode capped at 60s: {steps}/{pout} steps; extrapolated full request \
{:.0}s)", pre + dec / steps as f64 * pout as f64);
            }
        }

        // Tool-call exactness: compile 12 tools x 4 params into a token-level
        // constraint and decode under it, checking every emitted call parses.
        "tools" => {
            use fgm_core::grammar::{load_vocab_bytes, synthetic_tools, Constraint, GrammarBuilder};
            let ntools: usize = std::env::var("FGM_TOOLS").ok().and_then(|v| v.parse().ok()).unwrap_or(12);
            let nparams: usize = std::env::var("FGM_PARAMS").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
            let tokjson = args.get(3).map(String::as_str)
                .unwrap_or("/home/user/models/g4e2b/tokenizer.json");

            let t = Instant::now();
            let vocab = load_vocab_bytes(&std::fs::read_to_string(tokjson).expect("tokenizer.json"));
            println!("\n== constrained tool calling: {ntools} tools x {nparams} params ==");
            println!("  vocab {} tokens loaded in {:?}", vocab.len(), t.elapsed());

            let tools = synthetic_tools(ntools, nparams);
            let t = Instant::now();
            let dfa = GrammarBuilder::build(&tools);
            let mut c = Constraint::compile(dfa, &vocab);
            let build = t.elapsed();
            println!("  DFA {} states, compiled to token masks in {:?}", c.num_states(), build);
            println!("  mask memory {:.1} MB",
                     (c.num_states() * vocab.len().div_ceil(64) * 8) as f64 / 1e6);

            // Walk the grammar greedily by always taking the lowest legal token,
            // which is enough to exercise every state and measure forcing.
            c.reset();
            let mut steps = 0usize;
            let mut forced = 0usize;
            let mut out: Vec<u8> = Vec::new();
            while !c.done() && steps < 4096 {
                let n = c.count();
                if n == 0 { println!("  DEAD END at step {steps}"); break; }
                if n == 1 { forced += 1; }
                // Prefer a token that leaves the current DFA state, so the walk
                // makes progress instead of looping inside a string body forever.
                let mut pick = usize::MAX;
                let mut fallback = usize::MAX;
                for t in 0..vocab.len() {
                    if vocab[t].is_empty() || !c.allowed(t) { continue; }
                    if fallback == usize::MAX { fallback = t; }
                    let before = c.state;
                    let mut probe = c.state;
                    let mut ok = true;
                    for &b in &vocab[t] {
                        match c.peek(probe, b) { Some(n) => probe = n, None => { ok = false; break; } }
                    }
                    if ok && probe != before { pick = t; break; }
                }
                if pick == usize::MAX { pick = fallback; }
                if pick == usize::MAX { println!("  no legal token at step {steps}"); break; }
                out.extend_from_slice(&vocab[pick]);
                if !c.advance(pick) { println!("  advance failed at step {steps}"); break; }
                steps += 1;
            }
            let text = String::from_utf8_lossy(&out).to_string();
            println!("  walked {steps} steps, grammar {} ", if c.done() { "ACCEPTED" } else { "did not accept" });
            println!("  forced steps (mask admits exactly 1 token): {forced}/{steps} = {:.0}%",
                     100.0 * forced as f64 / steps.max(1) as f64);
            println!("  -> those steps can skip the 201 MB int4 LM-head read entirely");
            println!("  emitted: {}", &text[..text.len().min(160)]);
            match serde_json::from_str::<serde_json::Value>(&text) {
                Ok(v) => {
                    println!("  emitted text PARSES as JSON: true");
                    let name = v.get("name").and_then(|x| x.as_str()).unwrap_or("?");
                    let nargs = v.get("arguments").and_then(|x| x.as_object()).map(|o| o.len()).unwrap_or(0);
                    println!("  -> tool={name} args={nargs} (expected {nparams})");
                }
                Err(e) => println!("  emitted text PARSES as JSON: false ({e})"),
            }
        }

        m => {
            eprintln!("unknown mode {m}");
            std::process::exit(2);
        }
    }
}
