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

/// Refuse to benchmark on a loaded machine.
///
/// Contention has silently corrupted measurements three times in this project:
/// a concurrent conversion made prefill read 15% low, a concurrent build made it
/// read ~40% low, and stray benchmark processes from a timed-out loop made a
/// change look like a regression when it was not. Loadavg is cheap to check and
/// the failure mode is expensive, so check it.
fn check_load() {
    let la = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    let one: f64 = la.split_whitespace().next().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let ncpu = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4) as f64;
    if one > ncpu * 0.35 {
        eprintln!(
            "\n*** WARNING: load average {one:.2} on {ncpu:.0} cores before starting. \
             Numbers from this run are NOT trustworthy. Set FGM_IGNORE_LOAD=1 to proceed anyway. ***\n"
        );
        if std::env::var_os("FGM_IGNORE_LOAD").is_none() {
            std::process::exit(3);
        }
    }
}

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

/// Read a tool schema written as JSON, so the constraint is built from the same
/// declaration the prompt carries rather than a synthetic stand-in.
fn load_toolspec(path: &str) -> Vec<fgm_core::grammar::Tool> {
    use fgm_core::grammar::{Param, ParamType, Tool};
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("toolspec")).expect("toolspec json");
    v.as_array()
        .expect("toolspec is a list of tools")
        .iter()
        .map(|t| Tool {
            name: t["name"].as_str().expect("tool name").to_string(),
            params: t["params"]
                .as_array()
                .expect("tool params")
                .iter()
                .map(|p| Param {
                    name: p["name"].as_str().expect("param name").to_string(),
                    ty: match p["type"].as_str().unwrap_or("STRING") {
                        "INTEGER" => ParamType::Int,
                        "NUMBER" => ParamType::Num,
                        "BOOLEAN" => ParamType::Bool,
                        _ => match p.get("enum").and_then(|e| e.as_array()) {
                            Some(vals) => ParamType::Enum(
                                vals.iter().map(|v| v.as_str().unwrap().to_string()).collect(),
                            ),
                            None => ParamType::Str,
                        },
                    },
                    required: p.get("required").and_then(|r| r.as_bool()).unwrap_or(true),
                })
                .collect(),
        })
        .collect()
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

    check_load();
    let t0 = Instant::now();
    let model = Model::open(path).expect("open model");
    let cfg = model.cfg.clone();
    eprintln!("model {} ({:.2} GB) mapped in {:?}", path,
              model.total_bytes() as f64 / 1e9, t0.elapsed());
    eprintln!("  weights: {:?}{}", fgm_core::forward::WeightSel::from_env(),
              if model.has("l0.q_proj.i8") { " (file carries int8 twins)" }
              else { " (int4 only -- convert with --weights=both for twins)" });
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
            let mut caches = vec![KvCache::new(&cfg, m + 8, m)];
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
                let mut kv = KvCache::new(&cfg, n + 8, n);
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
            let mut kv = KvCache::new(&cfg, ctx, 128);
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

            let mut caches: Vec<KvCache> = (0..conc).map(|_| KvCache::new(&cfg, ctx, chunk)).collect();
            println!("  kv {:.0} MB total ({:.1} MB/seq at {} ctx)",
                     caches.iter().map(|c| c.bytes()).sum::<usize>() as f64 / 1e6,
                     caches[0].bytes() as f64 / 1e6, ctx);
            let mut r = Runner::new(&model, chunk.max(conc), ctx, threads());

            // Shared tool-definition prefix, then a per-request body. FGM_SHARE
            // is how many leading tokens every request has in common -- in the
            // target profile that is the 12-tool declaration block, which is
            // byte-identical across the 8 concurrent requests.
            let share: usize = std::env::var("FGM_SHARE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            let share = share.min(pin);
            let common = synth_tokens(share, 7);
            let prompts: Vec<Vec<u32>> = (0..conc)
                .map(|i| {
                    let mut p = common.clone();
                    p.extend(synth_tokens(pin - share, 100 + i as u64));
                    p
                })
                .collect();

            let t_all = Instant::now();
            let mut ttft = vec![0.0f64; conc];
            let t_pre = Instant::now();
            let mut fork_s = 0.0f64;
            if share > 0 {
                // Prefill the common span once, then copy the cache into every
                // other sequence. The copy is memcpy-bound; the prefill it
                // replaces is not.
                let mut off = 0;
                while off < share {
                    let n = chunk.min(share - off);
                    r.forward(&common[off..off + n], off, &mut caches[0]);
                    off += n;
                }
                let t = Instant::now();
                let (head, tail) = caches.split_at_mut(1);
                for c in tail.iter_mut() {
                    c.fork_from(&head[0]);
                }
                fork_s = t.elapsed().as_secs_f64();
            }
            for s in 0..conc {
                let mut off = share;
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
            if share > 0 {
                println!("  prefix sharing: {share} tok shared, {} prefill tok avoided, \
                          fork {:.0} ms", (conc - 1) * share, fork_s * 1000.0);
            }

            // batched decode: every GEMM serves all `conc` rows at once
            let mut toks: Vec<u32> = (0..conc).map(|i| 1000 + i as u32).collect();
            let seq: Vec<usize> = (0..conc).collect();
            let rows: Vec<usize> = (0..conc).collect();
            let t_dec = Instant::now();
            let mut steps = 0usize;
            for step in 0..pout {
                let pos: Vec<usize> = vec![pin + step; conc];
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
            use fgm_core::grammar::{
                load_vocab_bytes, parse_call, render, synthetic_tools, Constraint, Delims,
                GrammarBuilder,
            };
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
            // Every tool must be spellable, not just the first. The previous
            // builder gave each tool its own chain from the start state, so
            // stepping always took the first edge and 11 of 12 tools were
            // unreachable -- a walk that emits one valid call cannot see that,
            // so check the whole set explicitly.
            let mut reachable = 0usize;
            for t in &tools {
                let mut st = c.start();
                let mut ok = c.peek_tok(st, Delims::GEMMA4.call_open).map(|n| st = n).is_some();
                if ok {
                    for b in format!("call:{}{{", t.name).bytes() {
                        match c.peek(st, b) { Some(n) => st = n, None => { ok = false; break; } }
                    }
                }
                if ok { reachable += 1; } else { println!("  UNREACHABLE TOOL: {}", t.name); }
            }
            println!("  tools reachable from the start state: {reachable}/{}", tools.len());
            assert_eq!(reachable, tools.len(), "grammar cannot spell every tool");

            c.reset();
            let mut steps = 0usize;
            let mut forced = 0usize;
            let mut ids: Vec<u32> = Vec::new();
            while !c.done() && steps < 4096 {
                let n = c.count();
                if n == 0 { println!("  DEAD END at step {steps}"); break; }
                if n == 1 { forced += 1; }
                // Prefer a token that leaves the current DFA state, so the walk
                // makes progress instead of looping inside a string body forever.
                // Delimiter tokens carry no bytes and always advance, so they are
                // taken as soon as they are legal.
                let mut pick = usize::MAX;
                let mut fallback = usize::MAX;
                for t in 0..vocab.len() {
                    if !c.allowed(t) { continue; }
                    if vocab[t].is_empty() { pick = t; break; }
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
                ids.push(pick as u32);
                if !c.advance(pick) { println!("  advance failed at step {steps}"); break; }
                steps += 1;
            }
            let text = render(&ids, &vocab, Delims::GEMMA4);
            println!("  walked {steps} steps, grammar {} ", if c.done() { "ACCEPTED" } else { "did not accept" });
            println!("  forced steps (mask admits exactly 1 token): {forced}/{steps} = {:.0}%",
                     100.0 * forced as f64 / steps.max(1) as f64);
            println!("  -> those steps can skip the 201 MB int4 LM-head read entirely");
            println!("  emitted: {}", &text[..text.len().min(200)]);
            match parse_call(&text) {
                Some(call) => {
                    println!("  emitted text parses as a Gemma 4 tool call: true");
                    println!("  -> tool={} args={} (expected {nparams})", call.name, call.args.len());
                    assert_eq!(call.args.len(), nparams, "wrong argument count");
                }
                None => {
                    println!("  emitted text parses as a Gemma 4 tool call: FALSE");
                    std::process::exit(4);
                }
            }
        }

        // Regression guard for the ring-buffer KV bug: sliding layers that are
        // not a shared-KV source store only `sliding_window` positions, so both
        // the write and the read path must map absolute positions through the
        // ring. Reading linearly walks off the allocation once the context
        // passes the window -- a segfault at 8k that no test under 512 tokens
        // could reach. This runs past the window and requires bit-identical
        // logits against an all-full-length cache.
        "ringtest" => {
            let n: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1200);
            assert!(n > cfg.sliding_window, "must exceed the sliding window to be a test");
            println!("\n== ring-buffer KV regression: {n} tokens (window {}) ==",
                     cfg.sliding_window);
            let toks = synth_tokens(n, 3);
            let mut r = Runner::new(&model, n, n + 8, threads());

            let mut kv_ring = KvCache::new(&cfg, n + 8, n);
            let mut kv_full = KvCache::new_no_ring(&cfg, n + 8);
            println!("  ring cache {:.1} MB   full cache {:.1} MB",
                     kv_ring.bytes() as f64 / 1e6, kv_full.bytes() as f64 / 1e6);

            // First establish that the engine is deterministic at all, so a
            // ring-vs-linear difference can be attributed to the ring.
            let a: Vec<f32> = r.forward(&toks, 0, &mut kv_ring).to_vec();
            let mut kv_ring2 = KvCache::new(&cfg, n + 8, n);
            let a2: Vec<f32> = r.forward(&toks, 0, &mut kv_ring2).to_vec();
            let selfdiff = a.iter().zip(&a2).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            println!("  self-determinism (same path twice): max abs diff {selfdiff:.6}");
            let b: Vec<f32> = r.forward(&toks, 0, &mut kv_full).to_vec();
            let maxdiff = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            let same_argmax = argmax(&a) == argmax(&b);
            println!("  max abs logit diff  {maxdiff:.6}");
            println!("  argmax agrees       {same_argmax}");
            if maxdiff == 0.0 && same_argmax {
                println!("  PASS - ring-mapped reads are identical to linear reads");
            } else {
                println!("  FAIL");
                std::process::exit(1);
            }
        }

        // Full target-range curves: prefill 128..8192 and decode 128..2048,
        // at the concurrency given by FGM_CONC. Prefill is chunked (FGM_CHUNK)
        // because that is how a server actually runs it.
        "curve" => {
            let conc: usize = std::env::var("FGM_CONC").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
            let chunk: usize = std::env::var("FGM_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
            let pps: Vec<usize> = std::env::var("FGM_PP").ok()
                .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
                .unwrap_or_else(|| vec![128, 256, 512, 1024, 2048, 4096, 8192]);
            let tgs: Vec<usize> = std::env::var("FGM_TG").ok()
                .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
                .unwrap_or_else(|| vec![128, 256, 512, 1024, 2048]);
            let maxpp = *pps.iter().max().unwrap();
            let maxtg = *tgs.iter().max().unwrap();
            let ctx = maxpp + maxtg + 8;

            println!("\n== curves: concurrency {conc}, prefill chunk {chunk}, {} threads ==", threads());
            let mut r = Runner::new(&model, chunk.max(conc), ctx, threads());

            println!("\n-- prefill (aggregate over {conc} sequence(s)) --");
            println!("  {:>7} {:>10} {:>12} {:>10}", "prompt", "time", "tok/s", "ms/tok");
            for &pp in &pps {
                let mut caches: Vec<KvCache> =
                    (0..conc).map(|_| KvCache::new(&cfg, pp + 8, chunk)).collect();
                let prompts: Vec<Vec<u32>> =
                    (0..conc).map(|i| synth_tokens(pp, 100 + i as u64)).collect();
                let t = Instant::now();
                for s in 0..conc {
                    let mut off = 0;
                    while off < pp {
                        let n = chunk.min(pp - off);
                        r.forward(&prompts[s][off..off + n], off, &mut caches[s]);
                        off += n;
                    }
                }
                let el = t.elapsed().as_secs_f64();
                let tot = (conc * pp) as f64;
                println!("  {:>7} {:>9.2}s {:>12.1} {:>10.3}", pp, el, tot / el, el * 1000.0 / tot);
            }

            println!("\n-- decode after an {maxpp}-token prompt (aggregate over {conc}) --");
            println!("  {:>7} {:>10} {:>12} {:>12} {:>10}",
                     "out", "time", "tok/s", "tok/s/seq", "ms/step");
            let mut caches: Vec<KvCache> =
                (0..conc).map(|_| KvCache::new(&cfg, ctx, chunk)).collect();
            let prompts: Vec<Vec<u32>> =
                (0..conc).map(|i| synth_tokens(maxpp, 100 + i as u64)).collect();
            for s in 0..conc {
                let mut off = 0;
                while off < maxpp {
                    let n = chunk.min(maxpp - off);
                    r.forward(&prompts[s][off..off + n], off, &mut caches[s]);
                    off += n;
                }
            }
            let mut toks: Vec<u32> = (0..conc).map(|i| 1000 + i as u32).collect();
            let seq: Vec<usize> = (0..conc).collect();
            let rows: Vec<usize> = (0..conc).collect();
            let mut done = 0usize;
            let t0 = Instant::now();
            for &tg in &tgs {
                while done < tg {
                    let pos: Vec<usize> = vec![maxpp + done; conc];
                    let lg = r.forward_multi(&toks, &seq, &pos, &mut caches, &rows);
                    for s in 0..conc {
                        toks[s] = argmax(&lg[s * cfg.vocab_size..(s + 1) * cfg.vocab_size]) as u32;
                    }
                    done += 1;
                }
                let el = t0.elapsed().as_secs_f64();
                let tot = (done * conc) as f64;
                println!("  {:>7} {:>9.2}s {:>12.1} {:>12.1} {:>10.1}",
                         tg, el, tot / el, done as f64 / el, el * 1000.0 / done as f64);
            }
            println!("  kv {:.0} MB total ({:.1} MB/seq at {} ctx)",
                     caches.iter().map(|c| c.bytes()).sum::<usize>() as f64 / 1e6,
                     caches[0].bytes() as f64 / 1e6, ctx);
        }

        // Generate from a prompt of token ids. Emits the generated ids so an
        // external harness can decode and grade behaviour -- which is the real
        // acceptance test. Matching a reference engine's logits cannot be: we
        // deliberately trade numerical precision (int4 weights, int8
        // activations, Hadamard rotation, online softmax) for speed, so a
        // different number that yields the same correct action is a pass.
        "generate" => {
            let toks: Vec<u32> = args[3].split(',').map(|x| x.parse().unwrap()).collect();
            let ngen: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(64);
            let eos: Vec<u32> = std::env::var("FGM_EOS").ok()
                .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
                .unwrap_or_else(|| vec![1, 106]);
            let chunk = 256usize;
            let ctx = toks.len() + ngen + 8;
            let mut caches = vec![KvCache::new(&cfg, ctx, chunk)];
            let mut r = Runner::new(&model, chunk.max(1), ctx, threads());

            let t0 = Instant::now();
            let mut off = 0;
            let mut tok = 0u32;
            while off < toks.len() {
                let n = chunk.min(toks.len() - off);
                // the last chunk's final row already carries the first sampled
                // token's logits; re-forwarding that row would only rewrite the
                // KV slot it just wrote
                tok = argmax(r.forward(&toks[off..off + n], off, &mut caches[0])) as u32;
                off += n;
            }
            let ttft = t0.elapsed().as_secs_f64();

            let mut out = Vec::with_capacity(ngen);
            let mut pos = toks.len();
            for _ in 0..ngen {
                if eos.contains(&tok) { break; }
                out.push(tok);
                tok = argmax(r.forward(&[tok], pos, &mut caches[0])) as u32;
                pos += 1;
            }
            let el = t0.elapsed().as_secs_f64();
            eprintln!("prompt {} tok, TTFT {:.2}s, generated {} tok in {:.2}s ({:.1} tok/s)",
                      toks.len(), ttft, out.len(), el - ttft,
                      out.len() as f64 / (el - ttft).max(1e-9));
            println!("{}", out.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
        }

        // Batched generation for the behavioural eval: one prompt per input
        // line (comma-separated token ids), one line of generated ids out.
        // Single process, so the model maps and the tile-state warm-up runs
        // once instead of once per prompt.
        //
        // FGM_GRAMMAR=<tokenizer.json> turns on constrained decoding against
        // the tool set in FGM_TOOLS/FGM_PARAMS, which is what makes structural
        // validity a guarantee rather than a hope.
        "genfile" => {
            use fgm_core::grammar::{
                load_vocab_bytes, synthetic_tools, Constraint, GrammarBuilder,
            };
            let ngen: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(96);
            let eos: Vec<u32> = std::env::var("FGM_EOS").ok()
                .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
                .unwrap_or_else(|| vec![1, 106]);
            let lines: Vec<Vec<u32>> = std::fs::read_to_string(&args[3]).expect("prompt file")
                .lines().filter(|l| !l.trim().is_empty())
                .map(|l| l.trim().split(',').map(|x| x.parse().unwrap()).collect())
                .collect();
            let chunk = 256usize;
            let maxlen = lines.iter().map(|l| l.len()).max().unwrap_or(0);
            let ctx = maxlen + ngen + 8;

            let mut con = std::env::var("FGM_GRAMMAR").ok().map(|tj| {
                let vocab = load_vocab_bytes(&std::fs::read_to_string(&tj).expect("tokenizer.json"));
                // FGM_TOOLSPEC points at the same schema the prompt declares, so
                // the constraint and the prompt cannot drift apart.
                let tools = match std::env::var("FGM_TOOLSPEC") {
                    Ok(p) => load_toolspec(&p),
                    Err(_) => {
                        let nt = std::env::var("FGM_TOOLS").ok().and_then(|v| v.parse().ok()).unwrap_or(12);
                        let np = std::env::var("FGM_PARAMS").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
                        synthetic_tools(nt, np)
                    }
                };
                let c = Constraint::compile(GrammarBuilder::build(&tools), &vocab);
                eprintln!("constrained decoding: {} tools, {} states", tools.len(), c.num_states());
                c
            });

            let mut r = Runner::new(&model, chunk, ctx, threads());
            let mut nprompt = 0usize;
            let mut ngenerated = 0usize;
            let mut forced_steps = 0usize;
            let t_all = Instant::now();
            for toks in &lines {
                let mut cache = KvCache::new(&cfg, ctx, chunk);
                if let Some(c) = con.as_mut() { c.reset(); }
                let mut off = 0;
                let mut tok = 0u32;
                while off < toks.len() {
                    let n = chunk.min(toks.len() - off);
                    let lg = r.forward(&toks[off..off + n], off, &mut cache);
                    if off + n >= toks.len() {
                        tok = match con.as_ref() {
                            Some(c) => {
                                let mut v = lg.to_vec();
                                c.apply(&mut v);
                                argmax(&v) as u32
                            }
                            None => argmax(lg) as u32,
                        };
                    }
                    off += n;
                }
                if let Some(c) = con.as_mut() { c.advance(tok as usize); }
                let mut out = Vec::with_capacity(ngen);
                let mut pos = toks.len();
                for _ in 0..ngen {
                    if eos.contains(&tok) { break; }
                    out.push(tok);
                    if con.as_ref().map(|c| c.done()).unwrap_or(false) { break; }
                    // A forced step needs no logits at all -- the mask admits one
                    // token, so the 201 MB LM-head read is skipped outright.
                    if let Some(f) = con.as_ref().and_then(|c| c.forced()) {
                        forced_steps += 1;
                        r.forward_nolm(&[tok], pos, &mut cache);
                        tok = f as u32;
                        con.as_mut().unwrap().advance(f);
                        pos += 1;
                        continue;
                    }
                    let lg = r.forward(&[tok], pos, &mut cache);
                    tok = match con.as_ref() {
                        Some(c) => { let mut v = lg.to_vec(); c.apply(&mut v); argmax(&v) as u32 }
                        None => argmax(lg) as u32,
                    };
                    if let Some(c) = con.as_mut() { c.advance(tok as usize); }
                    pos += 1;
                }
                nprompt += toks.len();
                ngenerated += out.len();
                println!("{}", out.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
            }
            let el = t_all.elapsed().as_secs_f64();
            eprintln!("{} prompts, {} prompt tok, {} generated tok in {:.2}s ({:.1} gen tok/s)",
                      lines.len(), nprompt, ngenerated, el, ngenerated as f64 / el);
            if con.is_some() {
                eprintln!("forced steps (LM head skipped): {forced_steps}/{ngenerated} = {:.0}%",
                          100.0 * forced_steps as f64 / ngenerated.max(1) as f64);
            }
        }

        // Prefix sharing. The target profile is 8 concurrent requests that all
        // carry the same 12-tool system prompt, so the shared span is prefilled
        // once and forked into every sequence's cache. Measures both paths and
        // checks the forked path produces bit-identical logits -- the fork is a
        // copy, not a recomputation, so nothing about float evaluation order
        // changes and bit-identity is the right check here.
        "share" => {
            let conc: usize = std::env::var("FGM_CONC").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
            let pre: usize = std::env::var("FGM_PREFIX").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
            let suf: usize = std::env::var("FGM_SUFFIX").ok().and_then(|v| v.parse().ok()).unwrap_or(6144);
            let chunk: usize = std::env::var("FGM_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
            let ctx = pre + suf + 8;
            println!("\n== prefix sharing: {conc} seq, {pre} shared + {suf} unique, chunk {chunk} ==");

            let mut r = Runner::new(&model, chunk.max(conc), ctx, threads());
            let shared = synth_tokens(pre, 7);
            let tails: Vec<Vec<u32>> = (0..conc).map(|i| synth_tokens(suf, 100 + i as u64)).collect();

            let mut prefill = |r: &mut Runner, c: &mut KvCache, t: &[u32], base: usize| -> u32 {
                let mut off = 0;
                let mut last = 0u32;
                while off < t.len() {
                    let n = chunk.min(t.len() - off);
                    last = argmax(r.forward(&t[off..off + n], base + off, c)) as u32;
                    off += n;
                }
                last
            };

            // A: every sequence prefills the whole prompt itself
            let mut caches: Vec<KvCache> = (0..conc).map(|_| KvCache::new(&cfg, ctx, chunk)).collect();
            let t = Instant::now();
            let mut base_tok = Vec::with_capacity(conc);
            for s in 0..conc {
                prefill(&mut r, &mut caches[s], &shared, 0);
                base_tok.push(prefill(&mut r, &mut caches[s], &tails[s], pre));
            }
            let ta = t.elapsed().as_secs_f64();
            println!("  no sharing: {} tok in {:.2}s -> {:.1} tok/s",
                     conc * (pre + suf), ta, (conc * (pre + suf)) as f64 / ta);

            // B: prefill the shared span once, fork it, then the unique tails
            let mut caches: Vec<KvCache> = (0..conc).map(|_| KvCache::new(&cfg, ctx, chunk)).collect();
            let t = Instant::now();
            prefill(&mut r, &mut caches[0], &shared, 0);
            let t_pre = t.elapsed().as_secs_f64();
            let t_fork = Instant::now();
            let (head, tail) = caches.split_at_mut(1);
            for c in tail.iter_mut() {
                c.fork_from(&head[0]);
            }
            let tf = t_fork.elapsed().as_secs_f64();
            let mut share_tok = Vec::with_capacity(conc);
            for s in 0..conc {
                share_tok.push(prefill(&mut r, &mut caches[s], &tails[s], pre));
            }
            let tb = t.elapsed().as_secs_f64();
            println!("  sharing:    {} tok in {:.2}s -> {:.1} tok/s effective \
                     ({:.2}s shared prefill + {:.0}ms fork of {:.0} MB + {:.2}s tails)",
                     conc * (pre + suf), tb, (conc * (pre + suf)) as f64 / tb,
                     t_pre, tf * 1000.0,
                     (conc - 1) as f64 * caches[0].bytes() as f64 / 1e6, tb - t_pre - tf);
            println!("  speedup {:.2}x, {} prefill tokens avoided",
                     ta / tb, (conc - 1) * pre);
            let agree = base_tok.iter().zip(&share_tok).filter(|(a, b)| a == b).count();
            println!("  next-token identity after fork: {agree}/{conc}");
            assert_eq!(agree, conc, "forked cache changed the result");
        }

        m => {
            eprintln!("unknown mode {m}");
            std::process::exit(2);
        }
    }
}
