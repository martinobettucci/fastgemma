//! fastgemma benchmark driver.
//!
//! Modes:
//!   dump  <model> <tokens-csv> [out.bin]   one forward, write logits (validation)
//!   sweep <model>                          prefill/decode throughput vs shape
//!
//! `FGM_THREADS` sets the pool size (default: all available cores).

use fgm_core::forward::{NPHASE, PHASE_NAMES};
use fgm_core::{KvCache, Model, Runner};
use std::time::Instant;

fn loadavg() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next()?.parse().ok())
        .unwrap_or(0.0)
}

/// PIDs of other fgm-bench processes running right now.
///
/// Matched on /proc/<pid>/comm, which is the executable name -- not the command
/// line, which would also match any shell whose arguments happen to mention
/// fgm-bench, including a watcher waiting for this very process to exit.
fn sibling_benches() -> Vec<u32> {
    let me = std::process::id();
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir("/proc") else { return out };
    for e in rd.flatten() {
        let Some(pid) = e.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        if pid == me {
            continue;
        }
        if let Ok(c) = std::fs::read_to_string(format!("/proc/{pid}/comm")) {
            if c.trim() == "fgm-bench" {
                out.push(pid);
            }
        }
    }
    out
}

/// Refuse to benchmark on a loaded machine.
///
/// Contention has silently corrupted measurements five times in this project: a
/// concurrent conversion made prefill read 15% low, a concurrent build made it
/// read ~40% low, stray processes from a timed-out loop made a change look like
/// a regression when it was not, a `cargo build` fired off mid-run, and an
/// orphaned bench from a killed script made an int4/int8 comparison read 2x low
/// on one side only.
///
/// That last one got through the loadavg check, and the reason matters:
/// **loadavg is a one-minute decaying average, so it badly under-reports a
/// competitor that just started.** A four-thread process one second old barely
/// moves it. So the primary check is now the direct one -- is another bench
/// running at all -- with loadavg kept as a secondary signal for everything
/// else (builds, conversions, whatever else the box is doing).
fn check_load() {
    let sibs = sibling_benches();
    if !sibs.is_empty() {
        eprintln!(
            "\n*** REFUSING: {} other fgm-bench process(es) running ({sibs:?}). \
             Two benches on 4 cores read about 2x low, and loadavg will not catch it \
             for the first minute. Set FGM_IGNORE_LOAD=1 to proceed anyway. ***\n",
            sibs.len()
        );
        if std::env::var_os("FGM_IGNORE_LOAD").is_none() {
            std::process::exit(3);
        }
    }
    let one = loadavg();
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

/// Re-check after the run. Contention that *starts* mid-run is invisible to any
/// check made before it, and that is precisely the case that produced a
/// published-looking 2x error. Loud on stdout, so it lands in the same log as
/// the numbers it invalidates rather than in a stderr stream nobody kept.
///
/// Only sibling processes are checked. A post-run **loadavg** threshold was
/// tried first and is unfixable: this benchmark runs `threads()` busy threads,
/// so it drives loadavg to roughly ncpu by itself, and any threshold that
/// catches a real competitor also catches the bench measuring its own load. It
/// fired on a perfectly clean prefix-sharing run (0.95 -> 3.76 on 4 cores) whose
/// sibling check was silent and whose numbers were fine.
///
/// A guard that cries wolf is worse than no guard: the next real warning gets
/// read as noise. The sibling scan is precise -- it names PIDs -- so it is the
/// whole check.
fn verify_clean() {
    let sibs = sibling_benches();
    if !sibs.is_empty() {
        println!(
            "\n*** NUMBERS ABOVE ARE SUSPECT: {} other fgm-bench process(es) appeared \
             during this run ({sibs:?}). Discard and re-measure. ***",
            sibs.len()
        );
    }
}

/// Pool size. Defaults to every available core rather than a hardcoded 4 --
/// the old default silently used 4 threads on a 16-core box, which reads as
/// "the engine does not scale" when it is really "the engine was not asked to".
fn threads() -> usize {
    std::env::var("FGM_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4))
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

/// Print and reset the per-phase timers. Called per shape so the breakdown is
/// attributable: "attention is 11% of prefill" is meaningless without saying at
/// which context, since that is the only share that moves.
fn prof_line(r: &mut Runner, label: &str) {
    if std::env::var_os("FGM_PROFILE").is_none() {
        return;
    }
    let tot: f64 = r.prof.iter().sum();
    if tot < 1e-6 {
        return;
    }
    let mut idx: Vec<usize> = (0..NPHASE).collect();
    idx.sort_by(|&a, &b| r.prof[b].total_cmp(&r.prof[a]));
    let parts: Vec<String> = idx
        .iter()
        .filter(|&&i| r.prof[i] / tot > 0.005)
        .map(|&i| format!("{} {:.1}%", PHASE_NAMES[i], 100.0 * r.prof[i] / tot))
        .collect();
    println!("      [{label}] {:.2}s: {}", tot, parts.join(", "));
    r.prof = [0.0; NPHASE];
}

/// Median and full range of a sample. Median rather than mean because a single
/// contended run is an outlier, not a shifted sample, and the range is printed
/// so a reader can see whether a reported delta clears the noise.
fn med_range(v: &[f64]) -> (f64, f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    let med = if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 };
    (med, s[0], s[n - 1])
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
    // FGM_SCALAR_WFILL=1 selects the scalar P.V weight fill for A/B.
    fgm_kernels::set_scalar_wfill(std::env::var_os("FGM_SCALAR_WFILL").is_some());
    let t0 = Instant::now();
    let model = Model::open(path).expect("open model");
    let cfg = model.cfg.clone();
    eprintln!("model {} ({:.2} GB) mapped in {:?}", path,
              model.total_bytes() as f64 / 1e9, t0.elapsed());
    // Probe an FFN weight, not an attention one. q/k/v/o are already int8 at
    // conversion (--attn-bits 8), so `--weights=both` never makes them a twin
    // and `l0.q_proj.i8` is absent even on a dual-format file. This banner
    // reported "int4 only" for the dual model through every A/B in this
    // session; the measurements were unaffected because the twins that matter
    // are the FFN ones, but the label was wrong.
    eprintln!("  weights: {:?}{}", fgm_core::forward::WeightSel::from_env(),
              if model.has("l0.gate_proj.i8") { " (file carries int8 FFN twins)" }
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

            // FGM_TOOLSPEC measures the real schema. Forced-run length depends
            // strongly on name length, and synthetic tool_N/p0 names are the
            // shortest possible case -- the least favourable estimate of the
            // batching win, not a representative one.
            let tools = match std::env::var("FGM_TOOLSPEC") {
                Ok(p) => load_toolspec(&p),
                Err(_) => synthetic_tools(ntools, nparams),
            };
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

            // Run lengths matter more than the raw percentage. A forced token is
            // determined by the DFA alone, independent of the model, so a RUN of
            // k consecutive forced tokens can be emitted at once and its KV
            // advanced in a single batched forward -- k steps collapse to 1.
            // Scattered forced steps collapse nothing. This is exact, not
            // speculative: no draft, no verification, no acceptance rate.
            {
                let mut c2 = Constraint::compile(GrammarBuilder::build(&tools), &vocab);
                c2.reset();
                let mut runs: Vec<usize> = Vec::new();
                let mut cur = 0usize;
                let mut n = 0usize;
                while !c2.done() && n < 4096 {
                    if let Some(f) = c2.forced() {
                        cur += 1;
                        c2.advance(f);
                    } else {
                        if cur > 0 { runs.push(cur); cur = 0; }
                        let mut pick = usize::MAX;
                        for t in 0..vocab.len() {
                            if !c2.allowed(t) { continue; }
                            if vocab[t].is_empty() { pick = t; break; }
                            let before = c2.state;
                            let mut probe = before;
                            let mut ok = true;
                            for &b in &vocab[t] {
                                match c2.peek(probe, b) { Some(x) => probe = x, None => { ok = false; break; } }
                            }
                            if ok && probe != before { pick = t; break; }
                        }
                        if pick == usize::MAX { break; }
                        c2.advance(pick);
                    }
                    n += 1;
                }
                if cur > 0 { runs.push(cur); }
                let total: usize = runs.iter().sum();
                let saved: usize = runs.iter().map(|r| r - 1).sum();
                println!("  forced RUNS: {:?}", runs);
                println!("  -> {} forced tokens in {} runs; batching each run into one",
                         total, runs.len());
                println!("     forward collapses {saved} of {n} decode steps ({:.0}%)",
                         100.0 * saved as f64 / n.max(1) as f64);
            }
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
                prof_line(&mut r, &format!("prefill {pp}"));
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
            r.prof = [0.0; NPHASE];
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
                prof_line(&mut r, &format!("decode {tg} @ctx {maxpp}"));
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

        // Full (prompt x output) matrix. `curve` measures decode only after the
        // longest prompt, which hides that decode cost depends on the context it
        // decodes from. Every cell here is measured: prefill at `pp`, then
        // decode with checkpoints at each `tg`, so a cell is a real end-to-end
        // request of that shape rather than two numbers added together.
        //
        // This is what a shape-conditional policy has to be fitted against.
        "matrix" => {
            let conc: usize = std::env::var("FGM_CONC").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
            let chunk: usize = std::env::var("FGM_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
            // Repeat each cell and report the median. A single run of a fixed
            // configuration on this box has std 4.7% on prefill and 4.4% on
            // decode, with a full range over five runs of ~11%. Deltas below
            // ~10% are therefore not measurable from one sample each, and two
            // claims were published from single runs before this was known.
            let reps: usize = std::env::var("FGM_REPEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
            let pps: Vec<usize> = std::env::var("FGM_PP").ok()
                .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
                .unwrap_or_else(|| vec![128, 256, 512, 1024, 2048, 4096, 8192]);
            let tgs: Vec<usize> = std::env::var("FGM_TG").ok()
                .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
                .unwrap_or_else(|| vec![128, 512, 2048]);
            let maxtg = *tgs.iter().max().unwrap();
            let ctx = *pps.iter().max().unwrap() + maxtg + 8;
            let mut r = Runner::new(&model, chunk.max(conc), ctx, threads());
            println!("\n== (prompt x output) matrix, concurrency {conc}, chunk {chunk}, \
                     {reps} rep(s) ==");
            println!("  weights {:?}", r.wsel);
            if reps > 1 {
                // AMX tile-guard retries are reported per row because the
                // platform's corruption rate is not constant: runs have shown
                // 5/12 and 7/12 trials corrupted at warm-up, and every retry
                // re-runs a whole GEMM block. A slow run with a high retry
                // count is the platform, not the code.
                println!("  {:>7} {:>7} {:>10} {:>15} {:>10} {:>15} {:>10} {:>10}",
                         "in", "out", "pp_tok/s", "pp_range", "tg_tok/s", "tg_range",
                         "req_tok/s", "retries");
            } else {
                println!("  {:>7} {:>7} {:>9} {:>9} {:>10} {:>10} {:>10}",
                         "in", "out", "ttft_s", "gen_s", "pp_tok/s", "tg_tok/s", "req_tok/s");
            }
            for &pp in &pps {
                let mut ttfts: Vec<f64> = Vec::with_capacity(reps);
                let mut gens: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); tgs.len()];
                for _ in 0..reps {
                    let mut caches: Vec<KvCache> =
                        (0..conc).map(|_| KvCache::new(&cfg, pp + maxtg + 8, chunk)).collect();
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
                    ttfts.push(t.elapsed().as_secs_f64());
                    prof_line(&mut r, &format!("prefill {pp}"));

                    let mut toks: Vec<u32> = (0..conc).map(|i| 1000 + i as u32).collect();
                    let seq: Vec<usize> = (0..conc).collect();
                    let rows: Vec<usize> = (0..conc).collect();
                    let mut done = 0usize;
                    let t0 = Instant::now();
                    for (gi, &tg) in tgs.iter().enumerate() {
                        while done < tg {
                            let pos: Vec<usize> = vec![pp + done; conc];
                            let lg = r.forward_multi(&toks, &seq, &pos, &mut caches, &rows);
                            for s in 0..conc {
                                toks[s] =
                                    argmax(&lg[s * cfg.vocab_size..(s + 1) * cfg.vocab_size]) as u32;
                            }
                            done += 1;
                        }
                        gens[gi].push(t0.elapsed().as_secs_f64());
                        prof_line(&mut r, &format!("decode {tg} @ctx {pp}"));
                    }
                }
                for (gi, &tg) in tgs.iter().enumerate() {
                    let pp_rates: Vec<f64> =
                        ttfts.iter().map(|t| (conc * pp) as f64 / t).collect();
                    let tg_rates: Vec<f64> =
                        gens[gi].iter().map(|t| (conc * tg) as f64 / t).collect();
                    let (ppm, pplo, pphi) = med_range(&pp_rates);
                    let (tgm, tglo, tghi) = med_range(&tg_rates);
                    let req: Vec<f64> = ttfts.iter().zip(&gens[gi])
                        .map(|(a, b)| (conc * (pp + tg)) as f64 / (a + b)).collect();
                    let (reqm, _, _) = med_range(&req);
                    if reps > 1 {
                        println!("  {:>7} {:>7} {:>10.1} {:>15} {:>10.1} {:>15} {:>10.1} {:>10}",
                                 pp, tg, ppm, format!("{pplo:.0}-{pphi:.0}"),
                                 tgm, format!("{tglo:.1}-{tghi:.1}"), reqm,
                                 fgm_kernels::tile_retries());
                    } else {
                        println!("  {:>7} {:>7} {:>9.2} {:>9.2} {:>10.1} {:>10.1} {:>10.1}",
                                 pp, tg, ttfts[0], gens[gi][0], ppm, tgm, reqm);
                    }
                }
            }
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
            let (mut mtp_batched, mut mtp_drafted, mut mtp_accepted) = (0usize, 0usize, 0usize);
            let mut dec_secs = 0.0f64;
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
                // ---------------------------------------------- MTP paths
                // Both are OFF by default and exist to be A/B'd. Neither has
                // been benchmarked end to end: the host lost its AMX units
                // before that was possible, so these ship measured only at the
                // DFA/tokeniser level (JOURNAL 28).
                //
                //   FGM_MTP_GRAMMAR=1  batch runs of DFA-forced tokens. EXACT:
                //                      a forced token is determined by the
                //                      grammar alone, so there is nothing to
                //                      verify. Sized at ~11% of decode steps.
                //   FGM_MTP_LOOKUP=k   prompt-lookup speculation, drafting up
                //                      to k tokens by finding the last 2 emitted
                //                      tokens in the prompt and copying what
                //                      followed. Sized at 21.6% draft accuracy
                //                      on the eval's tool calls. Verified
                //                      against the model, so acceptance is
                //                      checked, never assumed.
                let mtp_grammar = std::env::var_os("FGM_MTP_GRAMMAR").is_some();
                let lookahead: usize =
                    std::env::var("FGM_MTP_LOOKUP").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                let vsz = cfg.vocab_size;

                let mut out = Vec::with_capacity(ngen);
                let mut pos = toks.len();
                // Decode is timed separately from prefill. Every MTP path acts
                // only on decode, and decode is a quarter of this harness's
                // wall clock -- an 11% decode win is a 3% total win, which is
                // inside the noise floor. Timing the whole run would have
                // measured nothing and reported it as "no effect".
                let t_dec = Instant::now();
                while out.len() < ngen {
                    if eos.contains(&tok) { break; }
                    out.push(tok);
                    if con.as_ref().map(|c| c.done()).unwrap_or(false) { break; }

                    // --- exact path: emit the whole forced run in one forward
                    if mtp_grammar {
                        let run: Vec<u32> = con.as_ref()
                            .map(|c| c.forced_run(15).into_iter().map(|t| t as u32).collect())
                            .unwrap_or_default();
                        if run.len() > 1 && out.len() + run.len() <= ngen {
                            // [tok, run[0..k-1]] advances KV for all of them;
                            // logits come from the last row, which is the first
                            // position the grammar does NOT determine.
                            let mut batch = Vec::with_capacity(run.len());
                            batch.push(tok);
                            batch.extend_from_slice(&run[..run.len() - 1]);
                            let lg = r.forward(&batch, pos, &mut cache);
                            let mut v = lg.to_vec();
                            for &t in &run { out.push(t); }
                            let last = *run.last().unwrap();
                            if let Some(c) = con.as_mut() {
                                for &t in &run { c.advance(t as usize); }
                                c.apply(&mut v);
                            }
                            let _ = &v;
                            mtp_batched += run.len() - 1;
                            pos += batch.len();
                            // `last` was emitted but its KV is not written yet;
                            // it becomes the next step's `tok`.
                            tok = last;
                            out.pop();
                            continue;
                        }
                    }

                    // --- speculative path: draft from the prompt, verify
                    if lookahead > 0 && out.len() >= 2 {
                        let ng = [out[out.len() - 2], out[out.len() - 1]];
                        let mut draft: Vec<u32> = Vec::new();
                        if let Some(j) = (2..toks.len())
                            .rev()
                            .find(|&j| toks[j - 2] == ng[0] && toks[j - 1] == ng[1])
                        {
                            for d in 0..lookahead.min(toks.len() - j) {
                                draft.push(toks[j + d]);
                            }
                        }
                        if !draft.is_empty() && out.len() + draft.len() + 1 <= ngen {
                            let mut batch = Vec::with_capacity(draft.len() + 1);
                            batch.push(tok);
                            batch.extend_from_slice(&draft);
                            let seq1 = vec![0usize; batch.len()];
                            let posv: Vec<usize> = (pos..pos + batch.len()).collect();
                            let rows: Vec<usize> = (0..batch.len()).collect();
                            let lg = r.forward_multi(&batch, &seq1, &posv, std::slice::from_mut(&mut cache), &rows)
                                .to_vec();
                            // Verify sequentially. Row i predicts the token that
                            // follows batch[i], so rows 0..draft.len()-1 check
                            // draft[0..], and the LAST row is the continuation
                            // after a fully accepted draft.
                            //
                            // That last row is what the first version of this
                            // block threw away, and it is why a fully accepted
                            // draft re-emitted its own last token: `next` was
                            // left holding the token just accepted. Together
                            // with an out.pop()/out.push(batch[0]) pair that
                            // deleted an accepted token and duplicated `tok`,
                            // the path produced 1403 tokens where the exact
                            // path produced 940 -- and every one of the 25
                            // outputs differed from baseline. Speculation is
                            // only worth anything if it is indistinguishable
                            // from not speculating, so that difference is the
                            // whole test.
                            let mut acc = 0usize;
                            let mut v = lg[0..vsz].to_vec();
                            if let Some(c) = con.as_ref() { c.apply(&mut v); }
                            let mut next = argmax(&v) as u32;
                            // EOS is never emitted into `out` on the ordinary
                            // path -- the loop breaks on it first -- so an
                            // accepted EOS must not be pushed here either.
                            while acc < draft.len() && next == draft[acc] && !eos.contains(&next) {
                                out.push(next);
                                if let Some(c) = con.as_mut() { c.advance(next as usize); }
                                acc += 1;
                                let mut v = lg[acc * vsz..(acc + 1) * vsz].to_vec();
                                if let Some(c) = con.as_ref() { c.apply(&mut v); }
                                next = argmax(&v) as u32;
                            }
                            mtp_drafted += draft.len();
                            mtp_accepted += acc;
                            // `next` becomes the following iteration's `tok`,
                            // which the ordinary path would have advanced when
                            // it chose it, so advance it here unconditionally.
                            if let Some(c) = con.as_mut() { c.advance(next as usize); }
                            // KV past the accepted prefix holds rejected tokens;
                            // it is never read, because the next forward writes
                            // those same positions before any attention reads
                            // beyond `pos`.
                            pos += acc + 1;
                            tok = next;
                            continue;
                        }
                    }

                    // --- ordinary single-token step
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
                dec_secs += t_dec.elapsed().as_secs_f64();
                nprompt += toks.len();
                ngenerated += out.len();
                println!("{}", out.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
            }
            let el = t_all.elapsed().as_secs_f64();
            eprintln!("{} prompts, {} prompt tok, {} generated tok in {:.2}s ({:.1} gen tok/s)",
                      lines.len(), nprompt, ngenerated, el, ngenerated as f64 / el);
            eprintln!("decode only: {ngenerated} tok in {dec_secs:.2}s ({:.2} tok/s), \
                       prefill {} tok in {:.2}s ({:.1} tok/s)",
                      ngenerated as f64 / dec_secs.max(1e-9), nprompt, el - dec_secs,
                      nprompt as f64 / (el - dec_secs).max(1e-9));
            if con.is_some() {
                eprintln!("forced steps (LM head skipped): {forced_steps}/{ngenerated} = {:.0}%",
                          100.0 * forced_steps as f64 / ngenerated.max(1) as f64);
            }
            if mtp_batched > 0 {
                eprintln!("MTP grammar: {mtp_batched} decode steps collapsed by batching forced runs");
            }
            if mtp_drafted > 0 {
                eprintln!("MTP lookup: {mtp_accepted}/{mtp_drafted} drafted tokens accepted = {:.0}%",
                          100.0 * mtp_accepted as f64 / mtp_drafted as f64);
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

            let prefill = |r: &mut Runner, c: &mut KvCache, t: &[u32], base: usize| -> u32 {
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
    verify_clean();
}
