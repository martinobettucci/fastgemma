//! The batching scheduler: one thread owning the `Runner`, the KV slots, and
//! the decode loop.
//!
//! Why this exists at all: decode on this engine is **weight-read bound**, not
//! multiply bound. One decode step reads ~1.4 GB of int4 weights whether it is
//! computing one row or eight, and the GEMM kernel's minimum block is eight
//! rows regardless — so eight concurrent sequences cost almost exactly what one
//! costs. Serving requests one at a time throws that away and delivers the
//! batch-1 number, which is the *worst* number this engine produces.
//!
//! Shape of the loop:
//!
//!   admit  fill free slots from the queue, prefilling each on arrival
//!   step   one `forward_multi` across every active slot at its own position
//!   emit   per slot, decode the new token, apply stops, stream or retire
//!
//! Prefill is **not** batched across sequences and deliberately so: it is
//! compute-bound, already runs 256 rows per forward, and interleaving it would
//! buy throughput it does not need while making the scheduler much harder to
//! reason about. It does mean a request arriving mid-generation stalls the
//! decode of everyone already running for the length of its prefill; that is a
//! TTFT cost paid by the batch, and it is the honest trade for a scheduler this
//! size.

use fgm_core::{KvCache, Model, Runner};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};

use crate::tokenizer::Tokenizer;

pub enum Event {
    /// Newly decoded text, already past every stop check.
    Text(String),
    Done { finish: &'static str, prompt_tokens: usize, completion_tokens: usize },
    /// Reserved for failures the engine can report without dying. Nothing
    /// constructs it today -- the connection side treats a dropped sender as a
    /// dead engine, which covers the only failure mode that exists.
    #[allow(dead_code)]
    Error(String),
}

pub struct Job {
    pub toks: Vec<u32>,
    pub max_tokens: usize,
    pub stops: Vec<String>,
    pub tx: Sender<Event>,
}

struct Slot {
    job: Job,
    out_ids: Vec<u32>,
    text: String,
    /// bytes of `text` already handed to the client
    sent: usize,
    pos: usize,
    tok: u32,
}

/// Length of the prefix of `t` that will not change as more tokens arrive.
///
/// `decode` runs the whole emitted run through `from_utf8_lossy` each step, so
/// a multi-byte character still missing its tail shows up as a trailing
/// U+FFFD that a later step replaces with the real character. Streaming that
/// byte range would send a replacement character the client can never take
/// back, and would leave `sent` pointing past text that has since changed.
/// Hold the trailing run back instead; it is at most a few bytes and one step.
fn stable_len(t: &str) -> usize {
    let b = t.as_bytes();
    let mut n = b.len();
    while n >= 3 && &b[n - 3..n] == [0xEF, 0xBF, 0xBD] {
        n -= 3;
    }
    n
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

pub struct Config {
    pub ctx: usize,
    pub batch: usize,
    pub threads: usize,
    pub prefill_chunk: usize,
    /// Shortest shared token prefix worth caching. 0 disables prefix sharing.
    pub prefix_min: usize,
}

/// Length of the longest common leading run of two token sequences.
fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// A prefill result kept so the next request that starts with the same tokens
/// can fork it instead of recomputing it.
///
/// The whole cache is forked as a unit -- `kv.len == toks.len` -- because a
/// sliding-window layer that has wrapped stores its live positions in ring
/// order, so an arbitrary sub-prefix of it is not contiguous and cannot be
/// copied out. That is why reuse requires a job to start with *exactly* these
/// tokens, and why the template holds the shared prefix itself rather than some
/// longer prompt it was a prefix of.
struct Prefix {
    /// The shared tokens; a job reuses this only if it begins with all of them.
    toks: Vec<u32>,
    /// Prefilled to exactly `toks.len()` positions.
    kv: KvCache,
}

/// Run until the job channel is closed. Owns everything the forward pass needs,
/// which is why it is a thread rather than a lock: `Runner` holds the worker
/// pool and the scratch buffers, and there is exactly one of it.
pub fn run(model: &Model, tok: &Tokenizer, cfg: Config, rx: Receiver<Job>) {
    let mcfg = model.cfg.clone();
    let vocab = mcfg.vocab_size;
    let mut runner = Runner::with_logit_rows(
        model,
        cfg.prefill_chunk.max(cfg.batch),
        cfg.ctx,
        cfg.threads,
        cfg.batch.max(1),
    );
    let mut caches: Vec<KvCache> =
        (0..cfg.batch).map(|_| KvCache::new(&mcfg, cfg.ctx, cfg.prefill_chunk)).collect();
    eprintln!(
        "fastgemma: {} slots x {} ctx = {:.0} MB of KV",
        cfg.batch,
        cfg.ctx,
        caches.iter().map(|c| c.bytes()).sum::<usize>() as f64 / 1e6
    );

    let mut slots: Vec<Option<Slot>> = (0..cfg.batch).map(|_| None).collect();

    // Prefix sharing. In the target workload every request carries the same
    // 12-tool declaration block, so its KV is worth computing once and forking.
    // The template is (re)built lazily: when two consecutive prompts share a
    // prefix at least `prefix_min` long and nothing already covers it, that
    // prefix is prefilled once into `tmpl_kv` and reused by every later prompt
    // that starts with it. `last_toks` remembers the previous prompt so the
    // shared length can be discovered without the client declaring it.
    let mut prefix: Option<Prefix> = None;
    let mut last_toks: Vec<u32> = Vec::new();

    loop {
        // ---- admit -------------------------------------------------------
        loop {
            let Some(i) = slots.iter().position(|s| s.is_none()) else { break };
            let idle = slots.iter().all(|s| s.is_none());
            let job = if idle {
                // Nothing to decode, so block rather than spin.
                match rx.recv() {
                    Ok(j) => j,
                    Err(_) => return,
                }
            } else {
                match rx.try_recv() {
                    Ok(j) => j,
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => break,
                }
            };

            // Reuse the template when this prompt begins with exactly its
            // tokens. `<` not `<=`: leave at least one token to forward so the
            // first logits come from a real step rather than a stale cache row.
            let reuse = match &prefix {
                Some(p) if p.toks.len() < job.toks.len()
                    && job.toks[..p.toks.len()] == p.toks[..] => p.toks.len(),
                _ => 0,
            };

            caches[i].clear();
            let t0 = std::time::Instant::now();
            if reuse > 0 {
                caches[i].fork_from(&prefix.as_ref().unwrap().kv);
            }
            let mut off = reuse;
            let mut first = 0u32;
            while off < job.toks.len() {
                let n = cfg.prefill_chunk.min(job.toks.len() - off);
                let lg = runner.forward(&job.toks[off..off + n], off, &mut caches[i]);
                if off + n >= job.toks.len() {
                    first = argmax(lg) as u32;
                }
                off += n;
            }
            let dt = t0.elapsed().as_secs_f64();
            if reuse > 0 {
                eprintln!(
                    "  slot {i}: prefill {} tok ({} shared, {} new) in {:.2}s",
                    job.toks.len(), reuse, job.toks.len() - reuse, dt
                );
            } else {
                eprintln!(
                    "  slot {i}: prefill {} tok in {:.2}s ({:.1} tok/s)",
                    job.toks.len(), dt, job.toks.len() as f64 / dt.max(1e-9)
                );
            }

            // (Re)build the template when this prompt and the previous one
            // share a long enough prefix that the current template does not
            // already cover. One extra prefill of the shared span, amortised
            // over every subsequent request that reuses it.
            if cfg.prefix_min > 0 && reuse == 0 {
                // Floor the shared length to a whole prefill chunk. A full
                // prefill groups positions into GEMM calls at [0,C),[C,2C),...;
                // reusing a template of length T then chunking the suffix from T
                // only reproduces those exact groupings -- and therefore the
                // exact int8/int4 reduction order -- when T is a multiple of C.
                // At a ragged T the boundary chunk has a different row count, the
                // low-precision sums land a hair apart, and greedy occasionally
                // flips a token. Aligning T makes reuse bit-identical to a full
                // prefill rather than merely close.
                let raw = lcp(&job.toks, &last_toks).min(job.toks.len().saturating_sub(1));
                let share = raw / cfg.prefill_chunk * cfg.prefill_chunk;
                let covered = prefix.as_ref().map(|p| p.toks.len()).unwrap_or(0);
                if share >= cfg.prefix_min && share != covered {
                    let mut kv = KvCache::new(&mcfg, cfg.ctx, cfg.prefill_chunk);
                    let mut o = 0usize;
                    while o < share {
                        let n = cfg.prefill_chunk.min(share - o);
                        runner.forward(&job.toks[o..o + n], o, &mut kv);
                        o += n;
                    }
                    prefix = Some(Prefix { toks: job.toks[..share].to_vec(), kv });
                    eprintln!("  prefix cache: built {share}-token shared prefix");
                }
            }
            last_toks = job.toks.clone();

            let pos = job.toks.len();
            slots[i] = Some(Slot {
                job,
                out_ids: Vec::new(),
                text: String::new(),
                sent: 0,
                pos,
                tok: first,
            });
        }

        // ---- one batched decode step -------------------------------------
        let active: Vec<usize> = (0..slots.len()).filter(|&i| slots[i].is_some()).collect();
        if active.is_empty() {
            continue;
        }

        // Retire anything that finished on the token it is holding, before
        // spending a forward on it.
        let mut retire: Vec<(usize, &'static str)> = Vec::new();
        for &i in &active {
            let s = slots[i].as_mut().unwrap();
            if tok.eos.contains(&s.tok) {
                retire.push((i, "stop"));
                continue;
            }
            if s.out_ids.len() >= s.job.max_tokens {
                retire.push((i, "length"));
                continue;
            }
            s.out_ids.push(s.tok);
            // Decode the whole run each step rather than per token: one UTF-8
            // character can span several byte-fallback tokens, and decoding
            // them individually emits replacement characters mid-word.
            s.text = tok.decode(&s.out_ids, true);
            if let Some(cut) = s.job.stops.iter().filter_map(|p| s.text.find(p.as_str())).min() {
                s.text.truncate(cut);
                retire.push((i, "stop"));
                continue;
            }
            let stable = stable_len(&s.text);
            if stable > s.sent {
                let delta = s.text[s.sent..stable].to_string();
                s.sent = stable;
                let _ = s.job.tx.send(Event::Text(delta));
            }
        }
        for (i, finish) in retire {
            finish_slot(&mut slots, i, finish);
        }

        let active: Vec<usize> = (0..slots.len()).filter(|&i| slots[i].is_some()).collect();
        if active.is_empty() {
            continue;
        }

        let tokens: Vec<u32> = active.iter().map(|&i| slots[i].as_ref().unwrap().tok).collect();
        let pos: Vec<usize> = active.iter().map(|&i| slots[i].as_ref().unwrap().pos).collect();
        // `seq[r]` indexes `caches`, so a row's cache is its slot's cache.
        let seq: Vec<usize> = active.clone();
        let rows: Vec<usize> = (0..active.len()).collect();
        let lg = runner.forward_multi(&tokens, &seq, &pos, &mut caches, &rows);

        for (r, &i) in active.iter().enumerate() {
            let s = slots[i].as_mut().unwrap();
            s.tok = argmax(&lg[r * vocab..(r + 1) * vocab]) as u32;
            s.pos += 1;
            if s.pos >= cfg.ctx - 1 {
                finish_slot(&mut slots, i, "length");
            }
        }
    }
}

fn finish_slot(slots: &mut [Option<Slot>], i: usize, finish: &'static str) {
    let Some(s) = slots[i].take() else { return };
    // Flush whatever was held back for the UTF-8 boundary, or truncated off by
    // a stop string. The run is complete now, so nothing more can change.
    if s.text.len() > s.sent {
        let _ = s.job.tx.send(Event::Text(s.text[s.sent..].to_string()));
    }
    let _ = s.job.tx.send(Event::Done {
        finish,
        prompt_tokens: s.job.toks.len(),
        completion_tokens: s.out_ids.len(),
    });
}
