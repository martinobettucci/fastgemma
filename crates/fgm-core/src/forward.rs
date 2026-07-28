//! Gemma 4 text forward pass.
//!
//! Per layer:
//!     h += post_attention_layernorm( attn( input_layernorm(h) ) )
//!     h += post_feedforward_layernorm( mlp( pre_feedforward_layernorm(h) ) )
//!     h += post_per_layer_input_norm( ple_proj( gelu(ple_gate(h)) * ple[l] ) )
//!     h *= layer_scalar
//!
//! Attention scaling is 1.0 (q_norm carries the magnitude), KV-shared layers own
//! no k/v weights and read their source layer's cache, and full-attention layers
//! use head_dim 512 against the sliding layers' 256.
//!
//! Every linear was Hadamard-rotated along K by the converter, so activations
//! get the matching FWHT before quantising to int8 — that is what keeps the
//! per-token activation scale off the outliers.
//!
//! All weights are resolved once into `LayerW` and all scratch is preallocated:
//! the hot path does no string formatting, no map lookups and no allocation.

use std::time::Instant;

use crate::kv::KvCache;
use crate::model::{Config, Model, QLinear};
use crate::pool::{AttnJob, GemmJob, Pool, PrepJob, RowRef};
use fgm_kernels as k;

/// A weight the file may carry in two quantisations.
///
/// Prefill and decode are different problems and want different weights. A
/// prefill GEMM has hundreds of rows sharing one weight read, so it is
/// compute-bound, and int4's per-group accumulator drain (every `group/64` tile
/// steps) is the binding cost. A decode GEMM has one row per sequence, so the
/// weight read itself is the cost and int4 halves the bytes. Measured
/// standalone, group 256 int4 runs 2.19 TOPS against int8's ~3.9; measured at
/// batch 1, int4 wins because nothing amortises the read.
///
/// `--weights=both` at conversion stores both (+~1.56 GB for E2B: an int8 twin
/// is twice the int4 bytes) and the runtime picks per GEMM from the row count.
pub struct DualW<'a> {
    lo: QLinear<'a>,
    hi: Option<QLinear<'a>>,
}

/// Which weight a GEMM should use, as a row-count threshold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WeightSel {
    /// always the file's primary (int4 where the converter wrote int4)
    Lo,
    /// always the int8 twin where one exists
    Hi,
    /// int8 once a GEMM has at least this many rows
    Above(usize),
}

impl WeightSel {
    /// `FGM_WEIGHTS=int4|int8|auto[:M]`, default `auto:16`.
    ///
    /// 16 is the AMX tile height: below it a GEMM cannot fill one tile of rows,
    /// so it is decode-shaped no matter what the caller calls it.
    pub fn from_env() -> Self {
        match std::env::var("FGM_WEIGHTS").as_deref() {
            Ok("int4") => WeightSel::Lo,
            Ok("int8") => WeightSel::Hi,
            Ok(s) if s.starts_with("auto:") => {
                WeightSel::Above(s[5..].parse().unwrap_or(16))
            }
            _ => WeightSel::Above(16),
        }
    }
}

impl<'a> DualW<'a> {
    pub fn new(lo: QLinear<'a>, hi: Option<QLinear<'a>>) -> Self {
        DualW { lo, hi }
    }

    #[inline]
    fn pick(&self, m: usize, sel: WeightSel) -> &QLinear<'a> {
        match (&self.hi, sel) {
            (Some(h), WeightSel::Hi) => h,
            (Some(h), WeightSel::Above(t)) if m >= t => h,
            _ => &self.lo,
        }
    }

    pub fn has_hi(&self) -> bool {
        self.hi.is_some()
    }
}

/// Weights for one decoder layer, resolved once at construction.
pub struct LayerW<'m> {
    pub input_ln: &'m [f32],
    pub post_attn_ln: &'m [f32],
    pub pre_ffn_ln: &'m [f32],
    pub post_ffn_ln: &'m [f32],
    pub post_ple_ln: &'m [f32],
    pub q_norm: &'m [f32],
    pub k_norm: Option<&'m [f32]>,
    pub layer_scalar: f32,
    pub q: DualW<'m>,
    pub o: DualW<'m>,
    pub kp: Option<DualW<'m>>,
    pub vp: Option<DualW<'m>>,
    pub gate: DualW<'m>,
    pub up: DualW<'m>,
    pub down: DualW<'m>,
    pub ple_gate: QLinear<'m>,
    pub ple_proj: QLinear<'m>,
    pub head_dim: usize,
    pub inter: usize,
    pub sliding: bool,
    pub shared: bool,
    pub kv_src: usize,
}

/// Scratch for the GEMM wrapper, kept separate so it can be borrowed disjointly
/// from the activation buffers.
struct Gemm {
    rot: Vec<f32>,
    qa: Vec<i8>,
    qs: Vec<f32>,
    pa: Vec<i8>,
}

impl Gemm {
    /// `out[m, 0..w.n) = w . a[m, 0..w.k)`: rotate, quantise, tile-pack, dispatch.
    fn run(&mut self, pool: &Pool, w: &QLinear, a: &[f32], m: usize, lda: usize, out: &mut [f32]) {
        let (kk, n) = (w.k, w.n);
        // Rotate + quantise + tile-pack, parallel over 16-row tile blocks. This
        // used to run single-threaded while the GEMM below it did not.
        pool.prep(PrepJob {
            src: a.as_ptr(),
            lda,
            rot: self.rot.as_mut_ptr(),
            qa: self.qa.as_mut_ptr(),
            qs: self.qs.as_mut_ptr(),
            pa: self.pa.as_mut_ptr(),
            m,
            k: kk,
            hsz: w.hadamard,
        });
        pool.gemm(GemmJob {
            m,
            n,
            kdim: kk,
            a: self.pa.as_ptr(),
            a_scale: self.qs.as_ptr(),
            bq: w.q4.as_ptr(),
            b8: w.q8.as_ptr(),
            s4: w.s4.as_ptr(),
            s8: w.s8.as_ptr(),
            bits: w.bits,
            group: if w.group == 0 { 64 } else { w.group },
            c: out.as_mut_ptr(),
            ldc: n,
        });
    }
}

/// Preallocated activation buffers.
struct Buf {
    h: Vec<f32>,
    xn: Vec<f32>,
    q: Vec<f32>,
    kbuf: Vec<f32>,
    vbuf: Vec<f32>,
    ao: Vec<f32>,
    proj: Vec<f32>,
    g: Vec<f32>,
    u: Vec<f32>,
    act: Vec<f32>,
    p: Vec<f32>,
    pout: Vec<f32>,
    ple: Vec<f32>,
    ple_raw: Vec<f32>,
    ple_proj: Vec<f32>,
    tmp: Vec<f32>,
    /// one quantised KV row, staged before the transposed scatter into V
    vrow: Vec<i8>,
    logits: Vec<f32>,
}

pub struct Runner<'m> {
    pub model: &'m Model,
    pub cfg: Config,
    pub pool: Pool,
    layers: Vec<LayerW<'m>>,
    lm_head: QLinear<'m>,
    ple_model_proj: QLinear<'m>,
    ple_norm: &'m [f32],
    final_norm: &'m [f32],
    inv_full: &'m [f32],
    inv_slide: &'m [f32],
    gm: Gemm,
    b: Buf,
    dump: Vec<f32>,
    /// Per-row KV view, rebuilt each layer (cheap: m entries).
    rows: Vec<RowRef>,
    /// which of a dual-format weight's two quantisations each GEMM uses
    pub wsel: WeightSel,
    /// FGM_ROWBATCH=0|1 overrides the head-batching shape rule, for A/B
    rowbatch_force: Option<bool>,
    /// FGM_KV4=1: simulate a 4-bit KV cache to measure its accuracy cost
    kv4: bool,
    /// Per-phase seconds, accumulated when FGM_PROFILE is set.
    pub prof: [f64; NPHASE],
    profiling: bool,
}

pub const NPHASE: usize = 10;
pub const PHASE_NAMES: [&str; NPHASE] = [
    "embed", "ple", "norms", "qkv_gemm", "qk_norm_rope", "attention",
    "o_gemm", "ffn_gemm", "ple_inject", "lm_head",
];

/// Wall-clock helpers. These accumulate into a local array so they compose with
/// the disjoint `&mut self.gm` / `&mut self.b` borrows in the hot loop.
macro_rules! tick {
    ($on:expr) => {
        if $on { Some(Instant::now()) } else { None }
    };
}
macro_rules! tock {
    ($t:expr, $prof:expr, $i:expr) => {
        if let Some(t) = $t {
            $prof[$i] += t.elapsed().as_secs_f64();
        }
    };
}

impl<'m> Runner<'m> {
    pub fn new(model: &'m Model, max_tokens: usize, max_ctx: usize, threads: usize) -> Self {
        // Prefill only needs the last row's logits; batched decode needs one per
        // sequence. Validation asks for all of them, so cap and assert rather
        // than let a large request scribble past the buffer.
        Self::with_logit_rows(model, max_tokens, max_ctx, threads, max_tokens.min(16))
    }

    pub fn with_logit_rows(
        model: &'m Model, max_tokens: usize, max_ctx: usize, threads: usize,
        max_logit_rows: usize,
    ) -> Self {
        assert!(k::amx_init(), "AMX XTILEDATA permission denied");
        // Warm-up: only pay for the corruption guard on platforms that need it.
        let (guard, bad, hold) = k::tile_guard_autodetect(12);
        if guard && hold > 0 {
            eprintln!(
                "fastgemma: AMX tile state NOT preserved across context switches \
                 ({bad}/12 trials corrupted at {}us hold) -- guard ENABLED (~1-2%)",
                hold
            );
        } else if !guard {
            eprintln!("fastgemma: AMX tile state verified across preemption -- guard disabled");
        }
        let cfg = model.cfg.clone();
        let (hs, nl) = (cfg.hidden_size, cfg.num_hidden_layers);
        let pd = cfg.hidden_size_per_layer_input;

        let mut layers = Vec::with_capacity(nl);
        for l in 0..nl {
            let shared = cfg.is_shared(l);
            layers.push(LayerW {
                input_ln: model.f32s(&format!("l{l}.input_layernorm")),
                post_attn_ln: model.f32s(&format!("l{l}.post_attention_layernorm")),
                pre_ffn_ln: model.f32s(&format!("l{l}.pre_feedforward_layernorm")),
                post_ffn_ln: model.f32s(&format!("l{l}.post_feedforward_layernorm")),
                post_ple_ln: model.f32s(&format!("l{l}.post_per_layer_input_norm")),
                q_norm: model.f32s(&format!("l{l}.q_norm")),
                k_norm: (!shared).then(|| model.f32s(&format!("l{l}.k_norm"))),
                layer_scalar: model.f32s(&format!("l{l}.layer_scalar"))[0],
                q: model.dual(&format!("l{l}.q_proj")),
                o: model.dual(&format!("l{l}.o_proj")),
                kp: (!shared).then(|| model.dual(&format!("l{l}.k_proj"))),
                vp: (!shared).then(|| model.dual(&format!("l{l}.v_proj"))),
                gate: model.dual(&format!("l{l}.gate_proj")),
                up: model.dual(&format!("l{l}.up_proj")),
                down: model.dual(&format!("l{l}.down_proj")),
                ple_gate: model.linear(&format!("l{l}.per_layer_input_gate")),
                ple_proj: model.linear(&format!("l{l}.per_layer_projection")),
                head_dim: cfg.head_dim_of(l),
                inter: cfg.inter_of(l),
                sliding: cfg.is_sliding(l),
                shared,
                kv_src: cfg.kv_source(l),
            });
        }

        let nh = cfg.num_attention_heads;
        let kvh = cfg.num_key_value_heads;
        let hdmax = cfg.head_dim.max(cfg.global_head_dim);
        let imax = (0..nl).map(|l| cfg.inter_of(l)).max().unwrap();
        let kmax = imax.max(hs).max(nh * hdmax);
        let m = max_tokens;

        Runner {
            lm_head: model.linear("lm_head"),
            ple_model_proj: model.linear("per_layer_model_projection"),
            ple_norm: model.f32s("per_layer_projection_norm"),
            final_norm: model.f32s("norm"),
            inv_full: model.f32s("rope.full_attention.inv_freq"),
            inv_slide: model.f32s("rope.sliding_attention.inv_freq"),
            gm: Gemm {
                rot: vec![0.0; m * kmax],
                qa: vec![0; m * kmax],
                qs: vec![0.0; m + 16],
                pa: vec![0; k::packed_a_len(m, kmax)],
            },
            b: Buf {
                h: vec![0.0; m * hs],
                xn: vec![0.0; m * hs],
                q: vec![0.0; m * nh * hdmax],
                kbuf: vec![0.0; m * kvh * hdmax],
                vbuf: vec![0.0; m * kvh * hdmax],
                ao: vec![0.0; m * nh * hdmax],
                proj: vec![0.0; m * hs],
                g: vec![0.0; m * imax],
                u: vec![0.0; m * imax],
                act: vec![0.0; m * imax],
                p: vec![0.0; m * pd],
                pout: vec![0.0; m * hs],
                ple: vec![0.0; m * nl * pd],
                ple_raw: vec![0.0; nl * pd],
                ple_proj: vec![0.0; nl * pd],
                tmp: vec![0.0; hs.max(kmax)],
                vrow: vec![0; kvh * hdmax],
                logits: vec![0.0; max_logit_rows * cfg.vocab_size],
            },
            pool: Pool::new(threads, k::attend_scratch(max_ctx + 64, hdmax)
                    .max(k::rowbatch_scratch(nh, max_ctx + 64, hdmax))),
            layers,
            model,
            cfg,
            dump: Vec::new(),
            rows: vec![
                RowRef {
                    kc: std::ptr::null(), ks: std::ptr::null(),
                    vc: std::ptr::null(), vs: std::ptr::null(),
                    k_len: 0, pos: 0, ring: 0,
                };
                m
            ],
            wsel: WeightSel::from_env(),
            rowbatch_force: std::env::var("FGM_ROWBATCH").ok().map(|v| v != "0"),
            kv4: std::env::var_os("FGM_KV4").is_some(),
            prof: [0.0; NPHASE],
            profiling: std::env::var_os("FGM_PROFILE").is_some(),
        }
    }

    /// Single-sequence convenience wrapper: contiguous positions, one cache,
    /// logits for the final token only.
    pub fn forward(&mut self, tokens: &[u32], pos0: usize, kv: &mut KvCache) -> &[f32] {
        let m = tokens.len();
        // Local, not scratch fields: taking them out of `self` and putting them
        // back around the call meant reconstructing the returned slice from a
        // raw pointer to dodge the borrow checker. m is at most a prefill chunk,
        // so two small Vecs per call are free next to the forward pass itself.
        let seq = vec![0usize; m];
        let pos: Vec<usize> = (pos0..pos0 + m).collect();
        self.forward_multi(tokens, &seq, &pos, std::slice::from_mut(kv), &[m - 1])
    }

    /// Forward that advances the KV cache but computes no logits.
    ///
    /// When a grammar constraint admits exactly one token, the next token is
    /// already known and the 201 MB int4 LM-head read produces an answer nobody
    /// reads. The rest of the layer stack still has to run, because the cache
    /// must carry this position. On a 12-tool schema that is 42% of the steps
    /// inside a tool call.
    pub fn forward_nolm(&mut self, tokens: &[u32], pos0: usize, kv: &mut KvCache) {
        let m = tokens.len();
        let seq = vec![0usize; m];
        let pos: Vec<usize> = (pos0..pos0 + m).collect();
        self.forward_multi(tokens, &seq, &pos, std::slice::from_mut(kv), &[]);
    }

    /// Batched forward. Row `r` carries `tokens[r]` for sequence `seq[r]` at
    /// absolute position `pos[r]`. Every GEMM sees all rows at once, which is
    /// the whole point: at batch 8 the weight traffic is amortised 8 ways while
    /// attention stays per-sequence.
    ///
    /// Returns `logit_rows.len() * vocab_size` logits, row-major.
    pub fn forward_multi(
        &mut self,
        tokens: &[u32],
        seq: &[usize],
        pos: &[usize],
        caches: &mut [KvCache],
        logit_rows: &[usize],
    ) -> &[f32] {
        assert!(
            logit_rows.len() * self.cfg.vocab_size <= self.b.logits.len(),
            "requested {} logit rows but the runner was built for {}",
            logit_rows.len(),
            self.b.logits.len() / self.cfg.vocab_size
        );
        let model = self.model;
        let cfg = &self.cfg;
        let wsel = self.wsel;
        let (hs, m) = (cfg.hidden_size, tokens.len());
        let (nl, pd) = (cfg.num_hidden_layers, cfg.hidden_size_per_layer_input);
        let (nh, kvh) = (cfg.num_attention_heads, cfg.num_key_value_heads);
        let eps = cfg.rms_norm_eps;
        let dumping = std::env::var_os("FGM_DUMP").is_some();
        let profiling = self.profiling;
        let mut prof = [0.0f64; NPHASE];
        // Diagnostic for the non-determinism defect: if zeroing scratch makes
        // two identical calls agree, some buffer is read before it is written.
        if let Ok(z) = std::env::var("FGM_ZERO") {
            let b = &mut self.b;
            for (name, buf) in [
                ("h", &mut b.h), ("xn", &mut b.xn), ("q", &mut b.q),
                ("kbuf", &mut b.kbuf), ("vbuf", &mut b.vbuf), ("ao", &mut b.ao),
                ("proj", &mut b.proj), ("g", &mut b.g), ("u", &mut b.u),
                ("act", &mut b.act), ("p", &mut b.p), ("pout", &mut b.pout),
                ("ple", &mut b.ple), ("ple_raw", &mut b.ple_raw),
                ("ple_proj", &mut b.ple_proj), ("tmp", &mut b.tmp),
                ("logits", &mut b.logits),
            ] {
                if z == "all" || z == name {
                    buf.fill(0.0);
                }
            }
            if z == "all" || z == "gm" {
                self.gm.rot.fill(0.0);
                self.gm.qs.fill(0.0);
                self.gm.qa.fill(0);
                self.gm.pa.fill(0);
            }
        }
        if dumping {
            self.dump.clear();
        }

        let _t = tick!(profiling);
        for (r, &t) in tokens.iter().enumerate() {
            model.gather("embed_tokens", t as usize, &mut self.b.h[r * hs..(r + 1) * hs]);
            k::scale(&mut self.b.h[r * hs..(r + 1) * hs], cfg.embed_scale);
        }
        tock!(_t, prof, 0);

        // ple = (rmsnorm(proj(x) * H^-0.5) + table[tok] * sqrt(pd)) * 2^-0.5
        let _t = tick!(profiling);
        {
            let (gm, b) = (&mut self.gm, &mut self.b);
            gm.run(&self.pool, &self.ple_model_proj, &b.h, m, hs, &mut b.ple);
            for r in 0..m {
                b.ple_proj[..nl * pd].copy_from_slice(&b.ple[r * nl * pd..(r + 1) * nl * pd]);
                k::scale(&mut b.ple_proj[..nl * pd], cfg.ple_model_projection_scale);
                model.gather("embed_tokens_per_layer", tokens[r] as usize, &mut b.ple_raw);
                k::scale(&mut b.ple_raw[..nl * pd], cfg.ple_embed_scale);
                for li in 0..nl {
                    k::rmsnorm(&mut b.tmp[..pd], &b.ple_proj[li * pd..(li + 1) * pd],
                               self.ple_norm, eps);
                    let base = r * nl * pd + li * pd;
                    for j in 0..pd {
                        b.ple[base + j] = (b.tmp[j] + b.ple_raw[li * pd + j]) * cfg.ple_input_scale;
                    }
                }
            }
        }
        tock!(_t, prof, 1);

        for l in 0..nl {
            let w = &self.layers[l];
            let hd = w.head_dim;
            let inv = if w.sliding { self.inv_slide } else { self.inv_full };
            let (gm, b) = (&mut self.gm, &mut self.b);

            // ---- attention
            let _t = tick!(profiling);
            for r in 0..m {
                let (a, z) = (r * hs, (r + 1) * hs);
                b.tmp[..hs].copy_from_slice(&b.h[a..z]);
                k::rmsnorm(&mut b.xn[a..z], &b.tmp[..hs], w.input_ln, eps);
            }
            tock!(_t, prof, 2);
            let _t = tick!(profiling);
            gm.run(&self.pool, w.q.pick(m, wsel), &b.xn, m, hs, &mut b.q);
            tock!(_t, prof, 3);
            let _t = tick!(profiling);
            for r in 0..m {
                for h in 0..nh {
                    let o = r * nh * hd + h * hd;
                    b.tmp[..hd].copy_from_slice(&b.q[o..o + hd]);
                    k::rmsnorm(&mut b.q[o..o + hd], &b.tmp[..hd], w.q_norm, eps);
                }
                k::rope(&mut b.q[r * nh * hd..(r + 1) * nh * hd], nh, hd, inv, pos[r]);
            }
            tock!(_t, prof, 4);

            if !w.shared {
                let _t = tick!(profiling);
                gm.run(&self.pool, w.kp.as_ref().unwrap().pick(m, wsel), &b.xn, m, hs, &mut b.kbuf);
                gm.run(&self.pool, w.vp.as_ref().unwrap().pick(m, wsel), &b.xn, m, hs, &mut b.vbuf);
                tock!(_t, prof, 3);
                let _t = tick!(profiling);
                let rl = caches[0].layers[l].as_ref().unwrap().row_len();
                for r in 0..m {
                    let lk = caches[seq[r]].layers[l].as_mut().unwrap();
                    for h in 0..kvh {
                        let o = r * kvh * hd + h * hd;
                        b.tmp[..hd].copy_from_slice(&b.kbuf[o..o + hd]);
                        k::rmsnorm(&mut b.kbuf[o..o + hd], &b.tmp[..hd], w.k_norm.unwrap(), eps);
                        b.tmp[..hd].copy_from_slice(&b.vbuf[o..o + hd]);
                        k::rmsnorm_noscale(&mut b.vbuf[o..o + hd], &b.tmp[..hd], hd, eps);
                    }
                    k::rope(&mut b.kbuf[r * kvh * hd..(r + 1) * kvh * hd], kvh, hd, inv, pos[r]);
                    let slot = lk.slot(pos[r]);
                    let mut s1 = [0.0f32; 1];
                    k::quant_act(&b.kbuf[r * rl..(r + 1) * rl], 1, rl,
                                 &mut lk.k[slot * rl..(slot + 1) * rl], &mut s1);
                    // FGM_KV4=1 simulates a 4-bit KV cache by collapsing the
                    // int8 values onto 15 levels, keeping the same per-row
                    // scale and the same fast paths. It moves no fewer bytes,
                    // so it measures ONLY the accuracy cost -- which is the
                    // half worth knowing first, because a real int4 cache is
                    // nibble packing plus an unpack before every dpbusd, and
                    // there is no point building that to find the answers are
                    // wrong. The attention ceiling probe says we are
                    // bandwidth-bound at 1 MAC/byte, so halving KV bytes is
                    // the largest remaining lever -- if it survives the gate.
                    if self.kv4 {
                        for v in &mut lk.k[slot * rl..(slot + 1) * rl] {
                            *v = ((*v as i32 * 7 + 64) / 127).clamp(-7, 7) as i8 * 18;
                        }
                    }
                    lk.ks[slot] = s1[0];
                    // V goes into the cache transposed, so P.V can reduce over
                    // positions with an integer dot product. Quantise into a
                    // scratch row first, then scatter.
                    k::quant_act(&b.vbuf[r * rl..(r + 1) * rl], 1, rl,
                                 &mut b.vrow[..rl], &mut s1);
                    if self.kv4 {
                        for v in &mut b.vrow[..rl] {
                            *v = ((*v as i32 * 7 + 64) / 127).clamp(-7, 7) as i8 * 18;
                        }
                    }
                    k::store_v_t(&mut lk.v, &b.vrow[..rl], rl, slot);
                    lk.vs[slot] = s1[0];
                }
                tock!(_t, prof, 4);
            }

            let _t = tick!(profiling);
            let span = pos[..m].iter().copied().max().unwrap_or(0) + 1;
            for r in 0..m {
                let lk = caches[seq[r]].layers[w.kv_src].as_ref().unwrap();
                debug_assert!(
                    !lk.ring || m + cfg.sliding_window <= lk.capacity,
                    "ring cache holds {} but a batch of {m} rows with window {} needs {}",
                    lk.capacity, cfg.sliding_window, m + cfg.sliding_window
                );
                let ring = if lk.ring { lk.capacity } else { 0 };
                self.rows[r] = RowRef {
                    kc: lk.k.as_ptr(),
                    ks: lk.ks.as_ptr(),
                    vc: lk.v.as_ptr(),
                    vs: lk.vs.as_ptr(),
                    k_len: lk.capacity,
                    pos: pos[r],
                    ring,
                };
            }
            self.pool.attn(AttnJob {
                out: b.ao.as_mut_ptr(),
                q: b.q.as_ptr(),
                rows: self.rows.as_ptr(),
                nh,
                kvh,
                hd,
                m,
                window: if w.sliding { cfg.sliding_window } else { 0 },
                // Head-batching pays only once the K set for this layer exceeds
                // L2; below that the per-head kernel's four-position reduction
                // amortisation wins. It also needs enough rows to split on,
                // since one thread owns all heads of a row -- at decode (m=1)
                // that would serialise attention entirely.
                // FGM_ROWBATCH=0|1 forces the path for A/B. Same binary, both
                // arms, alternating -- the only design that survives this
                // project's confounds: AMX corruption drift within a host
                // (Trap 18) and, twice now, the host itself changing
                // mid-session (2.10 -> 2.80 no-AMX -> 2.30 GHz).
                rowbatch: match self.rowbatch_force {
                    Some(v) => v && kvh == 1,
                    None => m >= 16
                        && k::rowbatch_worthwhile(
                        kvh,
                        hd,
                        if w.sliding { cfg.sliding_window.min(span) } else { span },
                    ),
                },
            });

            tock!(_t, prof, 5);

            let _t = tick!(profiling);
            gm.run(&self.pool, w.o.pick(m, wsel), &b.ao, m, nh * hd, &mut b.proj);
            for r in 0..m {
                let (a, z) = (r * hs, (r + 1) * hs);
                b.tmp[..hs].copy_from_slice(&b.proj[a..z]);
                k::rmsnorm(&mut b.proj[a..z], &b.tmp[..hs], w.post_attn_ln, eps);
            }
            k::add(&mut b.h[..m * hs], &b.proj[..m * hs]);
            tock!(_t, prof, 6);

            // ---- MLP
            let _t = tick!(profiling);
            for r in 0..m {
                let (a, z) = (r * hs, (r + 1) * hs);
                b.tmp[..hs].copy_from_slice(&b.h[a..z]);
                k::rmsnorm(&mut b.xn[a..z], &b.tmp[..hs], w.pre_ffn_ln, eps);
            }
            gm.run(&self.pool, w.gate.pick(m, wsel), &b.xn, m, hs, &mut b.g);
            gm.run(&self.pool, w.up.pick(m, wsel), &b.xn, m, hs, &mut b.u);
            k::gelu_mul(&mut b.act, &b.g, &b.u, m * w.inter);
            gm.run(&self.pool, w.down.pick(m, wsel), &b.act, m, w.inter, &mut b.proj);
            for r in 0..m {
                let (a, z) = (r * hs, (r + 1) * hs);
                b.tmp[..hs].copy_from_slice(&b.proj[a..z]);
                k::rmsnorm(&mut b.proj[a..z], &b.tmp[..hs], w.post_ffn_ln, eps);
            }
            k::add(&mut b.h[..m * hs], &b.proj[..m * hs]);
            tock!(_t, prof, 7);

            // ---- per-layer input injection
            let _t = tick!(profiling);
            gm.run(&self.pool, &w.ple_gate, &b.h, m, hs, &mut b.p);
            k::gelu(&mut b.p[..m * pd]);
            for r in 0..m {
                let off = r * nl * pd + l * pd;
                for j in 0..pd {
                    b.p[r * pd + j] *= b.ple[off + j];
                }
            }
            gm.run(&self.pool, &w.ple_proj, &b.p, m, pd, &mut b.pout);
            for r in 0..m {
                let (a, z) = (r * hs, (r + 1) * hs);
                b.tmp[..hs].copy_from_slice(&b.pout[a..z]);
                k::rmsnorm(&mut b.pout[a..z], &b.tmp[..hs], w.post_ple_ln, eps);
            }
            k::add(&mut b.h[..m * hs], &b.pout[..m * hs]);
            k::scale(&mut b.h[..m * hs], w.layer_scalar);
            tock!(_t, prof, 8);

            if dumping {
                self.dump.extend_from_slice(&self.b.h[..m * hs]);
            }
        }

        for r in 0..m {
            let c = &mut caches[seq[r]];
            c.len = c.len.max(pos[r] + 1);
        }

        // ---- final norm, then LM head for the requested rows.
        // Batching the head is nearly free: it is a 201 MB int4 read that does
        // not grow with m, so 8 rows cost about what 1 row costs.
        let _t = tick!(profiling);
        {
            let (gm, b) = (&mut self.gm, &mut self.b);
            let norm_all = dumping;
            for r in 0..m {
                if !norm_all && !logit_rows.contains(&r) {
                    continue;
                }
                let (a, z) = (r * hs, (r + 1) * hs);
                b.tmp[..hs].copy_from_slice(&b.h[a..z]);
                k::rmsnorm(&mut b.h[a..z], &b.tmp[..hs], self.final_norm, eps);
            }
            if dumping {
                self.dump.extend_from_slice(&b.h[..m * hs]);
            }
            let nr = logit_rows.len();
            for (i, &r) in logit_rows.iter().enumerate() {
                b.xn[i * hs..(i + 1) * hs].copy_from_slice(&b.h[r * hs..(r + 1) * hs]);
            }
            // nr == 0 is a deliberate caller request to skip the head entirely
            // (see `forward_nolm`), not a degenerate case to push through the
            // GEMM.
            if nr > 0 {
                gm.run(&self.pool, &self.lm_head, &b.xn, nr, hs, &mut b.logits);
            }
        }
        if let (Some(cap), false) = (cfg.final_logit_softcapping, logit_rows.is_empty()) {
            let n = logit_rows.len() * self.cfg.vocab_size;
            k::softcap(&mut self.b.logits[..n], cap);
        }
        tock!(_t, prof, 9);
        for i in 0..NPHASE {
            self.prof[i] += prof[i];
        }

        if dumping {
            if let Ok(path) = std::env::var("FGM_DUMP") {
                use std::io::Write;
                let mut f = std::fs::File::create(&path).expect("dump");
                f.write_all(&((nl + 1) as u32).to_le_bytes()).unwrap();
                f.write_all(&(m as u32).to_le_bytes()).unwrap();
                f.write_all(&(hs as u32).to_le_bytes()).unwrap();
                for v in &self.dump {
                    f.write_all(&v.to_le_bytes()).unwrap();
                }
                eprintln!("dumped {} entries -> {path}", self.dump.len() / (m * hs));
            }
        }
        &self.b.logits[..logit_rows.len() * self.cfg.vocab_size]
    }
}
