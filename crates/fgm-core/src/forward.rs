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
use crate::pool::{AttnJob, FaRow, GemmJob, Pool, PrepJob, RowRef};
use fgm_kernels as k;

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
    pub q: QLinear<'m>,
    pub o: QLinear<'m>,
    pub kp: Option<QLinear<'m>>,
    pub vp: Option<QLinear<'m>>,
    pub gate: QLinear<'m>,
    pub up: QLinear<'m>,
    pub down: QLinear<'m>,
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
    /// same rows in the blocked kernel's C layout
    fa: Vec<FaRow>,
    /// FGM_NO_BLOCKED_ATTN=1 forces the per-(row, head) path, for A/B testing
    no_blocked: bool,
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
                q: model.linear(&format!("l{l}.q_proj")),
                o: model.linear(&format!("l{l}.o_proj")),
                kp: (!shared).then(|| model.linear(&format!("l{l}.k_proj"))),
                vp: (!shared).then(|| model.linear(&format!("l{l}.v_proj"))),
                gate: model.linear(&format!("l{l}.gate_proj")),
                up: model.linear(&format!("l{l}.up_proj")),
                down: model.linear(&format!("l{l}.down_proj")),
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
                logits: vec![0.0; max_logit_rows * cfg.vocab_size],
            },
            pool: Pool::new(
                threads,
                (max_ctx + 64).max(k::blocked_scratch(nh, hdmax, kvh)),
            ),
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
            fa: vec![
                FaRow {
                    kc: std::ptr::null(), vc: std::ptr::null(),
                    ks: std::ptr::null(), vs: std::ptr::null(),
                    k_len: 0, pos: 0,
                };
                m
            ],
            no_blocked: std::env::var_os("FGM_NO_BLOCKED_ATTN").is_some(),
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
            gm.run(&self.pool, &w.q, &b.xn, m, hs, &mut b.q);
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
                gm.run(&self.pool, w.kp.as_ref().unwrap(), &b.xn, m, hs, &mut b.kbuf);
                gm.run(&self.pool, w.vp.as_ref().unwrap(), &b.xn, m, hs, &mut b.vbuf);
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
                    lk.ks[slot] = s1[0];
                    k::quant_act(&b.vbuf[r * rl..(r + 1) * rl], 1, rl,
                                 &mut lk.v[slot * rl..(slot + 1) * rl], &mut s1);
                    lk.vs[slot] = s1[0];
                }
                tock!(_t, prof, 4);
            }

            let _t = tick!(profiling);
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
                self.fa[r] = FaRow {
                    kc: lk.k.as_ptr(),
                    vc: lk.v.as_ptr(),
                    ks: lk.ks.as_ptr(),
                    vs: lk.vs.as_ptr(),
                    k_len: ring as i32,
                    pos: pos[r] as i32,
                };
            }
            // Blocking only pays when there are enough query rows to amortise a
            // K/V block; at m=1 (decode) the KV range is read once anyway and
            // row-splitting would leave threads idle, so keep the per-head path.
            //
            // It is also only *correct* when every row in a block reads the same
            // cache, because the block's K is dequantised once and shared. That
            // holds for chunked prefill (one sequence at a time) but not for
            // batched decode, where each row is a different sequence.
            let same_cache = seq[..m].iter().all(|&x| x == seq[0]);
            let blocked = m >= 16 && same_cache && !self.no_blocked;
            self.pool.attn(AttnJob {
                out: b.ao.as_mut_ptr(),
                q: b.q.as_ptr(),
                rows: self.rows.as_ptr(),
                fa: self.fa.as_ptr(),
                nh,
                kvh,
                hd,
                m,
                window: if w.sliding { cfg.sliding_window } else { 0 },
                blocked,
            });

            tock!(_t, prof, 5);

            let _t = tick!(profiling);
            gm.run(&self.pool, &w.o, &b.ao, m, nh * hd, &mut b.proj);
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
            gm.run(&self.pool, &w.gate, &b.xn, m, hs, &mut b.g);
            gm.run(&self.pool, &w.up, &b.xn, m, hs, &mut b.u);
            k::gelu_mul(&mut b.act, &b.g, &b.u, m * w.inter);
            gm.run(&self.pool, &w.down, &b.act, m, w.inter, &mut b.proj);
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
            gm.run(&self.pool, &self.lm_head, &b.xn, nr, hs, &mut b.logits);
        }
        if let Some(cap) = cfg.final_logit_softcapping {
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
