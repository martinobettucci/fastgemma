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

use crate::kv::KvCache;
use crate::model::{Config, Model, QLinear};
use fgm_kernels as k;

pub struct Scratch {
    pub h: Vec<f32>,      // [M, H] residual stream
    pub xn: Vec<f32>,     // [M, H] normed input
    pub rot: Vec<f32>,    // [M, Kmax] rotated activation
    pub qa: Vec<i8>,      // [M, Kmax] quantised activation
    pub qs: Vec<f32>,     // [M] activation scales
    pub pa: Vec<i8>,      // tile-packed activation
    pub y: Vec<f32>,      // [M, Nmax] GEMM output
    pub y2: Vec<f32>,     // [M, Nmax] second GEMM output (up_proj)
    pub attn: Vec<f32>,   // [M, nh*hd]
    pub kv: Vec<f32>,     // [kv_heads*hd]
    pub ple: Vec<f32>,    // [M, L, 256]
    pub ple_raw: Vec<f32>,// [L*256]
    pub sc: Vec<f32>,     // attention softmax scratch
    pub logits: Vec<f32>,
}

impl Scratch {
    pub fn new(cfg: &Config, max_tokens: usize, max_ctx: usize) -> Self {
        let h = cfg.hidden_size;
        let l = cfg.num_hidden_layers;
        let nmax = cfg
            .layer_types
            .iter()
            .enumerate()
            .map(|(i, _)| cfg.inter_of(i).max(cfg.num_attention_heads * cfg.head_dim_of(i)))
            .max()
            .unwrap()
            .max(l * cfg.hidden_size_per_layer_input);
        let kmax = nmax.max(h);
        let m = max_tokens;
        Scratch {
            h: vec![0.0; m * h],
            xn: vec![0.0; m * h],
            rot: vec![0.0; m * kmax],
            qa: vec![0; m * kmax],
            qs: vec![0.0; m + 16],
            pa: vec![0; k::packed_a_len(m, kmax)],
            y: vec![0.0; m * nmax],
            y2: vec![0.0; m * nmax],
            attn: vec![0.0; m * cfg.num_attention_heads * cfg.global_head_dim],
            kv: vec![0.0; cfg.num_key_value_heads * cfg.global_head_dim],
            ple: vec![0.0; m * l * cfg.hidden_size_per_layer_input],
            ple_raw: vec![0.0; l * cfg.hidden_size_per_layer_input],
            sc: vec![0.0; max_ctx + 64],
            logits: vec![0.0; cfg.vocab_size],
        }
    }
}

pub struct Runner<'m> {
    pub model: &'m Model,
    pub s: Scratch,
    inv_full: Vec<f32>,
    inv_slide: Vec<f32>,
    /// per-layer hidden states, captured only when FGM_DUMP is set
    dump: Vec<f32>,
}

impl<'m> Runner<'m> {
    pub fn new(model: &'m Model, max_tokens: usize, max_ctx: usize) -> Self {
        assert!(k::amx_init(), "AMX XTILEDATA permission denied");
        let s = Scratch::new(&model.cfg, max_tokens, max_ctx);
        Runner {
            inv_full: model.f32s("rope.full_attention.inv_freq").to_vec(),
            inv_slide: model.f32s("rope.sliding_attention.inv_freq").to_vec(),
            model,
            s,
            dump: Vec::new(),
        }
    }

    /// `out[m, 0..w.n) = w . a[m, 0..w.k)` for `m` rows.
    ///
    /// Applies the rotconv FWHT, quantises to int8 per row, tile-packs, then
    /// dispatches to the int4 or int8 AMX kernel.
    fn matmul(&mut self, w: &QLinear, a: &[f32], m: usize, lda: usize, out: &mut [f32]) {
        let (kk, n) = (w.k, w.n);
        for r in 0..m {
            self.s.rot[r * kk..r * kk + kk].copy_from_slice(&a[r * lda..r * lda + kk]);
        }
        if w.hadamard > 0 {
            k::fwht(&mut self.s.rot[..m * kk], w.hadamard);
        }
        k::quant_act(&self.s.rot[..m * kk], m, kk, &mut self.s.qa, &mut self.s.qs);
        k::pack_a(m, kk, &self.s.qa, &mut self.s.pa);
        let nb = n / 16;
        if w.bits == 4 {
            k::gemm_q4g(m, n, kk, &self.s.pa, &self.s.qs, w.q4, w.s4, w.group, out, n, 0, nb);
        } else {
            k::gemm_q8c(m, n, kk, &self.s.pa, &self.s.qs, w.q8, w.s8, out, n, 0, nb);
        }
    }

    /// Per-Layer Embedding inputs for one token:
    ///   ple = (rmsnorm(proj(x) * H^-0.5) + table[tok] * sqrt(256)) * 2^-0.5
    fn build_ple(&mut self, tok: u32, x_row: usize) {
        let cfg = &self.model.cfg;
        let (l, pd) = (cfg.num_hidden_layers, cfg.hidden_size_per_layer_input);
        self.model.gather("embed_tokens_per_layer", tok as usize, &mut self.s.ple_raw);
        k::scale(&mut self.s.ple_raw[..l * pd], cfg.ple_embed_scale);

        let w = self.model.linear("per_layer_model_projection");
        let hcopy: Vec<f32> = self.s.h[x_row * cfg.hidden_size..(x_row + 1) * cfg.hidden_size].to_vec();
        let mut proj = vec![0.0f32; l * pd];
        self.matmul(&w, &hcopy, 1, cfg.hidden_size, &mut proj);
        k::scale(&mut proj, cfg.ple_model_projection_scale);

        let norm = self.model.f32s("per_layer_projection_norm");
        let dst = &mut self.s.ple[x_row * l * pd..(x_row + 1) * l * pd];
        let mut tmp = vec![0.0f32; pd];
        for li in 0..l {
            k::rmsnorm(&mut tmp, &proj[li * pd..(li + 1) * pd], norm, cfg.rms_norm_eps);
            for j in 0..pd {
                dst[li * pd + j] = (tmp[j] + self.s.ple_raw[li * pd + j]) * cfg.ple_input_scale;
            }
        }
    }

    /// Run `tokens` at `positions` (contiguous, appended to `kv`).
    /// Returns logits for the final token.
    pub fn forward(&mut self, tokens: &[u32], pos0: usize, kv: &mut KvCache) -> &[f32] {
        self.dump.clear();
        let cfg = self.model.cfg.clone();
        let (hs, m) = (cfg.hidden_size, tokens.len());
        let eps = cfg.rms_norm_eps;

        // ---- embeddings + per-layer inputs
        for (r, &t) in tokens.iter().enumerate() {
            let mut row = vec![0.0f32; hs];
            self.model.gather("embed_tokens", t as usize, &mut row);
            k::scale(&mut row, cfg.embed_scale);
            self.s.h[r * hs..(r + 1) * hs].copy_from_slice(&row);
        }
        for (r, &t) in tokens.iter().enumerate() {
            self.build_ple(t, r);
        }

        let pd = cfg.hidden_size_per_layer_input;
        let nl = cfg.num_hidden_layers;

        for l in 0..nl {
            let hd = cfg.head_dim_of(l);
            let nh = cfg.num_attention_heads;
            let kvh = cfg.num_key_value_heads;
            let inv: Vec<f32> = if cfg.is_sliding(l) { self.inv_slide.clone() } else { self.inv_full.clone() };

            // ---- attention
            let iln = self.model.f32s(&format!("l{l}.input_layernorm")).to_vec();
            for r in 0..m {
                let (a, b) = (r * hs, (r + 1) * hs);
                let src: Vec<f32> = self.s.h[a..b].to_vec();
                k::rmsnorm(&mut self.s.xn[a..b], &src, &iln, eps);
            }

            let wq = self.model.linear(&format!("l{l}.q_proj"));
            let mut q = vec![0.0f32; m * nh * hd];
            let xn = self.s.xn.clone();
            self.matmul(&wq, &xn, m, hs, &mut q);
            let qn = self.model.f32s(&format!("l{l}.q_norm")).to_vec();
            let mut tmp = vec![0.0f32; hd];
            for r in 0..m {
                for h in 0..nh {
                    let o = r * nh * hd + h * hd;
                    k::rmsnorm(&mut tmp, &q[o..o + hd], &qn, eps);
                    q[o..o + hd].copy_from_slice(&tmp);
                }
                k::rope(&mut q[r * nh * hd..(r + 1) * nh * hd], nh, hd, &inv, pos0 + r);
            }

            // ---- k/v (only layers that own a cache)
            if !cfg.is_shared(l) {
                let wk = self.model.linear(&format!("l{l}.k_proj"));
                let wv = self.model.linear(&format!("l{l}.v_proj"));
                let mut kk = vec![0.0f32; m * kvh * hd];
                let mut vv = vec![0.0f32; m * kvh * hd];
                self.matmul(&wk, &xn, m, hs, &mut kk);
                self.matmul(&wv, &xn, m, hs, &mut vv);
                let kn = self.model.f32s(&format!("l{l}.k_norm")).to_vec();
                let lk = kv.layers[l].as_mut().unwrap();
                let rl = lk.row_len();
                for r in 0..m {
                    for h in 0..kvh {
                        let o = r * kvh * hd + h * hd;
                        k::rmsnorm(&mut tmp, &kk[o..o + hd], &kn, eps);
                        kk[o..o + hd].copy_from_slice(&tmp);
                        k::rmsnorm_noscale(&mut tmp, &vv[o..o + hd], hd, eps);
                        vv[o..o + hd].copy_from_slice(&tmp);
                    }
                    k::rope(&mut kk[r * kvh * hd..(r + 1) * kvh * hd], kvh, hd, &inv, pos0 + r);
                    let slot = lk.slot(pos0 + r);
                    let mut s1 = [0.0f32; 1];
                    k::quant_act(&kk[r * rl..(r + 1) * rl], 1, rl, &mut lk.k[slot * rl..(slot + 1) * rl], &mut s1);
                    lk.ks[slot] = s1[0];
                    k::quant_act(&vv[r * rl..(r + 1) * rl], 1, rl, &mut lk.v[slot * rl..(slot + 1) * rl], &mut s1);
                    lk.vs[slot] = s1[0];
                }
            }

            // ---- attend against the source layer's cache
            let src = cfg.kv_source(l);
            let lk = kv.layers[src].as_ref().unwrap();
            let mut ao = vec![0.0f32; m * nh * hd];
            for r in 0..m {
                let pos = pos0 + r;
                let (mut st, en) = kv.window(&cfg, l, pos);
                if lk.ring && en - st > lk.capacity {
                    st = en - lk.capacity;
                }
                debug_assert!(!lk.ring, "ring windows need slot-mapped attention");
                k::attend_q8(
                    &mut ao[r * nh * hd..(r + 1) * nh * hd],
                    &q[r * nh * hd..(r + 1) * nh * hd],
                    &lk.k, &lk.ks, &lk.v, &lk.vs,
                    nh, kvh, hd, st, en, &mut self.s.sc,
                );
            }

            let wo = self.model.linear(&format!("l{l}.o_proj"));
            let mut proj = vec![0.0f32; m * hs];
            self.matmul(&wo, &ao, m, nh * hd, &mut proj);
            let pan = self.model.f32s(&format!("l{l}.post_attention_layernorm")).to_vec();
            for r in 0..m {
                let (a, b) = (r * hs, (r + 1) * hs);
                let src: Vec<f32> = proj[a..b].to_vec();
                k::rmsnorm(&mut proj[a..b], &src, &pan, eps);
                let (hh, pp) = (&mut self.s.h[a..b], &proj[a..b]);
                k::add(hh, pp);
            }

            // ---- MLP
            let inter = cfg.inter_of(l);
            let pfn = self.model.f32s(&format!("l{l}.pre_feedforward_layernorm")).to_vec();
            for r in 0..m {
                let (a, b) = (r * hs, (r + 1) * hs);
                let src: Vec<f32> = self.s.h[a..b].to_vec();
                k::rmsnorm(&mut self.s.xn[a..b], &src, &pfn, eps);
            }
            let xn = self.s.xn.clone();
            let wg = self.model.linear(&format!("l{l}.gate_proj"));
            let wu = self.model.linear(&format!("l{l}.up_proj"));
            let mut g = vec![0.0f32; m * inter];
            let mut u = vec![0.0f32; m * inter];
            self.matmul(&wg, &xn, m, hs, &mut g);
            self.matmul(&wu, &xn, m, hs, &mut u);
            let mut act = vec![0.0f32; m * inter];
            k::gelu_mul(&mut act, &g, &u, m * inter);
            let wd = self.model.linear(&format!("l{l}.down_proj"));
            let mut d = vec![0.0f32; m * hs];
            self.matmul(&wd, &act, m, inter, &mut d);
            let pff = self.model.f32s(&format!("l{l}.post_feedforward_layernorm")).to_vec();
            for r in 0..m {
                let (a, b) = (r * hs, (r + 1) * hs);
                let src: Vec<f32> = d[a..b].to_vec();
                k::rmsnorm(&mut d[a..b], &src, &pff, eps);
                let (hh, dd) = (&mut self.s.h[a..b], &d[a..b]);
                k::add(hh, dd);
            }

            // ---- per-layer input injection
            let wgate = self.model.linear(&format!("l{l}.per_layer_input_gate"));
            let wproj = self.model.linear(&format!("l{l}.per_layer_projection"));
            let hcopy = self.s.h[..m * hs].to_vec();
            let mut p = vec![0.0f32; m * pd];
            self.matmul(&wgate, &hcopy, m, hs, &mut p);
            k::gelu(&mut p[..m * pd]);
            for r in 0..m {
                let off = r * nl * pd + l * pd;
                let (pr, pl) = (&mut p[r * pd..(r + 1) * pd], &self.s.ple[off..off + pd]);
                k::mul(pr, pl);
            }
            let mut pout = vec![0.0f32; m * hs];
            self.matmul(&wproj, &p, m, pd, &mut pout);
            let ppn = self.model.f32s(&format!("l{l}.post_per_layer_input_norm")).to_vec();
            let lsc = self.model.f32s(&format!("l{l}.layer_scalar"))[0];
            for r in 0..m {
                let (a, b) = (r * hs, (r + 1) * hs);
                let src: Vec<f32> = pout[a..b].to_vec();
                k::rmsnorm(&mut pout[a..b], &src, &ppn, eps);
                let (hh, pp) = (&mut self.s.h[a..b], &pout[a..b]);
                k::add(hh, pp);
                k::scale(&mut self.s.h[a..b], lsc);
            }
            if std::env::var_os("FGM_DUMP").is_some() {
                self.dump.extend_from_slice(&self.s.h[..m * hs]);
            }
        }

        // final norm, captured for validation before the LM head
        if std::env::var_os("FGM_DUMP").is_some() {
            let nrm0 = self.model.f32s("norm").to_vec();
            let mut fin0 = vec![0.0f32; m * hs];
            for r in 0..m {
                let src: Vec<f32> = self.s.h[r * hs..(r + 1) * hs].to_vec();
                k::rmsnorm(&mut fin0[r * hs..(r + 1) * hs], &src, &nrm0, eps);
            }
            self.dump.extend_from_slice(&fin0);
        }
        if let Ok(path) = std::env::var("FGM_DUMP") {
            use std::io::Write;
            let mut f = std::fs::File::create(&path).expect("dump");
            f.write_all(&((nl + 1) as u32).to_le_bytes()).unwrap();
            f.write_all(&(m as u32).to_le_bytes()).unwrap();
            f.write_all(&(hs as u32).to_le_bytes()).unwrap();
            for v in &self.dump { f.write_all(&v.to_le_bytes()).unwrap(); }
            eprintln!("dumped {} layers x {m} x {hs} -> {path}", self.dump.len() / (m * hs));
        }

        kv.len = pos0 + m;

        // ---- final norm + LM head for the last token only
        let nrm = self.model.f32s("norm").to_vec();
        let last = (m - 1) * hs;
        let src: Vec<f32> = self.s.h[last..last + hs].to_vec();
        let mut fin = vec![0.0f32; hs];
        k::rmsnorm(&mut fin, &src, &nrm, eps);
        let wl = self.model.linear("lm_head");
        let mut logits = vec![0.0f32; cfg.vocab_size];
        self.matmul(&wl, &fin, 1, hs, &mut logits);
        if let Some(cap) = cfg.final_logit_softcapping {
            k::softcap(&mut logits, cap);
        }
        self.s.logits.copy_from_slice(&logits);
        &self.s.logits
    }
}
