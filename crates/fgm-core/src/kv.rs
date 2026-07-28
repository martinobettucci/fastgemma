//! int8 KV cache.
//!
//! Only layers before `first_shared_layer` allocate storage at all — the shared
//! layers read the cache of their `store_full_length_kv` source. Sliding layers
//! that are *not* a shared-KV source only ever need `sliding_window` positions,
//! so they get a ring buffer; everything else is full length.
//!
//! For E2B at 8k context that is ~32 MB/seq instead of the ~1.8 GB a naive
//! all-layers-full-length cache would take for 8 concurrent sequences.

use crate::model::Config;

pub struct LayerKv {
    /// Positions this layer can hold. Full length, or `sliding_window`.
    pub capacity: usize,
    pub head_dim: usize,
    pub kv_heads: usize,
    /// `[capacity, kv_heads * head_dim]` int8
    pub k: Vec<i8>,
    pub v: Vec<i8>,
    /// per-position dequant scales
    pub ks: Vec<f32>,
    pub vs: Vec<f32>,
    pub ring: bool,
}

impl LayerKv {
    #[inline]
    pub fn slot(&self, pos: usize) -> usize {
        if self.ring { pos % self.capacity } else { pos }
    }
    #[inline]
    pub fn row_len(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

pub struct KvCache {
    pub layers: Vec<Option<LayerKv>>,
    pub len: usize,
    pub max_len: usize,
}

impl KvCache {
    pub fn new(cfg: &Config, max_len: usize) -> Self {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            if cfg.is_shared(l) {
                layers.push(None);
                continue;
            }
            let hd = cfg.head_dim_of(l);
            let kvh = cfg.num_key_value_heads;
            // A sliding layer needs full length only if a shared layer reads it.
            let needs_full = !cfg.is_sliding(l) || cfg.store_full_length_kv.contains(&l);
            let cap = if needs_full { max_len } else { cfg.sliding_window };
            layers.push(Some(LayerKv {
                capacity: cap,
                head_dim: hd,
                kv_heads: kvh,
                k: vec![0; cap * kvh * hd],
                v: vec![0; cap * kvh * hd],
                ks: vec![0.0; cap],
                vs: vec![0.0; cap],
                ring: !needs_full,
            }));
        }
        KvCache { layers, len: 0, max_len }
    }

    pub fn bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|l| l.k.len() + l.v.len() + 8 * l.ks.len())
            .sum()
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Window of positions layer `l` may attend to for a query at `pos`.
    pub fn window(&self, cfg: &Config, l: usize, pos: usize) -> (usize, usize) {
        if cfg.is_sliding(l) {
            (pos.saturating_sub(cfg.sliding_window - 1), pos + 1)
        } else {
            (0, pos + 1)
        }
    }
}
