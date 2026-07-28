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
    /// `max_batch` is the largest number of rows a single forward will submit
    /// (the prefill chunk size). A ring layer must hold `sliding_window +
    /// max_batch` positions, not merely `sliding_window`: a batched forward
    /// writes all its rows before any attention runs, so with only `window`
    /// slots the later rows overwrite history the earlier rows still need. That
    /// produces silently wrong logits, not a crash.
    pub fn new(cfg: &Config, max_len: usize, max_batch: usize) -> Self {
        Self::build(cfg, max_len, max_batch, true)
    }

    /// Every KV layer full length, no ring buffers. Only for the regression
    /// test that ring-mapped reads match linear ones — it costs more memory.
    pub fn new_no_ring(cfg: &Config, max_len: usize) -> Self {
        Self::build(cfg, max_len, 0, false)
    }

    fn build(cfg: &Config, max_len: usize, max_batch: usize, allow_ring: bool) -> Self {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            if cfg.is_shared(l) {
                layers.push(None);
                continue;
            }
            let hd = cfg.head_dim_of(l);
            let kvh = cfg.num_key_value_heads;
            // A sliding layer needs full length only if a shared layer reads it.
            let needs_full =
                !allow_ring || !cfg.is_sliding(l) || cfg.store_full_length_kv.contains(&l);
            let cap = if needs_full {
                max_len
            } else {
                (cfg.sliding_window + max_batch).min(max_len)
            };
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

    /// Copy `src`'s populated state into `self`, so this sequence continues
    /// from a prefix that was prefilled once and shared by every request.
    ///
    /// This is the whole prefix-sharing mechanism. In the target profile the
    /// 12-tool system prompt is identical across the 8 concurrent requests, so
    /// re-prefilling it per sequence is pure waste: at a 2048-token shared
    /// prefix the 8 sequences drop from 65536 prefill tokens to 51200, 22%
    /// less. The fork itself is a memcpy of ~35 MB/seq, which measures in
    /// milliseconds against tens of seconds of avoided prefill.
    ///
    /// Both caches must come from the same `Config` and the same `max_batch`,
    /// so slot mapping (`pos % capacity`) is identical on both sides and the
    /// copy needs no remapping. A ring layer that has already wrapped holds
    /// live data at every slot, so it is copied whole; otherwise only the
    /// `len` slots actually written are copied.
    pub fn fork_from(&mut self, src: &KvCache) {
        assert_eq!(self.layers.len(), src.layers.len(), "fork across configs");
        assert!(src.len <= self.max_len, "prefix longer than this cache");
        for (dst, s) in self.layers.iter_mut().zip(src.layers.iter()) {
            let (dst, s) = match (dst, s) {
                (Some(d), Some(s)) => (d, s),
                (None, None) => continue,
                _ => panic!("fork across configs: layer storage differs"),
            };
            assert_eq!(dst.capacity, s.capacity, "fork across cache geometries");
            let slots = if s.ring && src.len >= s.capacity { s.capacity } else { src.len };
            let row = s.row_len();
            dst.k[..slots * row].copy_from_slice(&s.k[..slots * row]);
            dst.v[..slots * row].copy_from_slice(&s.v[..slots * row]);
            dst.ks[..slots].copy_from_slice(&s.ks[..slots]);
            dst.vs[..slots].copy_from_slice(&s.vs[..slots]);
        }
        self.len = src.len;
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
