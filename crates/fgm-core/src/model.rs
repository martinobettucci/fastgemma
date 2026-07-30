//! `.fgm` loading: mmap the file, parse the header, hand out typed slices.
//!
//! Nothing is copied or repacked at load time — weights are already in AMX tile
//! order on disk, so a model "loads" in the time it takes to fault pages in.

use memmap2::Mmap;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

pub type F16 = u16;

#[derive(Deserialize, Debug, Clone)]
pub struct TensorEntry {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub nbytes: usize,
    #[serde(default)]
    pub meta: serde_json::Value,
}

#[derive(Deserialize, Debug, Clone)]
pub struct QuantBits {
    pub ffn: u32,
    pub attn: u32,
    pub ple: u32,
    pub emb: u32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub use_double_wide_mlp: bool,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub layer_types: Vec<String>,
    pub first_shared_layer: usize,
    pub store_full_length_kv: Vec<usize>,
    pub hidden_size_per_layer_input: usize,
    pub vocab_size: usize,
    pub vocab_size_per_layer_input: usize,
    pub rms_norm_eps: f32,
    pub final_logit_softcapping: Option<f32>,
    pub sliding_window: usize,
    pub embed_scale: f32,
    pub ple_embed_scale: f32,
    pub ple_model_projection_scale: f32,
    pub ple_input_scale: f32,
    pub hadamard: usize,
    pub quant: QuantBits,
}

impl Config {
    #[inline]
    pub fn is_shared(&self, l: usize) -> bool {
        l >= self.first_shared_layer
    }
    #[inline]
    pub fn is_sliding(&self, l: usize) -> bool {
        self.layer_types[l] == "sliding_attention"
    }
    #[inline]
    pub fn head_dim_of(&self, l: usize) -> usize {
        if self.is_sliding(l) { self.head_dim } else { self.global_head_dim }
    }
    #[inline]
    pub fn inter_of(&self, l: usize) -> usize {
        self.intermediate_size * if self.use_double_wide_mlp && self.is_shared(l) { 2 } else { 1 }
    }
    /// Layer whose KV this layer reads: itself if it owns KV, else the
    /// `store_full_length_kv` layer of the same type.
    pub fn kv_source(&self, l: usize) -> usize {
        if !self.is_shared(l) {
            return l;
        }
        *self
            .store_full_length_kv
            .iter()
            .find(|&&s| self.layer_types[s] == self.layer_types[l])
            .expect("no full-length KV source for layer type")
    }
}

#[derive(Deserialize, Debug)]
struct Header {
    config: Config,
    tensors: HashMap<String, TensorEntry>,
    #[serde(default)]
    meta: serde_json::Value,
}

pub struct Model {
    mmap: Mmap,
    pub cfg: Config,
    tensors: HashMap<String, TensorEntry>,
    pub meta: serde_json::Value,
}

/// A quantised linear weight: blob + scales + the geometry the kernel needs.
pub struct QLinear<'a> {
    pub bits: u32,
    pub n: usize,
    pub k: usize,
    pub q4: &'a [u8],
    pub q8: &'a [i8],
    pub s4: &'a [F16],
    pub s8: &'a [f32],
    pub group: usize,
    /// Hadamard block size folded into this weight, or 0.
    pub hadamard: usize,
}

impl Model {
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let f = File::open(path)?;
        let mmap = unsafe { Mmap::map(&f)? };
        assert_eq!(&mmap[0..8], b"FASTGEM1", "not a .fgm file");
        let hlen = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;
        let h: Header = serde_json::from_slice(&mmap[12..12 + hlen])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(Model { mmap, cfg: h.config, tensors: h.tensors, meta: h.meta })
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn entry(&self, name: &str) -> &TensorEntry {
        self.tensors.get(name).unwrap_or_else(|| panic!("missing tensor {name}"))
    }

    fn raw(&self, name: &str) -> &[u8] {
        let e = self.entry(name);
        &self.mmap[e.offset..e.offset + e.nbytes]
    }

    pub fn f32s(&self, name: &str) -> &[f32] {
        let b = self.raw(name);
        debug_assert_eq!(b.as_ptr() as usize % 4, 0);
        unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / 4) }
    }

    pub fn f16s(&self, name: &str) -> &[F16] {
        let b = self.raw(name);
        unsafe { std::slice::from_raw_parts(b.as_ptr() as *const F16, b.len() / 2) }
    }

    pub fn i8s(&self, name: &str) -> &[i8] {
        let b = self.raw(name);
        unsafe { std::slice::from_raw_parts(b.as_ptr() as *const i8, b.len()) }
    }

    pub fn u8s(&self, name: &str) -> &[u8] {
        self.raw(name)
    }

    pub fn shape(&self, name: &str) -> &[usize] {
        &self.entry(name).shape
    }

    /// Fetch a quantised linear by name, resolving dtype and scale layout.
    pub fn linear(&self, name: &str) -> QLinear<'_> {
        let e = self.entry(name);
        let (n, k) = (e.shape[0], e.shape[1]);
        let had = e.meta.get("hadamard").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        // int4 scale group along K; set by the converter's --group
        let grp = e.meta.get("group").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
        static EMPTY_U8: &[u8] = &[];
        static EMPTY_I8: &[i8] = &[];
        static EMPTY_F16: &[F16] = &[];
        static EMPTY_F32: &[f32] = &[];
        match e.dtype.as_str() {
            "q4g" => QLinear {
                bits: 4, n, k,
                q4: self.u8s(name), q8: EMPTY_I8,
                s4: self.f16s(&format!("{name}.scale")), s8: EMPTY_F32,
                group: grp, hadamard: had,
            },
            "q8c" => QLinear {
                bits: 8, n, k,
                q4: EMPTY_U8, q8: self.i8s(name),
                s4: EMPTY_F16, s8: self.f32s(&format!("{name}.scale")),
                group: 0, hadamard: had,
            },
            d => panic!("tensor {name} has non-linear dtype {d}"),
        }
    }

    /// A linear plus its int8 twin if `--weights=both` stored one.
    ///
    /// The twin lives at `<name>.i8`, so a file converted without it simply
    /// resolves to `None` and the runtime keeps using the primary -- old files
    /// stay loadable and nothing has to know which kind it opened.
    pub fn dual(&self, name: &str) -> crate::forward::DualW<'_> {
        let alt = format!("{name}.i8");
        crate::forward::DualW::new(
            self.linear(name),
            self.has(&alt).then(|| self.linear(&alt)),
        )
    }

    /// Gather one row of a row-major quantised table into `out`.
    pub fn gather(&self, name: &str, row: usize, out: &mut [f32]) {
        let e = self.entry(name);
        let k = e.shape[1];
        match e.dtype.as_str() {
            "q8r" => fgm_kernels::gather_q8r(out, self.i8s(name), self.f32s(&format!("{name}.scale")), row, k),
            "q4r" => fgm_kernels::gather_q4r(out, self.u8s(name), self.f16s(&format!("{name}.scale")), row, k),
            d => panic!("tensor {name} is not a gather table ({d})"),
        }
    }

    pub fn total_bytes(&self) -> usize {
        self.tensors.values().map(|t| t.nbytes).sum()
    }
}
