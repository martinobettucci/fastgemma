//! Persistent worker pool for the two parallel phases of the forward pass.
//!
//! GEMMs split along N (output columns) in multiples of 4 n-blocks, which is
//! what both kernel paths step by. Attention splits over flattened
//! (token, head) pairs, so it parallelises during decode (m=1, 8 heads) as well
//! as prefill (m=512, 8 heads) — splitting on tokens alone would leave three
//! threads idle every decode step.
//!
//! Threads are created once and parked on a barrier: a decode step issues ~280
//! GEMMs, so spawning per call (~30 us each) would cost more than the arithmetic.
//! AMX tile permission is per-thread, so every worker calls `amx_init` on entry.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread::JoinHandle;

use fgm_kernels as k;

#[derive(Clone, Copy)]
pub struct GemmJob {
    pub m: usize,
    pub n: usize,
    pub kdim: usize,
    pub a: *const i8,
    pub a_scale: *const f32,
    pub bq: *const u8,
    pub b8: *const i8,
    pub s4: *const u16,
    pub s8: *const f32,
    pub bits: u32,
    pub group: usize,
    pub c: *mut f32,
    pub ldc: usize,
}

/// One row's view of its sequence: the KV cache it reads and where it sits.
/// Prefill of a single sequence points every row at the same cache with
/// increasing `pos`; batched decode points each row at a different one.
#[derive(Clone, Copy)]
pub struct RowRef {
    pub kc: *const i8,
    pub ks: *const f32,
    pub vc: *const i8,
    pub vs: *const f32,
    pub k_len: usize,
    pub pos: usize,
    /// ring-buffer capacity for sliding layers, 0 if the cache is full length
    pub ring: usize,
}

#[derive(Clone, Copy)]
pub struct AttnJob {
    pub out: *mut f32,
    pub q: *const f32,
    /// `m` entries, one per row of the batch
    pub rows: *const RowRef,
    pub nh: usize,
    pub kvh: usize,
    pub hd: usize,
    pub m: usize,
    /// sliding window, or 0 for full attention
    pub window: usize,
    /// use the head-batched path (one thread per row, all heads together)
    pub rowbatch: bool,
}

/// Rotate + quantise + tile-pack a GEMM activation, split by 16-row tile blocks.
#[derive(Clone, Copy)]
pub struct PrepJob {
    pub src: *const f32,
    pub lda: usize,
    pub rot: *mut f32,
    pub qa: *mut i8,
    pub qs: *mut f32,
    pub pa: *mut i8,
    pub m: usize,
    pub k: usize,
    pub hsz: usize,
}

#[derive(Clone, Copy)]
pub enum Job {
    Gemm(GemmJob),
    Attn(AttnJob),
    Prep(PrepJob),
}

// Each worker gets a disjoint slice of the output; every pointer either targets
// immutable mmap'd weights or per-call scratch the caller keeps alive across the
// barrier pair.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

struct Inner {
    job: std::cell::UnsafeCell<Option<Job>>,
    start: Barrier,
    done: Barrier,
    stop: AtomicBool,
}
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

pub struct Pool {
    nt: usize,
    inner: Arc<Inner>,
    handles: Vec<JoinHandle<()>>,
    scratch: std::cell::UnsafeCell<Vec<f32>>,
}

/// Even split of `total` items across `nt` workers.
#[inline]
fn split_n(total: usize, nt: usize, tid: usize) -> (usize, usize) {
    let per = total / nt;
    let extra = total % nt;
    let start = tid * per + tid.min(extra);
    (start, start + per + usize::from(tid < extra))
}

/// Column split for GEMM, rounded to whole groups of 4 n-blocks.
#[inline]
fn split_gemm(nblocks: usize, nt: usize, tid: usize) -> (usize, usize) {
    let (a, b) = split_n(nblocks / 4, nt, tid);
    (a * 4, b * 4)
}

fn run_gemm(j: &GemmJob, n0: usize, n1: usize) {
    if n0 >= n1 {
        return;
    }
    let (m, n, kd) = (j.m, j.n, j.kdim);
    unsafe {
        let a = std::slice::from_raw_parts(j.a, k::packed_a_len(m, kd));
        let asc = std::slice::from_raw_parts(j.a_scale, m.max(1));
        let c = std::slice::from_raw_parts_mut(j.c, m * j.ldc);
        if j.bits == 4 {
            let b = std::slice::from_raw_parts(j.bq, n * kd / 2);
            let s = std::slice::from_raw_parts(j.s4, (kd / j.group) * n);
            k::gemm_q4g(m, n, kd, a, asc, b, s, j.group, c, j.ldc, n0, n1);
        } else {
            let b = std::slice::from_raw_parts(j.b8, n * kd);
            let s = std::slice::from_raw_parts(j.s8, n);
            k::gemm_q8c(m, n, kd, a, asc, b, s, c, j.ldc, n0, n1);
        }
    }
}

/// Work items are (row, head) pairs flattened as `row * nh + head`. Splitting on
/// heads as well as rows matters: batched decode has m=8 rows but 8 heads, so
/// row-only splitting would leave threads idle on smaller batches.
fn run_attn(j: &AttnJob, i0: usize, i1: usize, scratch: &mut [f32]) {
    let (nh, hd) = (j.nh, j.hd);
    let kvd = j.kvh * hd;
    unsafe {
        let rows = std::slice::from_raw_parts(j.rows, j.m);
        for i in i0..i1 {
            let (r, h) = (i / nh, i % nh);
            let rr = &rows[r];
            let (st, en) = if j.window > 0 {
                (rr.pos.saturating_sub(j.window - 1), rr.pos + 1)
            } else {
                (0, rr.pos + 1)
            };
            let kc = std::slice::from_raw_parts(rr.kc, rr.k_len * kvd);
            let ks = std::slice::from_raw_parts(rr.ks, rr.k_len);
            // V is transposed with 4-way interleave, so its length is rounded
            // up to whole groups of four slots.
            let vc = std::slice::from_raw_parts(rr.vc, rr.k_len.div_ceil(4) * 4 * kvd);
            let vs = std::slice::from_raw_parts(rr.vs, rr.k_len);
            let out = std::slice::from_raw_parts_mut(j.out.add((r * nh + h) * hd), hd);
            let q = std::slice::from_raw_parts(j.q.add((r * nh + h) * hd), hd);
            k::attend_q8_heads(out, q, kc, ks, vc, vs, nh, j.kvh, hd, st, en,
                               scratch, 0, 1, rr.ring, rr.k_len);

        }
    }
}

/// Head-batched attention: one thread owns ALL heads of a row, which is what
/// lets a K line loaded once serve every head. Split is therefore by row, not
/// by (row, head) -- 256 units at a prefill chunk, which is ample, and 1 unit
/// at decode, which is why the shape rule keeps decode on the per-head path.
fn run_attn_rows(j: &AttnJob, r0: usize, r1: usize, scratch: &mut [f32]) {
    let (nh, hd) = (j.nh, j.hd);
    let kvd = j.kvh * hd;
    unsafe {
        let rows = std::slice::from_raw_parts(j.rows, j.m);
        for r in r0..r1 {
            let rr = &rows[r];
            let (st, en) = if j.window > 0 {
                (rr.pos.saturating_sub(j.window - 1), rr.pos + 1)
            } else {
                (0, rr.pos + 1)
            };
            let kc = std::slice::from_raw_parts(rr.kc, rr.k_len * kvd);
            let ks = std::slice::from_raw_parts(rr.ks, rr.k_len);
            let vc = std::slice::from_raw_parts(rr.vc, rr.k_len.div_ceil(4) * 4 * kvd);
            let vs = std::slice::from_raw_parts(rr.vs, rr.k_len);
            let out = std::slice::from_raw_parts_mut(j.out.add(r * nh * hd), nh * hd);
            let q = std::slice::from_raw_parts(j.q.add(r * nh * hd), nh * hd);
            k::attend_q8_row(out, q, kc, ks, vc, vs, nh, j.kvh, hd, st, en,
                             scratch, rr.ring, rr.k_len);
        }
    }
}

fn run_prep(j: &PrepJob, b0: usize, b1: usize) {
    if b0 >= b1 {
        return;
    }
    let (r0, r1) = (b0 * 16, (b1 * 16).min(j.m));
    if r0 >= r1 {
        return;
    }
    unsafe {
        // Exactly the span touched: rows are lda apart but only k wide, so
        // m*lda would over-claim past the end of the caller's buffer whenever
        // k < lda (o_proj and down_proj both hit that).
        let src = std::slice::from_raw_parts(j.src, (j.m - 1) * j.lda + j.k);
        let rot = std::slice::from_raw_parts_mut(j.rot, j.m * j.k);
        let qa = std::slice::from_raw_parts_mut(j.qa, j.m * j.k);
        let qs = std::slice::from_raw_parts_mut(j.qs, j.m);
        let pa = std::slice::from_raw_parts_mut(j.pa, k::packed_a_len(j.m, j.k));
        k::prep_rows(src, j.lda, rot, qa, qs, pa, j.m, j.k, j.hsz, r0, r1);
    }
}

impl Pool {
    pub fn new(nt: usize, scratch_len: usize) -> Self {
        assert!(nt >= 1);
        let inner = Arc::new(Inner {
            job: std::cell::UnsafeCell::new(None),
            start: Barrier::new(nt),
            done: Barrier::new(nt),
            stop: AtomicBool::new(false),
        });
        let mut handles = Vec::new();
        for tid in 1..nt {
            let inner = inner.clone();
            handles.push(std::thread::spawn(move || {
                k::amx_init();
                let mut scratch = vec![0.0f32; scratch_len];
                loop {
                    inner.start.wait();
                    if inner.stop.load(Ordering::Acquire) {
                        return;
                    }
                    match unsafe { (*inner.job.get()).unwrap() } {
                        Job::Gemm(g) => {
                            let (a, b) = split_gemm(g.n / 16, nt, tid);
                            run_gemm(&g, a, b);
                        }
                        Job::Attn(at) => {
                            if at.rowbatch {
                                let (a, b) = split_n(at.m, nt, tid);
                                run_attn_rows(&at, a, b, &mut scratch);
                            } else {
                                let (a, b) = split_n(at.m * at.nh, nt, tid);
                                run_attn(&at, a, b, &mut scratch);
                            }
                        }
                        Job::Prep(pj) => {
                            let (a, b) = split_n(pj.m.div_ceil(16), nt, tid);
                            run_prep(&pj, a, b);
                        }
                    }
                    inner.done.wait();
                }
            }));
        }
        k::amx_init();
        Pool {
            nt,
            inner,
            handles,
            scratch: std::cell::UnsafeCell::new(vec![0.0f32; scratch_len]),
        }
    }

    pub fn threads(&self) -> usize {
        self.nt
    }

    pub fn gemm(&self, job: GemmJob) {
        if self.nt == 1 {
            run_gemm(&job, 0, job.n / 16);
            return;
        }
        unsafe { *self.inner.job.get() = Some(Job::Gemm(job)) };
        self.inner.start.wait();
        let (a, b) = split_gemm(job.n / 16, self.nt, 0);
        run_gemm(&job, a, b);
        self.inner.done.wait();
    }

    /// Run the activation preamble (rotate + quantise + tile-pack) across the
    /// pool, split by 16-row tile blocks.
    pub fn prep(&self, job: PrepJob) {
        let blocks = job.m.div_ceil(16);
        // Below ~2 tile blocks per thread the barrier costs more than the work.
        if self.nt == 1 || blocks < self.nt * 2 {
            run_prep(&job, 0, blocks);
            return;
        }
        unsafe { *self.inner.job.get() = Some(Job::Prep(job)) };
        self.inner.start.wait();
        let (a, b) = split_n(blocks, self.nt, 0);
        run_prep(&job, a, b);
        self.inner.done.wait();
    }

    /// Attention, split across the pool by (row, head) pair. With one KV head
    /// every query head reads the same cache lines, so this split gets cache
    /// reuse between threads for free rather than partitioning against it.
    pub fn attn(&self, job: AttnJob) {
        let scratch = unsafe { &mut *self.scratch.get() };
        if self.nt == 1 {
            if job.rowbatch { run_attn_rows(&job, 0, job.m, scratch); }
            else { run_attn(&job, 0, job.m * job.nh, scratch); }
            return;
        }
        unsafe { *self.inner.job.get() = Some(Job::Attn(job)) };
        self.inner.start.wait();
        if job.rowbatch {
            let (a, b) = split_n(job.m, self.nt, 0);
            run_attn_rows(&job, a, b, scratch);
        } else {
            let (a, b) = split_n(job.m * job.nh, self.nt, 0);
            run_attn(&job, a, b, scratch);
        }
        self.inner.done.wait();
    }

}

impl Drop for Pool {
    fn drop(&mut self) {
        if self.nt > 1 {
            self.inner.stop.store(true, Ordering::Release);
            self.inner.start.wait();
            for h in self.handles.drain(..) {
                let _ = h.join();
            }
        }
    }
}
