//! FFI to the AMX / AVX-512 kernels in `csrc/`.
//!
//! Every function here is a thin `unsafe extern "C"` binding plus a safe
//! wrapper that asserts the shape invariants the C side assumes (K a multiple
//! of 64, N-block ranges a multiple of 4, A pre-packed into tiles).

use std::sync::Once;

#[allow(non_camel_case_types)]
pub type f16 = u16;

extern "C" {
    fn fgm_amx_init() -> i32;
    fn fgm_tile_probe(trials: i32, sleep_us: i64) -> i32;
    fn fgm_tile_guard_set(on: i32);
    fn fgm_tile_retry_count() -> i64;
    fn fgm_tile_retry_reset();
    fn fgm_pack_a(m: i32, k: i32, a: *const i8, ap: *mut i8);
    fn fgm_gemm_q4g(
        m: i32, n: i32, k: i32, a: *const i8, a_scale: *const f32,
        bq: *const u8, b_scale: *const f16, group: i32,
        c: *mut f32, ldc: i32, n0: i32, n1: i32,
    );
    fn fgm_gemm_q8c(
        m: i32, n: i32, k: i32, a: *const i8, a_scale: *const f32,
        b: *const i8, b_scale: *const f32,
        c: *mut f32, ldc: i32, n0: i32, n1: i32,
    );

    fn fgm_rmsnorm(out: *mut f32, x: *const f32, w: *const f32, n: i32, eps: f32);
    fn fgm_rmsnorm_noscale(out: *mut f32, x: *const f32, n: i32, eps: f32);
    fn fgm_gelu_mul(out: *mut f32, gate: *const f32, up: *const f32, n: i32);
    fn fgm_gelu(x: *mut f32, n: i32);
    fn fgm_rope(x: *mut f32, n_heads: i32, head_dim: i32, inv_freq: *const f32, pos: i32);
    fn fgm_fwht(x: *mut f32, n: i32, hsz: i32);
    fn fgm_quant_act(x: *const f32, rows: i32, k: i32, q: *mut i8, scale: *mut f32);
    fn fgm_gather_q8r(out: *mut f32, tbl: *const i8, sc: *const f32, row: i32, k: i32);
    fn fgm_gather_q4r(out: *mut f32, tbl: *const u8, sc: *const f16, row: i32, k: i32);
    fn fgm_add(a: *mut f32, b: *const f32, n: i32);
    fn fgm_mul(a: *mut f32, b: *const f32, n: i32);
    fn fgm_scale(a: *mut f32, s: f32, n: i32);
    fn fgm_softcap(x: *mut f32, n: i32, cap: f32);
    fn fgm_attend_q8(
        out: *mut f32, q: *const f32, kc: *const i8, ks: *const f32,
        vc: *const i8, vs: *const f32, n_heads: i32, kv_heads: i32,
        head_dim: i32, start: i32, end: i32, scratch: *mut f32,
    );
    fn fgm_attend_q8_heads(
        out: *mut f32, q: *const f32, kc: *const i8, ks: *const f32,
        vc: *const i8, vs: *const f32, n_heads: i32, kv_heads: i32,
        head_dim: i32, start: i32, end: i32, scratch: *mut f32,
        h0: i32, h1: i32, ring: i32,
    );
}

static AMX: Once = Once::new();

/// Request XTILEDATA from the kernel. Must be called on every thread that runs
/// AMX instructions, before the first one — the permission is per-thread.
pub fn amx_init() -> bool {
    let mut ok = true;
    AMX.call_once(|| {});
    unsafe {
        ok &= fgm_amx_init() != 0;
    }
    ok
}

/// Bytes needed for the tile-packed copy of an `m x k` activation matrix.
#[inline]
pub fn packed_a_len(m: usize, k: usize) -> usize {
    m.div_ceil(16) * 16 * k
}

#[inline]
pub fn pack_a(m: usize, k: usize, a: &[i8], ap: &mut [i8]) {
    debug_assert!(k % 64 == 0);
    debug_assert!(a.len() >= m * k);
    debug_assert!(ap.len() >= packed_a_len(m, k));
    unsafe { fgm_pack_a(m as i32, k as i32, a.as_ptr(), ap.as_mut_ptr()) }
}

/// `C[m, n] = (A int8 * a_scale) . (int4-grouped B * b_scale)`, columns
/// `[n0*16, n1*16)`. `a` must already be tile-packed via [`pack_a`].
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn gemm_q4g(
    m: usize, n: usize, k: usize, a: &[i8], a_scale: &[f32],
    bq: &[u8], b_scale: &[f16], group: usize, c: &mut [f32], ldc: usize,
    n0: usize, n1: usize,
) {
    debug_assert!(k % 64 == 0 && n % 16 == 0);
    debug_assert!((n1 - n0) % 4 == 0, "n-block range must be a multiple of 4");
    unsafe {
        fgm_gemm_q4g(
            m as i32, n as i32, k as i32, a.as_ptr(), a_scale.as_ptr(),
            bq.as_ptr(), b_scale.as_ptr(), group as i32,
            c.as_mut_ptr(), ldc as i32, n0 as i32, n1 as i32,
        )
    }
}

/// `C[m, n] = (A int8 * a_scale) . (int8 per-channel B * b_scale)`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn gemm_q8c(
    m: usize, n: usize, k: usize, a: &[i8], a_scale: &[f32],
    b: &[i8], b_scale: &[f32], c: &mut [f32], ldc: usize, n0: usize, n1: usize,
) {
    debug_assert!(k % 64 == 0 && n % 16 == 0);
    debug_assert!((n1 - n0) % 4 == 0, "n-block range must be a multiple of 4");
    unsafe {
        fgm_gemm_q8c(
            m as i32, n as i32, k as i32, a.as_ptr(), a_scale.as_ptr(),
            b.as_ptr(), b_scale.as_ptr(), c.as_mut_ptr(), ldc as i32,
            n0 as i32, n1 as i32,
        )
    }
}

#[inline]
pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let n = w.len();
    unsafe { fgm_rmsnorm(out.as_mut_ptr(), x.as_ptr(), w.as_ptr(), n as i32, eps) }
}

#[inline]
pub fn rmsnorm_noscale(out: &mut [f32], x: &[f32], n: usize, eps: f32) {
    unsafe { fgm_rmsnorm_noscale(out.as_mut_ptr(), x.as_ptr(), n as i32, eps) }
}

#[inline]
pub fn gelu_mul(out: &mut [f32], gate: &[f32], up: &[f32], n: usize) {
    unsafe { fgm_gelu_mul(out.as_mut_ptr(), gate.as_ptr(), up.as_ptr(), n as i32) }
}

#[inline]
pub fn gelu(x: &mut [f32]) {
    let n = x.len();
    unsafe { fgm_gelu(x.as_mut_ptr(), n as i32) }
}

#[inline]
pub fn rope(x: &mut [f32], n_heads: usize, head_dim: usize, inv_freq: &[f32], pos: usize) {
    unsafe { fgm_rope(x.as_mut_ptr(), n_heads as i32, head_dim as i32, inv_freq.as_ptr(), pos as i32) }
}

/// In-place normalised fast Walsh-Hadamard transform on blocks of `hsz`.
#[inline]
pub fn fwht(x: &mut [f32], hsz: usize) {
    let n = x.len();
    debug_assert!(n % hsz == 0 && hsz.is_power_of_two());
    unsafe { fgm_fwht(x.as_mut_ptr(), n as i32, hsz as i32) }
}

#[inline]
pub fn quant_act(x: &[f32], rows: usize, k: usize, q: &mut [i8], scale: &mut [f32]) {
    unsafe { fgm_quant_act(x.as_ptr(), rows as i32, k as i32, q.as_mut_ptr(), scale.as_mut_ptr()) }
}

#[inline]
pub fn gather_q8r(out: &mut [f32], tbl: &[i8], sc: &[f32], row: usize, k: usize) {
    unsafe { fgm_gather_q8r(out.as_mut_ptr(), tbl.as_ptr(), sc.as_ptr(), row as i32, k as i32) }
}

#[inline]
pub fn gather_q4r(out: &mut [f32], tbl: &[u8], sc: &[f16], row: usize, k: usize) {
    unsafe { fgm_gather_q4r(out.as_mut_ptr(), tbl.as_ptr(), sc.as_ptr(), row as i32, k as i32) }
}

#[inline]
pub fn add(a: &mut [f32], b: &[f32]) {
    let n = a.len().min(b.len());
    unsafe { fgm_add(a.as_mut_ptr(), b.as_ptr(), n as i32) }
}

#[inline]
pub fn mul(a: &mut [f32], b: &[f32]) {
    let n = a.len().min(b.len());
    unsafe { fgm_mul(a.as_mut_ptr(), b.as_ptr(), n as i32) }
}

#[inline]
pub fn scale(a: &mut [f32], s: f32) {
    let n = a.len();
    unsafe { fgm_scale(a.as_mut_ptr(), s, n as i32) }
}

#[inline]
pub fn softcap(x: &mut [f32], cap: f32) {
    let n = x.len();
    unsafe { fgm_softcap(x.as_mut_ptr(), n as i32, cap) }
}

/// Attention for one query row against an int8 KV cache, positions `[start, end)`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn attend_q8(
    out: &mut [f32], q: &[f32], kc: &[i8], ks: &[f32], vc: &[i8], vs: &[f32],
    n_heads: usize, kv_heads: usize, head_dim: usize, start: usize, end: usize,
    scratch: &mut [f32],
) {
    debug_assert!(scratch.len() >= end - start);
    unsafe {
        fgm_attend_q8(
            out.as_mut_ptr(), q.as_ptr(), kc.as_ptr(), ks.as_ptr(),
            vc.as_ptr(), vs.as_ptr(), n_heads as i32, kv_heads as i32,
            head_dim as i32, start as i32, end as i32, scratch.as_mut_ptr(),
        )
    }
}

/// Attention restricted to heads `[h0, h1)`, so the pool can split the work.
/// `q` and `out` point at the first head in the range, not at head 0.
///
/// `ring` is the KV cache capacity for sliding layers stored in a ring buffer
/// (0 = linear). It MUST match what the write path used, or reads run off the
/// end of the allocation once the context passes the window.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn attend_q8_heads(
    out: &mut [f32], q: &[f32], kc: &[i8], ks: &[f32], vc: &[i8], vs: &[f32],
    n_heads: usize, kv_heads: usize, head_dim: usize, start: usize, end: usize,
    scratch: &mut [f32], h0: usize, h1: usize, ring: usize,
) {
    debug_assert!(scratch.len() >= end - start);
    debug_assert!(ring == 0 || kc.len() >= ring * kv_heads * head_dim);
    unsafe {
        fgm_attend_q8_heads(
            out.as_mut_ptr(), q.as_ptr(), kc.as_ptr(), ks.as_ptr(),
            vc.as_ptr(), vs.as_ptr(), n_heads as i32, kv_heads as i32,
            head_dim as i32, start as i32, end as i32, scratch.as_mut_ptr(),
            h0 as i32, h1 as i32, ring as i32,
        )
    }
}

/// Does this platform preserve AMX tile state across a context switch?
///
/// Some hypervisors do not: tiles come back zeroed, so any accumulator held in
/// a tile across a k-loop silently loses everything summed before the switch.
/// The probe forces the condition with an explicit `nanosleep` rather than
/// waiting for ambient load, so the answer does not depend on how busy the box
/// happens to be during warm-up.
///
/// Returns the number of corrupted trials out of `trials`. `hold_us` is how
/// long tile state is held per trial; the probe oversubscribes the machine with
/// spinner threads for the duration, because a *voluntary* context switch
/// (nanosleep) does not reproduce the fault — only involuntary preemption does.
pub fn tile_probe(trials: usize, hold_us: i64) -> usize {
    unsafe { fgm_tile_probe(trials as i32, hold_us) as usize }
}

/// Force the in-GEMM corruption guard on or off. The guard costs 1-2%, so it
/// should only run where [`tile_probe`] found a defect.
pub fn tile_guard_set(on: bool) {
    unsafe { fgm_tile_guard_set(i32::from(on)) }
}

/// Corruptions detected and recovered since the last reset.
pub fn tile_retries() -> u64 {
    unsafe { fgm_tile_retry_count() as u64 }
}

pub fn tile_retries_reset() {
    unsafe { fgm_tile_retry_reset() }
}

/// Run the probe and configure the guard accordingly.
/// `FGM_TILE_GUARD=on|off` overrides the decision.
///
/// The stakes are asymmetric: a false positive costs 1-2% throughput, a false
/// negative means silently wrong answers under load. So detection escalates the
/// hold time until it either finds corruption or has looked hard enough that a
/// clean result is trustworthy. Corruption probability rises steeply with hold
/// time (measured: ~3% at 100 us, 16% at 1 ms, 89% at 10 ms), so the 16 ms stage
/// is effectively conclusive.
///
/// Returns (guard_enabled, corrupted_trials, hold_us_that_found_it).
pub fn tile_guard_autodetect(trials: usize) -> (bool, usize, i64) {
    match std::env::var("FGM_TILE_GUARD").as_deref() {
        Ok("on") => {
            tile_guard_set(true);
            return (true, 0, 0);
        }
        Ok("off") => {
            tile_guard_set(false);
            return (false, 0, 0);
        }
        _ => {}
    }
    for hold_us in [2_000i64, 8_000, 16_000] {
        let bad = tile_probe(trials, hold_us);
        if bad > 0 {
            tile_guard_set(true);
            return (true, bad, hold_us);
        }
    }
    tile_guard_set(false);
    (false, 0, 0)
}
