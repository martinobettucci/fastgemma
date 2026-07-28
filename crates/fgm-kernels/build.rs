// Two translation units, two ISA targets.
//
// amx_gemm.c is the only file that may contain tile instructions, so it is the
// only one compiled for Sapphire Rapids. Everything else is restricted to
// Cascade Lake -- AVX-512F/DQ/BW/VL/CD plus VNNI and F16C -- which is the
// floor a fallback host is allowed to have. Keeping `-mtune=sapphirerapids`
// means the AMX box still gets its scheduling; only the instruction *set* is
// narrowed, and ops.c uses nothing outside it.
//
// This split is load-bearing, not defensive. Compiled at -march=sapphirerapids
// the one `_Float16` scale load in fgm_gather_q4r lowered to `vcvtsh2ss`, and
// ops.c -- which contains no AMX at all -- SIGILL'd on a VNNI-only host.
fn main() {
    // -O2, deliberately: GCC -O3 sinks the AVX stores that fill the int4 unpack
    // buffer past the _tile_loadd that reads it. amx_gemm.c carries an asm
    // barrier for that, but -O2 is the belt to its braces.
    cc::Build::new()
        .file("csrc/amx_gemm.c")
        .flag("-O2")
        .flag("-march=sapphirerapids")
        .flag("-mamx-int8")
        .flag("-mamx-tile")
        .flag("-mamx-bf16")
        .flag("-mavx512fp16")
        .flag("-fno-strict-aliasing")
        .compile("fgmamx");

    cc::Build::new()
        .file("csrc/ops.c")
        .flag("-O2")
        .flag("-march=cascadelake")
        .flag("-mtune=sapphirerapids")
        .flag("-fno-strict-aliasing")
        .compile("fgmkernels");

    // -O3 here, and only here. The VNNI inner loop keeps sixteen accumulators
    // live across a 16-step unrolled body; at -O2 GCC declines the unroll and
    // spills them to the stack, which measured 63 G MAC/s against 143 for the
    // same source at -O3. There is no tile instruction in this file for -O3 to
    // reorder around, which is what -O2 was protecting elsewhere.
    cc::Build::new()
        .file("csrc/vnni_gemm.c")
        .flag("-O3")
        .flag("-march=cascadelake")
        .flag("-mtune=sapphirerapids")
        .flag("-fno-strict-aliasing")
        .compile("fgmvnni");

    println!("cargo:rerun-if-changed=csrc/amx_gemm.c");
    println!("cargo:rerun-if-changed=csrc/ops.c");
    println!("cargo:rerun-if-changed=csrc/vnni_gemm.c");
}
