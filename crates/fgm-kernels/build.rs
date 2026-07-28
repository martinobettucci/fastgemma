fn main() {
    // -O2, deliberately: GCC -O3 sinks the AVX stores that fill the int4 unpack
    // buffer past the _tile_loadd that reads it. amx_gemm.c carries an asm
    // barrier for that, but -O2 is the belt to its braces.
    cc::Build::new()
        .file("csrc/amx_gemm.c")
        .file("csrc/ops.c")
        .flag("-O2")
        .flag("-march=sapphirerapids")
        .flag("-mamx-int8")
        .flag("-mamx-tile")
        .flag("-mamx-bf16")
        .flag("-mavx512fp16")
        .flag("-fno-strict-aliasing")
        .compile("fgmkernels");
    println!("cargo:rerun-if-changed=csrc/amx_gemm.c");
    println!("cargo:rerun-if-changed=csrc/ops.c");
}
