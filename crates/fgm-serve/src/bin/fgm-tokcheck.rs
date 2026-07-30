//! Encode a JSON array of strings from stdin, print a JSON array of id arrays.
//!
//! Exists so bench/eval/tokenizer_check.py can diff our ids against HF
//! `tokenizers` without embedding a Python binding in the server.
// The decoder half is unused here -- this binary only encodes -- and including
// the module wholesale is still the right call: a second copy of the tokenizer
// that "only does encoding" is a second thing that can drift.
#[allow(dead_code)]
#[path = "../tokenizer.rs"]
mod tokenizer;

use std::io::Read;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: fgm-tokcheck <tokenizer.json>  (JSON array of strings on stdin)");
        std::process::exit(2)
    });
    let t = tokenizer::Tokenizer::load(&path).expect("load tokenizer");
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s).expect("read stdin");
    let cases: Vec<String> = serde_json::from_str(&s).expect("stdin is a JSON array of strings");
    let out: Vec<Vec<u32>> = cases.iter().map(|c| t.encode(c, false)).collect();
    println!("{}", serde_json::to_string(&out).unwrap());
}
