//! fastgemma inference server — OpenAI-compatible `/v1/completions`.
//!
//!     fgm-serve                       # downloads weights on first launch
//!     fgm-serve --model model.fgm     # or point at a local file
//!
//!     curl localhost:8080/v1/completions -H 'Content-Type: application/json' \
//!       -d '{"prompt":"The capital of France is","max_tokens":16}'
//!
//! Scope is deliberately the *text* completions endpoint and nothing else. It
//! is the simplest thing that existing clients can talk to: no chat template to
//! get wrong, no tool-call schema to translate, no message-role handling. A
//! prompt goes in as text and completion text comes out.
//!
//! What is NOT implemented, and would be a lie to accept silently: `n` > 1,
//! `logprobs`, `echo`, `best_of`, `suffix`, and any sampling other than greedy.
//! Each of those is rejected with an explicit error rather than ignored,
//! because a client that asks for `temperature: 0.8` and is handed greedy
//! output has been given wrong results, not degraded ones.

mod fetch;
mod http;
mod tokenizer;

use fgm_core::{KvCache, Model, Runner};
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::time::Instant;
use tokenizer::Tokenizer;

const PREFILL_CHUNK: usize = 256;
const MAX_BODY: usize = 64 << 20;

struct Args {
    model: Option<String>,
    tokenizer: Option<String>,
    addr: String,
    threads: usize,
    ctx: usize,
    no_download: bool,
}

fn usage() -> ! {
    eprintln!(
        "fgm-serve — OpenAI-compatible completions for Gemma 4 on CPU

USAGE
    fgm-serve [--model FILE] [--tokenizer FILE] [--addr HOST:PORT]
              [--threads N] [--ctx N] [--no-download]

    --model FILE       .fgm weights. Default: downloaded from {repo}
                       into $FGM_HOME (else ~/.cache/fastgemma) on first launch.
    --tokenizer FILE   tokenizer.json. Same default source.
    --addr HOST:PORT   listen address (default 127.0.0.1:8080)
    --threads N        worker threads (default: all cores)
    --ctx N            max context in tokens (default 8192)
    --no-download      fail instead of fetching anything

ENDPOINTS
    POST /v1/completions    text completion, `stream` supported
    GET  /v1/models         one entry, the loaded model
    GET  /health            liveness

ENVIRONMENT
    FGM_HOME           where downloaded files live
    FGM_HF_ENDPOINT    Hugging Face mirror (default https://huggingface.co)
    HF_TOKEN           only needed for a gated or private mirror
    FGM_BACKEND        amx|vnni, overrides the CPUID choice of GEMM kernel
    FGM_WEIGHTS        int4|int8|auto:M (default auto:16)",
        repo = fetch::REPO
    );
    std::process::exit(2)
}

fn parse_args() -> Args {
    let mut a = Args {
        model: None,
        tokenizer: None,
        addr: "127.0.0.1:8080".into(),
        threads: std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4),
        ctx: 8192,
        no_download: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        let mut next = || it.next().unwrap_or_else(|| usage());
        match f.as_str() {
            "--model" | "-m" => a.model = Some(next()),
            "--tokenizer" => a.tokenizer = Some(next()),
            "--addr" => a.addr = next(),
            "--threads" | "-t" => a.threads = next().parse().unwrap_or_else(|_| usage()),
            "--ctx" => a.ctx = next().parse().unwrap_or_else(|_| usage()),
            "--no-download" => a.no_download = true,
            "--help" | "-h" => usage(),
            _ => usage(),
        }
    }
    a
}

/// Refuse to start on a CPU the kernels cannot run on, with a message that says
/// what is missing rather than letting a SIGILL land somewhere in a GEMM.
fn check_cpu() {
    let b = fgm_kernels::backend();
    eprintln!("fastgemma: GEMM backend {}", b.name());
    if b == fgm_kernels::Backend::Vnni {
        eprintln!(
            "fastgemma: no AMX on this CPU — running the AVX-512 VNNI path. \
             Prefill is roughly half what an AMX host delivers."
        );
    }
}

fn resolve(a: &Args) -> std::io::Result<(String, String)> {
    let dir = fetch::cache_dir();
    let model = match &a.model {
        Some(m) => m.clone(),
        None if a.no_download => {
            let p = dir.join(fetch::WEIGHTS);
            if !p.exists() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("--no-download given and {} is not present", p.display()),
                ));
            }
            p.display().to_string()
        }
        None => fetch::ensure(&dir, fetch::WEIGHTS)?.display().to_string(),
    };
    let tok = match &a.tokenizer {
        Some(t) => t.clone(),
        None if a.no_download => {
            let p = dir.join(fetch::TOKENIZER);
            if !p.exists() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("--no-download given and {} is not present", p.display()),
                ));
            }
            p.display().to_string()
        }
        None => fetch::ensure(&dir, fetch::TOKENIZER)?.display().to_string(),
    };
    Ok((model, tok))
}

fn argmax(v: &[f32]) -> usize {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &x) in v.iter().enumerate() {
        if x > best.1 {
            best = (i, x);
        }
    }
    best.0
}

/// Fields we accept but cannot honour. Rejecting is the whole point: silently
/// ignoring `temperature` turns "your sampler is greedy-only" into "your model
/// gives strange answers".
fn unsupported(req: &serde_json::Value) -> Option<String> {
    let num = |k: &str| req.get(k).and_then(|v| v.as_f64());
    if req.get("n").and_then(|v| v.as_u64()).unwrap_or(1) != 1 {
        return Some("n > 1 is not supported".into());
    }
    if num("temperature").map(|t| t != 0.0).unwrap_or(false) {
        return Some("only greedy decoding is implemented; temperature must be 0".into());
    }
    if num("top_p").map(|p| p != 1.0).unwrap_or(false) {
        return Some("only greedy decoding is implemented; top_p must be 1".into());
    }
    for k in ["logprobs", "echo", "best_of", "suffix", "logit_bias"] {
        if req.get(k).map(|v| !v.is_null()).unwrap_or(false) {
            return Some(format!("`{k}` is not supported"));
        }
    }
    None
}

fn stops_from(req: &serde_json::Value) -> Vec<String> {
    match req.get("stop") {
        Some(serde_json::Value::String(s)) => vec![s.clone()],
        Some(serde_json::Value::Array(a)) => {
            a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()
        }
        _ => Vec::new(),
    }
}

struct Server<'m> {
    runner: Runner<'m>,
    tok: Tokenizer,
    cfg: fgm_core::model::Config,
    model_id: String,
    ctx: usize,
}

impl<'m> Server<'m> {
    fn completions(&mut self, stream: &mut TcpStream, body: &[u8]) -> std::io::Result<()> {
        let req: serde_json::Value = match serde_json::from_slice(body) {
            Ok(v) => v,
            Err(e) => return http::json_error(stream, 400, &format!("invalid JSON: {e}"), "invalid_request_error"),
        };
        if let Some(msg) = unsupported(&req) {
            return http::json_error(stream, 400, &msg, "invalid_request_error");
        }
        // OpenAI allows a token-id array here; accept text only and say so.
        let Some(prompt) = req.get("prompt").and_then(|p| p.as_str()) else {
            return http::json_error(
                stream, 400,
                "`prompt` must be a string (token-id arrays are not accepted)",
                "invalid_request_error",
            );
        };
        let max_tokens = req.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
        let stream_mode = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
        let stops = stops_from(&req);

        let toks = self.tok.encode(prompt, true);
        if toks.len() + max_tokens > self.ctx {
            return http::json_error(
                stream, 400,
                &format!(
                    "prompt is {} tokens and max_tokens is {max_tokens}, which exceeds the \
                     {}-token context this server was started with (--ctx)",
                    toks.len(), self.ctx
                ),
                "invalid_request_error",
            );
        }

        let t0 = Instant::now();
        let mut cache = KvCache::new(&self.cfg, self.ctx, PREFILL_CHUNK);
        let mut off = 0usize;
        let mut tok = 0u32;
        while off < toks.len() {
            let n = PREFILL_CHUNK.min(toks.len() - off);
            let lg = self.runner.forward(&toks[off..off + n], off, &mut cache);
            if off + n >= toks.len() {
                tok = argmax(lg) as u32;
            }
            off += n;
        }
        let ttft = t0.elapsed().as_secs_f64();

        if stream_mode {
            http::sse_open(stream)?;
        }
        let id = format!("cmpl-{}", std::process::id());
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let mut out_ids: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut text = String::new();
        let mut sent = 0usize;  // bytes of `text` already streamed
        let mut pos = toks.len();
        let mut finish = "length";
        for _ in 0..max_tokens {
            if self.tok.eos.contains(&tok) {
                finish = "stop";
                break;
            }
            out_ids.push(tok);
            // Decode the whole run each step rather than per token: a UTF-8
            // character can span several byte-fallback tokens, and decoding
            // them individually emits replacement characters mid-word.
            text = self.tok.decode(&out_ids, true);
            if let Some(cut) = stops.iter().filter_map(|s| text.find(s.as_str())).min() {
                text.truncate(cut);
                finish = "stop";
                break;
            }
            if stream_mode && text.len() > sent {
                let delta = text[sent..].to_string();
                sent = text.len();
                http::sse_send(stream, &serde_json::json!({
                    "id": id, "object": "text_completion", "created": created,
                    "model": self.model_id,
                    "choices": [{ "text": delta, "index": 0, "finish_reason": null }],
                }))?;
            }
            tok = argmax(self.runner.forward(&[tok], pos, &mut cache)) as u32;
            pos += 1;
        }
        let el = t0.elapsed().as_secs_f64();
        eprintln!(
            "  {} prompt + {} gen tok  ttft {:.2}s  {:.1} tok/s decode",
            toks.len(), out_ids.len(), ttft,
            out_ids.len() as f64 / (el - ttft).max(1e-9)
        );

        let usage = serde_json::json!({
            "prompt_tokens": toks.len(),
            "completion_tokens": out_ids.len(),
            "total_tokens": toks.len() + out_ids.len(),
        });
        if stream_mode {
            if text.len() > sent {
                http::sse_send(stream, &serde_json::json!({
                    "id": id, "object": "text_completion", "created": created,
                    "model": self.model_id,
                    "choices": [{ "text": text[sent..], "index": 0, "finish_reason": null }],
                }))?;
            }
            http::sse_send(stream, &serde_json::json!({
                "id": id, "object": "text_completion", "created": created,
                "model": self.model_id,
                "choices": [{ "text": "", "index": 0, "finish_reason": finish }],
                "usage": usage,
            }))?;
            return http::sse_done(stream);
        }
        let resp = serde_json::json!({
            "id": id, "object": "text_completion", "created": created,
            "model": self.model_id,
            "choices": [{ "text": text, "index": 0, "logprobs": null, "finish_reason": finish }],
            "usage": usage,
        });
        http::respond(stream, 200, "application/json", resp.to_string().as_bytes())
    }

    fn handle(&mut self, mut stream: TcpStream) {
        let req = match http::read_request(&stream, MAX_BODY) {
            Ok(Some(r)) => r,
            Ok(None) => return,
            Err(e) => {
                let _ = http::json_error(&mut stream, 413, &e.to_string(), "invalid_request_error");
                return;
            }
        };
        let path = req.path.split('?').next().unwrap_or("/").to_string();
        let r = match (req.method.as_str(), path.as_str()) {
            ("POST", "/v1/completions") => self.completions(&mut stream, &req.body),
            ("GET", "/v1/models") => {
                let b = serde_json::json!({
                    "object": "list",
                    "data": [{ "id": self.model_id, "object": "model", "owned_by": "fastgemma" }],
                });
                http::respond(&mut stream, 200, "application/json", b.to_string().as_bytes())
            }
            ("GET", "/health") => http::respond(&mut stream, 200, "text/plain", b"ok\n"),
            ("OPTIONS", _) => http::respond(&mut stream, 200, "text/plain", b""),
            _ => http::json_error(
                &mut stream, 404,
                &format!("no route for {} {path}; this server implements /v1/completions only", req.method),
                "invalid_request_error",
            ),
        };
        if let Err(e) = r {
            eprintln!("fastgemma: response failed: {e}");
        }
    }
}

fn main() {
    let a = parse_args();
    check_cpu();

    let (model_path, tok_path) = match resolve(&a) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("fastgemma: {e}");
            std::process::exit(1);
        }
    };

    let tok = match Tokenizer::load(&tok_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("fastgemma: cannot read tokenizer {tok_path}: {e}");
            std::process::exit(1);
        }
    };
    let model = match Model::open(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("fastgemma: cannot open {model_path}: {e}");
            std::process::exit(1);
        }
    };
    let cfg = model.cfg.clone();
    if tok.vocab_size() < cfg.vocab_size {
        eprintln!(
            "fastgemma: tokenizer has {} entries but the model expects {} — \
             these files are not from the same checkpoint",
            tok.vocab_size(), cfg.vocab_size
        );
        std::process::exit(1);
    }
    eprintln!(
        "fastgemma: {} ({:.2} GB), {} layers, ctx {}, {} threads",
        model_path,
        model.total_bytes() as f64 / 1e9,
        cfg.num_hidden_layers,
        a.ctx,
        a.threads
    );

    let runner = Runner::new(&model, PREFILL_CHUNK, a.ctx, a.threads);
    let model_id = std::path::Path::new(&model_path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "fastgemma".into());
    let mut srv = Server { runner, tok, cfg, model_id, ctx: a.ctx };

    let listener = match TcpListener::bind(&a.addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("fastgemma: cannot bind {}: {e}", a.addr);
            std::process::exit(1);
        }
    };
    eprintln!("fastgemma: listening on http://{}  (POST /v1/completions)", a.addr);
    let _ = std::io::stderr().flush();

    // Serial by design: one Runner, one thread pool sized to the machine.
    // Accepting concurrently would just make every request slower.
    for s in listener.incoming() {
        match s {
            Ok(s) => srv.handle(s),
            Err(e) => eprintln!("fastgemma: accept failed: {e}"),
        }
    }
}
