//! First-launch download of the weights and tokenizer from Hugging Face.
//!
//! This shells out to `curl` (or `wget`) rather than linking an HTTP client.
//! The engine has no network dependency of its own and adding a TLS stack plus
//! a hub client to pull two files once, at first launch, would be the largest
//! dependency in the tree by a wide margin. `curl` is present on every host
//! that can plausibly run this, and its resume/redirect handling is better
//! than anything worth writing here.
//!
//! Downloads land on a `.part` file and are renamed only on success, so an
//! interrupted 4.3 GB transfer never leaves a truncated file that mmaps
//! cleanly and produces nonsense.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const REPO: &str = "P2Enjoy/fastgemma-gemma-4-E2B";
pub const WEIGHTS: &str = "g4e2b-dual.fgm";
pub const TOKENIZER: &str = "tokenizer.json";

/// Where downloaded files live: `$FGM_HOME`, else `$XDG_CACHE_HOME/fastgemma`,
/// else `~/.cache/fastgemma`.
pub fn cache_dir() -> PathBuf {
    if let Ok(d) = std::env::var("FGM_HOME") {
        return PathBuf::from(d);
    }
    let base = std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|_| std::env::var("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|_| PathBuf::from("/tmp"));
    base.join("fastgemma")
}

fn url(file: &str) -> String {
    let base = std::env::var("FGM_HF_ENDPOINT")
        .unwrap_or_else(|_| "https://huggingface.co".into());
    format!("{base}/{REPO}/resolve/main/{file}")
}

fn have(bin: &str) -> bool {
    Command::new(bin)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Download `file` into `dir` unless it is already there. Returns its path.
///
/// A partial file from an earlier run is resumed rather than restarted --
/// re-pulling 4.3 GB because a connection dropped at 90% is the kind of thing
/// that makes people give up on a first launch.
pub fn ensure(dir: &Path, file: &str) -> std::io::Result<PathBuf> {
    let dest = dir.join(file);
    if dest.exists() {
        return Ok(dest);
    }
    std::fs::create_dir_all(dir)?;
    let part = dir.join(format!("{file}.part"));
    let u = url(file);

    eprintln!("fastgemma: fetching {file} from {REPO}");
    eprintln!("           {u}");
    eprintln!("           -> {}", dest.display());

    // A progress bar redrawn into a log file is thousands of lines of hashes.
    // Show one only when stderr is a terminal; otherwise stay quiet but still
    // report errors, because a silent failed download is worse than noise.
    let tty = std::io::IsTerminal::is_terminal(&std::io::stderr());
    let mut cmd = if have("curl") {
        let mut c = Command::new("curl");
        c.args(["-L", "--fail", "--retry", "5", "--retry-delay", "2", "-C", "-"]);
        c.arg(if tty { "--progress-bar" } else { "-sS" });
        c.arg("-o").arg(&part).arg(&u);
        c
    } else if have("wget") {
        let mut c = Command::new("wget");
        c.args(["-c", "--tries=5"]);
        c.arg(if tty { "--progress=bar" } else { "-nv" });
        c.arg("-O").arg(&part).arg(&u);
        c
    } else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "neither curl nor wget is available. Download {u} to {} by hand, \
                 or pass --model <path>.",
                dest.display()
            ),
        ));
    };
    // HF_TOKEN is only needed for a gated or private mirror; sending it when
    // set costs nothing and saves a confusing 401 against one.
    if let Ok(tok) = std::env::var("HF_TOKEN") {
        if !tok.is_empty() {
            cmd.arg("-H").arg(format!("Authorization: Bearer {tok}"));
        }
    }

    let st = cmd.status()?;
    if !st.success() {
        let _ = std::io::stderr().flush();
        return Err(std::io::Error::other(format!(
            "download of {file} failed ({st}). The partial file is kept at {} \
             and will be resumed on the next run.",
            part.display()
        )));
    }
    std::fs::rename(&part, &dest)?;
    Ok(dest)
}
