//! A hand-rolled HTTP/1.1 server, because the alternative is worse.
//!
//! The endpoint is one POST with a JSON body and one response, optionally as
//! Server-Sent Events. Pulling in an async runtime and a web framework for
//! that would add more code to the dependency tree than the whole engine has,
//! and this file is small enough to read in one sitting.
//!
//! Deliberate limits, all of them documented rather than hidden:
//!
//! * HTTP/1.1, `Connection: close` on every response. No keep-alive, no
//!   pipelining, no chunked *request* bodies -- a `Content-Length` is required.
//! * One request at a time. The engine owns a thread pool sized to the machine
//!   and a single `Runner`; a second concurrent request would contend for the
//!   same cores and make both slower. Requests queue on the accept loop.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
}

/// Read one request. `None` means the peer closed or sent something malformed
/// enough that there is nothing to reply to.
pub fn read_request(stream: &TcpStream, max_body: usize) -> std::io::Result<Option<Request>> {
    let mut r = BufReader::new(stream);
    let mut line = String::new();
    if r.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut it = line.split_whitespace();
    let (Some(method), Some(path)) = (it.next(), it.next()) else {
        return Ok(None);
    };
    let (method, path) = (method.to_string(), path.to_string());

    let mut len = 0usize;
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 {
            break;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                len = v.trim().parse().unwrap_or(0);
            }
        }
    }
    if len > max_body {
        return Err(std::io::Error::other(format!(
            "request body {len} bytes exceeds the {max_body}-byte limit"
        )));
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body)?;
    Ok(Some(Request { method, path, body }))
}

pub fn respond(stream: &mut TcpStream, code: u16, ctype: &str, body: &[u8]) -> std::io::Result<()> {
    let reason = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        _ => "OK",
    };
    write!(
        stream,
        "HTTP/1.1 {code} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()
}

pub fn json_error(stream: &mut TcpStream, code: u16, msg: &str, kind: &str) -> std::io::Result<()> {
    let body = serde_json::json!({ "error": { "message": msg, "type": kind } });
    respond(stream, code, "application/json", body.to_string().as_bytes())
}

/// Open a Server-Sent Events response. The caller then writes `data:` frames
/// with [`sse_send`] and finishes with [`sse_done`].
pub fn sse_open(stream: &mut TcpStream) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\
         Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"
    )?;
    stream.flush()
}

pub fn sse_send(stream: &mut TcpStream, v: &serde_json::Value) -> std::io::Result<()> {
    write!(stream, "data: {v}\n\n")?;
    stream.flush()
}

pub fn sse_done(stream: &mut TcpStream) -> std::io::Result<()> {
    write!(stream, "data: [DONE]\n\n")?;
    stream.flush()
}
