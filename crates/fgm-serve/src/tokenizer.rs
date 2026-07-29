//! Gemma 4's tokenizer, read from the checkpoint's `tokenizer.json`.
//!
//! An OpenAI-shaped completions endpoint takes text and returns text, so the
//! server needs a real encoder and decoder rather than the byte table the
//! grammar compiler already uses. This is the same SentencePiece-BPE the
//! `tokenizers` library runs, implemented against the fields the file actually
//! declares rather than against what Gemma tokenizers usually do:
//!
//!   normalizer      Replace " " -> "\u{2581}"
//!   pre_tokenizer   Split on " ", MergedWithPrevious
//!   model           BPE, byte_fallback = true, ignore_merges = true
//!   decoder         Replace "\u{2581}" -> " ", ByteFallback, Fuse
//!
//! Two of those matter and are easy to get wrong:
//!
//! * The pre-tokenizer splits on a literal space, but the normalizer has
//!   already replaced every space with U+2581, so **nothing is ever split**.
//!   The whole prompt is one BPE unit. Implementing the split first, as the
//!   config reads top to bottom, gives different token boundaries.
//! * `ignore_merges` means a piece present in the vocab verbatim is emitted as
//!   that id without running merges at all. It is checked per pre-token, which
//!   here is the whole input, so it fires only for short prompts -- but when it
//!   fires it is the difference between one token and several.
//!
//! Added tokens (the 24 control tokens, including the tool-call delimiters) are
//! matched as literal strings before anything else, because they are not
//! reachable through merges.

use std::collections::HashMap;

pub struct Tokenizer {
    /// token string -> id, for the whole vocabulary
    vocab: HashMap<String, u32>,
    /// (left, right) -> merge rank; lower merges first
    ranks: HashMap<(String, String), u32>,
    /// id -> decoded bytes; `None` for a byte-fallback token, whose byte value
    /// is in `byte_val`
    pieces: Vec<Option<String>>,
    /// id -> raw byte for `<0xNN>` tokens
    byte_val: Vec<Option<u8>>,
    /// literal control strings matched before BPE, longest first
    added: Vec<(String, u32)>,
    pub bos: u32,
    pub eos: Vec<u32>,
}

impl Tokenizer {
    pub fn load(path: &str) -> std::io::Result<Tokenizer> {
        let raw = std::fs::read_to_string(path)?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let vocab_obj = v["model"]["vocab"]
            .as_object()
            .ok_or_else(|| bad("tokenizer.json has no model.vocab"))?;
        let mut vocab = HashMap::with_capacity(vocab_obj.len());
        let mut max_id = 0u32;
        for (tok, id) in vocab_obj {
            let id = id.as_u64().unwrap_or(0) as u32;
            max_id = max_id.max(id);
            vocab.insert(tok.clone(), id);
        }

        // merges are ["a", "b"] pairs in the modern format and "a b" strings in
        // the old one; accept both rather than assume.
        let mut ranks = HashMap::new();
        if let Some(ms) = v["model"]["merges"].as_array() {
            for (i, m) in ms.iter().enumerate() {
                let pair = match m {
                    serde_json::Value::Array(a) if a.len() == 2 => {
                        match (a[0].as_str(), a[1].as_str()) {
                            (Some(x), Some(y)) => Some((x.to_string(), y.to_string())),
                            _ => None,
                        }
                    }
                    serde_json::Value::String(s) => {
                        let mut it = s.splitn(2, ' ');
                        match (it.next(), it.next()) {
                            (Some(a), Some(b)) => Some((a.to_string(), b.to_string())),
                            _ => None,
                        }
                    }
                    _ => None,
                };
                if let Some(p) = pair {
                    ranks.entry(p).or_insert(i as u32);
                }
            }
        }

        let mut added: Vec<(String, u32)> = Vec::new();
        let mut special_ids = std::collections::HashSet::new();
        if let Some(list) = v.get("added_tokens").and_then(|a| a.as_array()) {
            for a in list {
                let (Some(id), Some(c)) = (
                    a.get("id").and_then(|i| i.as_u64()),
                    a.get("content").and_then(|c| c.as_str()),
                ) else {
                    continue;
                };
                max_id = max_id.max(id as u32);
                special_ids.insert(id as u32);
                added.push((c.to_string(), id as u32));
            }
        }
        // longest first, so <|tool_call> wins over any prefix of itself
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let n = max_id as usize + 1;
        let mut pieces = vec![None; n];
        let mut byte_val = vec![None; n];
        for (tok, &id) in &vocab {
            let i = id as usize;
            if let Some(b) = parse_byte_token(tok) {
                byte_val[i] = Some(b);
            } else {
                pieces[i] = Some(tok.replace('\u{2581}', " "));
            }
        }
        for (c, id) in &added {
            // control strings decode to themselves; a client that asks for them
            // back gets what it sent
            pieces[*id as usize] = Some(c.clone());
            byte_val[*id as usize] = None;
        }

        let bos = *vocab.get("<bos>").unwrap_or(&2);
        let mut eos = Vec::new();
        for t in ["<eos>", "<end_of_turn>"] {
            if let Some(&id) = vocab.get(t) {
                eos.push(id);
            }
        }
        if eos.is_empty() {
            eos.push(1);
        }

        Ok(Tokenizer { vocab, ranks, pieces, byte_val, added, bos, eos })
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// Encode text. `add_bos` prepends `<bos>`, which Gemma expects at the
    /// start of a sequence and which a raw completions prompt will not carry.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_bos {
            out.push(self.bos);
        }
        self.encode_into(text, &mut out);
        out
    }

    fn encode_into(&self, text: &str, out: &mut Vec<u32>) {
        // Control strings first: they cannot be produced by merging, so a
        // prompt that contains one has to be split around it.
        let mut i = 0usize;
        let bytes = text.as_bytes();
        let mut chunk_start = 0usize;
        'outer: while i < bytes.len() {
            for (s, id) in &self.added {
                if bytes[i..].starts_with(s.as_bytes()) {
                    if chunk_start < i {
                        self.bpe(&text[chunk_start..i], out);
                    }
                    out.push(*id);
                    i += s.len();
                    chunk_start = i;
                    continue 'outer;
                }
            }
            // advance one char, not one byte, so a multi-byte char is never cut
            i += utf8_len(bytes[i]);
        }
        if chunk_start < text.len() {
            self.bpe(&text[chunk_start..], out);
        }
    }

    fn bpe(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        // normalizer: space -> U+2581. The pre-tokenizer splits on a literal
        // space, of which there are now none, so this is one unit.
        let norm: String = text.replace(' ', "\u{2581}");

        // ignore_merges: a piece already in the vocab is emitted as-is
        if let Some(&id) = self.vocab.get(&norm) {
            out.push(id);
            return;
        }

        let mut parts: Vec<String> = norm.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..parts.len().saturating_sub(1) {
                let key = (parts[i].clone(), parts[i + 1].clone());
                if let Some(&r) = self.ranks.get(&key) {
                    if best.map(|(br, _)| r < br).unwrap_or(true) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", parts[i], parts[i + 1]);
            parts[i] = merged;
            parts.remove(i + 1);
        }

        for p in parts {
            match self.vocab.get(&p) {
                Some(&id) => out.push(id),
                // byte_fallback: anything unrepresentable goes out as <0xNN>
                None => {
                    for b in p.bytes() {
                        let key = format!("<0x{b:02X}>");
                        if let Some(&id) = self.vocab.get(&key) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    /// Decode ids to text. Byte-fallback tokens are accumulated and flushed as
    /// raw bytes, which is what the `ByteFallback` + `Fuse` decoder pair does:
    /// a single UTF-8 character split across several `<0xNN>` tokens only
    /// becomes valid text once all of them are in hand.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut out: Vec<u8> = Vec::new();
        for &id in ids {
            let i = id as usize;
            if i >= self.pieces.len() {
                continue;
            }
            if let Some(b) = self.byte_val[i] {
                out.push(b);
                continue;
            }
            if let Some(p) = &self.pieces[i] {
                if skip_special && self.is_control(id) {
                    continue;
                }
                out.extend_from_slice(p.as_bytes());
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    fn is_control(&self, id: u32) -> bool {
        self.added.iter().any(|(_, i)| *i == id)
    }
}

fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else if b >> 3 == 0b11110 {
        4
    } else {
        1
    }
}

/// `<0x41>` -> 0x41. Anything else is an ordinary piece.
fn parse_byte_token(t: &str) -> Option<u8> {
    let h = t.strip_prefix("<0x")?.strip_suffix('>')?;
    if h.len() != 2 {
        return None;
    }
    u8::from_str_radix(h, 16).ok()
}

fn bad(msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tk() -> Option<Tokenizer> {
        let p = std::env::var("FGM_TOKENIZER")
            .unwrap_or_else(|_| "/home/user/models/g4e2b/tokenizer.json".into());
        Tokenizer::load(&p).ok()
    }

    /// Round-tripping is necessary but weak: an encoder that emitted one byte
    /// token per byte would pass it. The id-level checks live in
    /// bench/eval/tokenizer_check.py, which compares against `tokenizers`.
    #[test]
    fn round_trips() {
        let Some(t) = tk() else { return };
        for s in [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "  leading and  doubled   spaces ",
            "unicode: éàü 日本語 🎉",
            "<|tool_call>call:get_weather{city:<|\"|>Tokyo<|\"|>}<tool_call|>",
        ] {
            let ids = t.encode(s, false);
            assert_eq!(t.decode(&ids, false), *s, "round trip failed for {s:?}");
        }
    }

    #[test]
    fn control_tokens_are_single_ids() {
        let Some(t) = tk() else { return };
        for (s, want) in [("<|tool_call>", 48u32), ("<tool_call|>", 49), ("<|\"|>", 52)] {
            let ids = t.encode(s, false);
            assert_eq!(ids, vec![want], "{s} did not encode to one id");
        }
    }
}
