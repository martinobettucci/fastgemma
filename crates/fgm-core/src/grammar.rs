//! Token-level constrained decoding for tool calls.
//!
//! The brief weights tool-call exactness equally with speed. Sampling freely and
//! hoping for valid JSON gets you a distribution over *nearly* valid JSON; the
//! only way to make structural validity a guarantee is to make invalid tokens
//! unrepresentable at every step.
//!
//! Approach: compile the tool schemas into a character-level DFA, then lift it
//! to the token level by walking every vocabulary token's bytes through the DFA
//! once per state. That yields, per state, a bitset of tokens that keep the
//! output on a path to a valid tool call. Masking is then one AND per step.
//!
//! Cost: |vocab| x |states| byte-walks at build time (~26 M for 262 k tokens and
//! 100 states, roughly a second) and |states| x |vocab| / 8 bytes of mask
//! (~3.3 MB), both one-time per tool set. Per-token cost at decode is a bitset
//! lookup, which is far cheaper than the LM head it gates.
//!
//! There is a second, larger win hiding here: when the mask admits only a few
//! tokens, the LM head only needs those rows. During the structural parts of a
//! tool call (`{"name": "`, `", "arguments": {`) the mask is often a single
//! token, so the 201 MB int4 LM-head read can be skipped entirely. See
//! `Constraint::forced`.

use std::collections::HashMap;

/// A tool parameter's value type, which decides what the DFA accepts.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamType {
    Str,
    Int,
    Num,
    Bool,
    /// closed set of allowed string values
    Enum(Vec<String>),
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub ty: ParamType,
    pub required: bool,
}

#[derive(Clone, Debug)]
pub struct Tool {
    pub name: String,
    pub params: Vec<Param>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ByteClass {
    /// JSON string body: anything except `"`, backslash and control bytes
    /// (no escapes, which keeps the DFA small and the model honest)
    StrBody,
    /// a digit, moving to the given state
    DigitTo(u32),
}

impl ByteClass {
    fn accepts(self, b: u8) -> bool {
        match self {
            ByteClass::StrBody => b >= 0x20 && b != b'"' && b != b'\\',
            ByteClass::DigitTo(_) => b.is_ascii_digit(),
        }
    }
    /// State to move to when the class accepts; None means "stay put".
    fn target(self) -> Option<usize> {
        match self {
            ByteClass::DigitTo(s) => Some(s as usize),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct State {
    /// literal continuations, keyed by first byte
    lits: Vec<(Vec<u8>, usize)>,
    /// optional character class loop: (class, exit-on-byte, exit state)
    class: Option<(ByteClass, u8, usize)>,
    accept: bool,
}

/// A compiled character DFA over the tool-call grammar.
pub struct Dfa {
    states: Vec<State>,
    start: usize,
}

impl Dfa {
    /// Step one byte. Returns the next state, or None if the byte is invalid.
    fn step(&self, s: usize, b: u8) -> Option<usize> {
        let st = &self.states[s];
        for (lit, next) in &st.lits {
            if lit[0] == b {
                return Some(if lit.len() == 1 { *next } else { s });
            }
        }
        if let Some((cls, exit_b, exit)) = st.class {
            if exit != usize::MAX && b == exit_b {
                return Some(exit);
            }
            if cls.accepts(b) {
                return Some(cls.target().unwrap_or(s));
            }
        }
        None
    }
}

/// Build the DFA for `{"name": "<tool>", "arguments": {...}}` over a tool set.
///
/// Kept deliberately small: one alternative per tool, then that tool's required
/// params in a fixed order. Fixing the order is a real constraint on the model,
/// but it removes the combinatorial blow-up of arbitrary key ordering and is
/// what makes the state count stay in the dozens rather than the thousands.
pub struct GrammarBuilder {
    states: Vec<State>,
}

impl GrammarBuilder {
    fn new() -> Self {
        GrammarBuilder { states: vec![State::default()] }
    }

    fn add(&mut self) -> usize {
        self.states.push(State::default());
        self.states.len() - 1
    }

    /// Chain a literal byte string from `from`, returning the end state.
    fn lit(&mut self, from: usize, s: &str) -> usize {
        let mut cur = from;
        for b in s.bytes() {
            let next = self.add();
            self.states[cur].lits.push((vec![b], next));
            cur = next;
        }
        cur
    }

    /// A JSON string body terminated by `"`.
    fn string_body(&mut self, from: usize) -> usize {
        let exit = self.add();
        self.states[from].class = Some((ByteClass::StrBody, b'"', exit));
        exit
    }

    /// JSON number: at least one digit, at most one '.', and at least one digit
    /// after the '.'. The naive "digits-and-dots" class accepted ".." and the
    /// empty string, which produced grammar-valid but JSON-invalid output.
    ///
    ///   from --digit--> intp --digit--> intp
    ///                    |  \--'.'--> frac0 --digit--> frac --digit--> frac
    ///                    \--exit-->                       \--exit-->
    fn number_body(&mut self, from: usize, exit_on: u8) -> usize {
        let intp = self.add();
        let frac0 = self.add();
        let frac = self.add();
        let exit = self.add();
        // `from` requires a digit to leave; no exit edge, so a number cannot be empty
        self.states[from].class = Some((ByteClass::DigitTo(intp as u32), 0, usize::MAX));
        self.states[intp].class = Some((ByteClass::DigitTo(intp as u32), exit_on, exit));
        self.states[intp].lits.push((vec![b'.'], frac0));
        self.states[frac0].class = Some((ByteClass::DigitTo(frac as u32), 0, usize::MAX));
        self.states[frac].class = Some((ByteClass::DigitTo(frac as u32), exit_on, exit));
        exit
    }

    fn value(&mut self, from: usize, ty: &ParamType, tail: &str) -> usize {
        match ty {
            ParamType::Str => {
                let open = self.lit(from, "\"");
                let end = self.string_body(open);
                self.lit(end, tail)
            }
            ParamType::Int | ParamType::Num => {
                // numbers end at the tail's first byte
                let end = self.number_body(from, tail.as_bytes()[0]);
                if tail.len() > 1 { self.lit(end, &tail[1..]) } else { end }
            }
            ParamType::Bool => {
                let t = self.lit(from, "true");
                let t = self.lit(t, tail);
                let f = self.lit(from, "false");
                let f = self.lit(f, tail);
                // merge: point both at a common accept-ish state
                self.states[f].lits.push((vec![0], t));
                t
            }
            ParamType::Enum(vals) => {
                let mut end = None;
                for v in vals {
                    let e = self.lit(from, &format!("\"{v}\""));
                    let e = self.lit(e, tail);
                    end = Some(match end {
                        None => e,
                        Some(prev) => {
                            self.states[e].lits.push((vec![0], prev));
                            prev
                        }
                    });
                }
                end.unwrap()
            }
        }
    }

    pub fn build(tools: &[Tool]) -> Dfa {
        let mut b = GrammarBuilder::new();
        let start = 0usize;
        let mut accepts = Vec::new();
        for t in tools {
            let s = b.lit(start, &format!("{{\"name\": \"{}\", \"arguments\": {{", t.name));
            let req: Vec<&Param> = t.params.iter().filter(|p| p.required).collect();
            let mut cur = s;
            for (i, p) in req.iter().enumerate() {
                cur = b.lit(cur, &format!("\"{}\": ", p.name));
                let tail = if i + 1 == req.len() { "}}" } else { ", " };
                cur = b.value(cur, &p.ty, tail);
            }
            if req.is_empty() {
                cur = b.lit(cur, "}}");
            }
            accepts.push(cur);
        }
        for a in accepts {
            b.states[a].accept = true;
        }
        Dfa { states: b.states, start }
    }
}

/// Token-level mask over a vocabulary, derived from the DFA.
pub struct Constraint {
    dfa: Dfa,
    /// `states x ceil(vocab/64)` bitset of allowed tokens
    mask: Vec<u64>,
    /// next state per (state, token), or usize::MAX if disallowed
    next: Vec<u32>,
    vocab: usize,
    words: usize,
    pub state: usize,
}

impl Constraint {
    /// `vocab_bytes[t]` is the UTF-8 byte string token `t` decodes to.
    pub fn compile(dfa: Dfa, vocab_bytes: &[Vec<u8>]) -> Self {
        let vocab = vocab_bytes.len();
        let words = vocab.div_ceil(64);
        let ns = dfa.states.len();
        let mut mask = vec![0u64; ns * words];
        let mut next = vec![u32::MAX; ns * vocab];
        for s in 0..ns {
            for (t, bytes) in vocab_bytes.iter().enumerate() {
                if bytes.is_empty() {
                    continue;
                }
                let mut cur = s;
                let mut ok = true;
                for &b in bytes {
                    match dfa.step(cur, b) {
                        Some(n) => cur = n,
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    mask[s * words + t / 64] |= 1u64 << (t % 64);
                    next[s * vocab + t] = cur as u32;
                }
            }
        }
        let start = dfa.start;
        Constraint { dfa, mask, next, vocab, words, state: start }
    }

    pub fn reset(&mut self) {
        self.state = self.dfa.start;
    }

    pub fn allowed(&self, t: usize) -> bool {
        self.mask[self.state * self.words + t / 64] & (1u64 << (t % 64)) != 0
    }

    /// Number of tokens the current state permits.
    pub fn count(&self) -> usize {
        let row = &self.mask[self.state * self.words..(self.state + 1) * self.words];
        row.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// If exactly one token is legal, return it — the LM head can be skipped
    /// entirely for this step, which is most of a tool call's structural bytes.
    pub fn forced(&self) -> Option<usize> {
        if self.count() != 1 {
            return None;
        }
        let row = &self.mask[self.state * self.words..(self.state + 1) * self.words];
        for (i, w) in row.iter().enumerate() {
            if *w != 0 {
                return Some(i * 64 + w.trailing_zeros() as usize);
            }
        }
        None
    }

    /// Set disallowed logits to -inf in place.
    pub fn apply(&self, logits: &mut [f32]) {
        debug_assert_eq!(logits.len(), self.vocab);
        let row = &self.mask[self.state * self.words..(self.state + 1) * self.words];
        for (w, chunk) in row.iter().zip(logits.chunks_mut(64)) {
            if *w == u64::MAX {
                continue;
            }
            for (i, v) in chunk.iter_mut().enumerate() {
                if *w & (1u64 << i) == 0 {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
    }

    pub fn advance(&mut self, t: usize) -> bool {
        let n = self.next[self.state * self.vocab + t];
        if n == u32::MAX {
            return false;
        }
        self.state = n as usize;
        true
    }

    /// Step a byte from an arbitrary state without mutating — lets callers
    /// look ahead to find a token that actually advances the grammar.
    pub fn peek(&self, state: usize, b: u8) -> Option<usize> {
        self.dfa.step(state, b)
    }

    pub fn done(&self) -> bool {
        self.dfa.states[self.state].accept
    }

    pub fn num_states(&self) -> usize {
        self.dfa.states.len()
    }
}

/// Load `tokenizer.json` and return each token id's decoded bytes.
///
/// Special/added tokens are returned as empty, which makes them permanently
/// illegal under any constraint. This matters: `<pad>` and friends have literal
/// byte forms (`"<pad>"`) that satisfy the JSON string-body class, so leaving
/// them in lets a constrained decode emit `{"p0": "<pad><pad>...` forever
/// without ever violating the grammar. Found exactly that way.
pub fn load_vocab_bytes(tokenizer_json: &str) -> Vec<Vec<u8>> {
    let v: serde_json::Value = serde_json::from_str(tokenizer_json).expect("tokenizer.json");
    let vocab = v["model"]["vocab"].as_object().expect("model.vocab");
    let mut max_id = 0usize;
    let mut pairs: Vec<(usize, &String)> = Vec::with_capacity(vocab.len());
    for (tok, id) in vocab {
        let id = id.as_u64().unwrap() as usize;
        max_id = max_id.max(id);
        pairs.push((id, tok));
    }
    let mut special = std::collections::HashSet::new();
    if let Some(added) = v.get("added_tokens").and_then(|a| a.as_array()) {
        for a in added {
            if let Some(id) = a.get("id").and_then(|i| i.as_u64()) {
                max_id = max_id.max(id as usize);
                special.insert(id as usize);
            }
        }
    }
    let mut out = vec![Vec::new(); max_id + 1];
    // SentencePiece-style: U+2581 marks a leading space.
    for (id, tok) in pairs {
        if special.contains(&id) {
            continue;
        }
        out[id] = tok.replace('\u{2581}', " ").into_bytes();
    }
    out
}

/// Convenience: a tool set matching the brief's shape (N tools x P params).
pub fn synthetic_tools(n: usize, params: usize) -> Vec<Tool> {
    let types = [ParamType::Str, ParamType::Int, ParamType::Bool, ParamType::Num];
    (0..n)
        .map(|i| Tool {
            name: format!("tool_{i}"),
            params: (0..params)
                .map(|j| Param {
                    name: format!("p{j}"),
                    ty: types[j % types.len()].clone(),
                    required: true,
                })
                .collect(),
        })
        .collect()
}

/// Map from tool name to its index, for scoring.
pub fn tool_index(tools: &[Tool]) -> HashMap<String, usize> {
    tools.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect()
}
