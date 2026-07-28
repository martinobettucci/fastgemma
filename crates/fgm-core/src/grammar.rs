//! Token-level constrained decoding for tool calls.
//!
//! The brief weights tool-call exactness equally with speed. Sampling freely and
//! hoping for a well-formed call gets you a distribution over *nearly* well-formed
//! calls; the only way to make structural validity a guarantee is to make invalid
//! tokens unrepresentable at every step.
//!
//! Approach: compile the tool schemas into a character-level DFA, then lift it to
//! the token level by walking every vocabulary token's bytes through the DFA once
//! per state. That yields, per state, a bitset of tokens that keep the output on a
//! path to a valid call. Masking is then one AND per step.
//!
//! # The grammar is Gemma 4's native call syntax, not JSON
//!
//! Gemma 4 does not emit JSON tool calls. `tokenizer_config.json` defines the
//! format in `response_template.fields.tool_calls`:
//!
//! ```text
//!   open_pattern: <\|tool_call>call:(?P<name>\w+)
//!   close:        <tool_call|>
//!   content:      json, with unquoted_keys = true
//!                 and string_delims = [["<|\"|>", "<|\"|>"]]
//! ```
//!
//! so a call looks like
//!
//! ```text
//!   <|tool_call>call:get_weather{city: <|"|>Paris<|"|>, days: 3}<tool_call|>
//! ```
//!
//! Three of those delimiters are *single vocabulary tokens*, not text:
//! `<|tool_call>` = 48, `<tool_call|>` = 49, `<|"|>` = 52. Keys are bare. Strings
//! are delimited by a token, which means a string body may contain `"` freely and
//! needs no escape machinery at all.
//!
//! Constraining this model to JSON instead would fight its own distribution at
//! every structural byte — the mask would forbid the token it wants and force one
//! it has never emitted in that position, which is the worst thing a constraint
//! can do to accuracy. So the DFA speaks the model's own syntax.
//!
//! Special tokens decode to empty byte strings (see `load_vocab_bytes`), so they
//! can never be reached by the byte walk. They are reachable only through explicit
//! per-state token edges, which is what keeps `<pad>` and friends out while
//! letting the three delimiters in exactly where the grammar wants them.
//!
//! Cost: |vocab| x |states| byte-walks at build time and |states| x |vocab| / 8
//! bytes of mask, both one-time per tool set. Per-token cost at decode is a bitset
//! lookup, far cheaper than the LM head it gates.
//!
//! There is a second, larger win hiding here: when the mask admits only one token,
//! the 201 MB int4 LM-head read can be skipped entirely. See `Constraint::forced`.

use std::collections::HashMap;

/// Vocabulary ids of the delimiters the call syntax is built from.
#[derive(Clone, Copy, Debug)]
pub struct Delims {
    pub call_open: u32,
    pub call_close: u32,
    pub quote: u32,
}

impl Delims {
    /// Gemma 4: `<|tool_call>`, `<tool_call|>`, `<|"|>`.
    pub const GEMMA4: Delims = Delims { call_open: 48, call_close: 49, quote: 52 };
}

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
    /// String body. Delimited by a *token*, so the only bytes to exclude are
    /// control characters; quotes and backslashes are ordinary content here.
    StrBody,
    /// a digit, moving to the given state
    DigitTo(u32),
}

impl ByteClass {
    fn accepts(self, b: u8) -> bool {
        match self {
            ByteClass::StrBody => b >= 0x20 || b == b'\n' || b == b'\t',
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
    /// literal byte edges, at most one per byte value
    lits: Vec<(u8, usize)>,
    /// special-token edges, keyed by vocabulary id
    toks: Vec<(u32, usize)>,
    /// optional character class loop: (class, exit-on-byte, exit state)
    class: Option<(ByteClass, u8, usize)>,
    accept: bool,
}

impl State {
    fn edge(&self, b: u8) -> Option<usize> {
        self.lits.iter().find(|(l, _)| *l == b).map(|(_, n)| *n)
    }
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
        if let Some(n) = st.edge(b) {
            return Some(n);
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

    fn step_tok(&self, s: usize, id: u32) -> Option<usize> {
        self.states[s].toks.iter().find(|(t, _)| *t == id).map(|(_, n)| *n)
    }
}

/// Builds the DFA for a tool set.
///
/// Alternatives — one per tool, `true`/`false`, enum members — are built as a
/// *trie* that converges on a shared successor. Both halves of that matter, and
/// both were originally wrong:
///
///   * Without trie sharing, each tool got its own chain from the start state,
///     so the start state held twelve edges on the same first byte and stepping
///     always took the first. Eleven of twelve tools were unreachable: the
///     constraint could only ever emit `tool_0`. Measured, before the fix:
///     `reachable 1/12`.
///   * Without convergence, the `false` branch of a bool ended in a state that
///     nothing continued from, so choosing `false` walked into a dead end where
///     the mask was empty and every logit became -inf.
pub struct GrammarBuilder {
    states: Vec<State>,
    d: Delims,
}

impl GrammarBuilder {
    fn new(d: Delims) -> Self {
        GrammarBuilder { states: vec![State::default()], d }
    }

    fn add(&mut self) -> usize {
        self.states.push(State::default());
        self.states.len() - 1
    }

    /// Walk `s` from `from`, reusing existing byte edges and creating the rest.
    /// Returns the state after the last byte.
    fn lit(&mut self, from: usize, s: &str) -> usize {
        let mut cur = from;
        for b in s.bytes() {
            cur = match self.states[cur].edge(b) {
                Some(n) => n,
                None => {
                    let n = self.add();
                    self.states[cur].lits.push((b, n));
                    n
                }
            };
        }
        cur
    }

    /// As `lit`, but the final byte lands on the existing state `to`, so several
    /// alternatives can converge. Intermediate bytes still share where possible.
    fn lit_to(&mut self, from: usize, s: &str, to: usize) {
        let bytes = s.as_bytes();
        assert!(!bytes.is_empty());
        let mut cur = from;
        for &b in &bytes[..bytes.len() - 1] {
            cur = match self.states[cur].edge(b) {
                Some(n) => n,
                None => {
                    let n = self.add();
                    self.states[cur].lits.push((b, n));
                    n
                }
            };
        }
        let last = bytes[bytes.len() - 1];
        assert!(self.states[cur].edge(last).is_none(), "conflicting literal edge");
        self.states[cur].lits.push((last, to));
    }

    /// Token edge, reusing an existing edge for the same id.
    fn tok(&mut self, from: usize, id: u32) -> usize {
        if let Some(n) = self.states[from].toks.iter().find(|(t, _)| *t == id) {
            return n.1;
        }
        let n = self.add();
        self.states[from].toks.push((id, n));
        n
    }

    fn tok_to(&mut self, from: usize, id: u32, to: usize) {
        self.states[from].toks.push((id, to));
    }

    /// Allow one optional space before whatever `f` builds. The real separator
    /// convention after `:` and `,` is not pinned down by the tokenizer config,
    /// and forcing the wrong one would push probability mass onto a token the
    /// model does not want there — so accept both.
    fn maybe_space<F: FnOnce(&mut Self, usize)>(&mut self, from: usize, f: F) {
        let v = self.add();
        f(self, v);
        // The space edge is pushed first so it wins the lookup; the value
        // subgraph never begins with a space, so nothing is shadowed.
        self.states[from].lits.push((b' ', v));
        let cloned = self.states[v].clone();
        let st = &mut self.states[from];
        st.lits.extend(cloned.lits.iter().copied());
        st.toks.extend(cloned.toks.iter().copied());
        if st.class.is_none() {
            st.class = cloned.class;
        }
        st.accept |= cloned.accept;
    }

    /// A number: optional sign, at least one digit, and — when `frac` is allowed
    /// — at most one `.` with at least one digit after it. Lands on `to` after
    /// consuming `exit_on`.
    ///
    /// The naive "digits-and-dots" class accepted `..` and the empty string,
    /// which produced grammar-valid but unparseable output. An `Int` parameter
    /// also has to *reject* the fraction outright, or the constraint permits
    /// `count: 1.5` for a field the tool will refuse.
    fn number_to(&mut self, from: usize, exit_on: u8, to: usize, allow_frac: bool) {
        let intp = self.add();
        let digit = ByteClass::DigitTo(intp as u32);
        // no exit edge on `from`: a number cannot be empty
        self.states[from].class = Some((digit, 0, usize::MAX));
        let neg = self.add();
        self.states[from].lits.push((b'-', neg));
        self.states[neg].class = Some((digit, 0, usize::MAX));
        self.states[intp].class = Some((digit, exit_on, to));
        if allow_frac {
            let frac0 = self.add();
            let frac = self.add();
            self.states[intp].lits.push((b'.', frac0));
            self.states[frac0].class = Some((ByteClass::DigitTo(frac as u32), 0, usize::MAX));
            self.states[frac].class = Some((ByteClass::DigitTo(frac as u32), exit_on, to));
        }
    }

    /// One parameter value, ending on `exit_on` and landing on `to`.
    fn value_to(&mut self, from: usize, ty: &ParamType, exit_on: u8, to: usize) {
        let q = self.d.quote;
        match ty {
            ParamType::Str => {
                let body = self.tok(from, q);
                self.states[body].class = Some((ByteClass::StrBody, 0, usize::MAX));
                let closed = self.tok(body, q);
                self.lit_to(closed, &(exit_on as char).to_string(), to);
            }
            ParamType::Int => self.number_to(from, exit_on, to, false),
            ParamType::Num => self.number_to(from, exit_on, to, true),
            ParamType::Bool => {
                self.lit_to(from, &format!("true{}", exit_on as char), to);
                self.lit_to(from, &format!("false{}", exit_on as char), to);
            }
            ParamType::Enum(vals) => {
                let open = self.tok(from, q);
                let closed = self.add();
                let mut seen = std::collections::HashSet::new();
                for v in vals {
                    if !seen.insert(v.as_str()) {
                        continue;
                    }
                    let end = self.lit(open, v);
                    self.tok_to(end, q, closed);
                }
                self.lit_to(closed, &(exit_on as char).to_string(), to);
            }
        }
    }

    /// `<|tool_call>call:NAME{k: v, ...}<tool_call|>`, one alternative per tool.
    ///
    /// Keys are emitted in schema order. Fixing the order is a real constraint on
    /// the model, but it removes the combinatorial blow-up of arbitrary key
    /// ordering and keeps the state count linear in the schema size.
    pub fn build_with(tools: &[Tool], d: Delims) -> Dfa {
        let mut b = GrammarBuilder::new(d);
        let start = 0usize;
        let head = b.tok(start, d.call_open);
        let head = b.lit(head, "call:");
        // every tool converges here, so the closing token exists once
        let end = b.add();
        let accept = b.tok(end, d.call_close);
        b.states[accept].accept = true;

        for t in tools {
            let named = b.lit(head, &t.name);
            let mut cur = b.lit(named, "{");
            let req: Vec<&Param> = t.params.iter().filter(|p| p.required).collect();
            if req.is_empty() {
                b.lit_to(cur, "}", end);
                continue;
            }
            for (i, p) in req.iter().enumerate() {
                let last = i + 1 == req.len();
                // `cur` is where this key begins: after `{` for the first param,
                // after the previous param's `,` for the rest.
                let next = if last { end } else { b.add() };
                let key = p.name.clone();
                let ty = p.ty.clone();
                let exit = if last { b'}' } else { b',' };
                b.maybe_space(cur, move |bb, k| {
                    let after = bb.lit(k, &format!("{key}:"));
                    bb.maybe_space(after, move |b2, v| b2.value_to(v, &ty, exit, next));
                });
                cur = next;
            }
        }
        Dfa { states: b.states, start }
    }

    pub fn build(tools: &[Tool]) -> Dfa {
        Self::build_with(tools, Delims::GEMMA4)
    }
}

/// Token-level mask over a vocabulary, derived from the DFA.
pub struct Constraint {
    dfa: Dfa,
    /// `states x ceil(vocab/64)` bitset of allowed tokens
    mask: Vec<u64>,
    /// next state per (state, token), or u32::MAX if disallowed
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
            // Delimiter tokens have no byte form, so they are admitted only here,
            // and only in the states whose grammar calls for them.
            for &(id, tgt) in &dfa.states[s].toks {
                let id = id as usize;
                if id >= vocab {
                    continue;
                }
                mask[s * words + id / 64] |= 1u64 << (id % 64);
                next[s * vocab + id] = tgt as u32;
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
        self.count_at(self.state)
    }

    pub fn count_at(&self, s: usize) -> usize {
        self.mask[s * self.words..(s + 1) * self.words].iter().map(|w| w.count_ones() as usize).sum()
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

    /// Step a byte from an arbitrary state without mutating — lets callers look
    /// ahead to find a token that actually advances the grammar.
    pub fn peek(&self, state: usize, b: u8) -> Option<usize> {
        self.dfa.step(state, b)
    }

    /// Step a delimiter token from an arbitrary state without mutating.
    pub fn peek_tok(&self, state: usize, id: u32) -> Option<usize> {
        self.dfa.step_tok(state, id)
    }

    pub fn start(&self) -> usize {
        self.dfa.start
    }

    pub fn done(&self) -> bool {
        self.dfa.states[self.state].accept
    }

    pub fn accepting(&self, s: usize) -> bool {
        self.dfa.states[s].accept
    }

    pub fn num_states(&self) -> usize {
        self.dfa.states.len()
    }
}

/// Load `tokenizer.json` and return each token id's decoded bytes.
///
/// Special/added tokens are returned as empty, which makes them unreachable by
/// the byte walk. This matters: `<pad>` and friends have literal byte forms
/// (`"<pad>"`) that satisfy the string-body class, so leaving them in lets a
/// constrained decode emit `<pad><pad>...` forever without ever violating the
/// grammar. Found exactly that way. The three delimiters the call syntax needs
/// come back in through explicit token edges instead.
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

/// Render a decoded token stream back to text, restoring the delimiters that
/// `load_vocab_bytes` blanked out. Validation needs to see them.
pub fn render(ids: &[u32], vocab_bytes: &[Vec<u8>], d: Delims) -> String {
    let mut s = String::new();
    for &t in ids {
        if t == d.call_open {
            s.push_str("<|tool_call>");
        } else if t == d.call_close {
            s.push_str("<tool_call|>");
        } else if t == d.quote {
            s.push_str("<|\"|>");
        } else if let Some(b) = vocab_bytes.get(t as usize) {
            s.push_str(&String::from_utf8_lossy(b));
        }
    }
    s
}

/// A parsed tool call: name plus (key, value) pairs in emission order.
#[derive(Debug, PartialEq)]
pub struct Call {
    pub name: String,
    pub args: Vec<(String, String)>,
}

/// Parse Gemma 4's native call syntax. Returns None if the text is not a
/// well-formed call, which is what the exactness check asserts against.
pub fn parse_call(s: &str) -> Option<Call> {
    let s = s.strip_prefix("<|tool_call>")?;
    let s = s.strip_suffix("<tool_call|>")?;
    let s = s.strip_prefix("call:")?;
    let br = s.find('{')?;
    let name = s[..br].to_string();
    if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }
    let body = s[br + 1..].strip_suffix('}')?;
    let mut args = Vec::new();
    let mut rest = body.trim();
    while !rest.is_empty() {
        let colon = rest.find(':')?;
        let key = rest[..colon].trim().to_string();
        if key.is_empty() {
            return None;
        }
        rest = rest[colon + 1..].trim_start();
        let val;
        if let Some(after) = rest.strip_prefix("<|\"|>") {
            let end = after.find("<|\"|>")?;
            val = after[..end].to_string();
            rest = &after[end + 5..];
        } else {
            let end = rest.find(',').unwrap_or(rest.len());
            val = rest[..end].trim().to_string();
            if val.is_empty() || !valid_scalar(&val) {
                return None;
            }
            rest = &rest[end..];
        }
        args.push((key, val));
        rest = rest.trim_start();
        match rest.strip_prefix(',') {
            Some(r) => rest = r.trim_start(),
            None if rest.is_empty() => {}
            None => return None,
        }
    }
    Some(Call { name, args })
}

fn valid_scalar(v: &str) -> bool {
    if v == "true" || v == "false" {
        return true;
    }
    let v = v.strip_prefix('-').unwrap_or(v);
    let mut parts = v.split('.');
    let int = parts.next().unwrap_or("");
    if int.is_empty() || !int.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    match parts.next() {
        None => true,
        Some(f) => {
            !f.is_empty() && f.bytes().all(|b| b.is_ascii_digit()) && parts.next().is_none()
        }
    }
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
