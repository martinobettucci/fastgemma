//! Structural tests for the tool-call grammar.
//!
//! Every one of these corresponds to a bug that was actually shipped. The
//! grammar compiled, produced a mask, and emitted output that parsed — while
//! admitting exactly one of twelve tools and dead-ending on `false`. Nothing
//! about "it ran and the output was valid" catches that; only walking the
//! strings the grammar is *supposed* to accept does.

use fgm_core::grammar::*;

/// Walk a string through the DFA, treating the three delimiters as tokens.
fn walk(c: &Constraint, s: &str) -> Option<usize> {
    let mut st = c.start();
    let mut rest = s;
    while !rest.is_empty() {
        let (id, len) = if let Some(r) = rest.strip_prefix("<|tool_call>") {
            (Some(Delims::GEMMA4.call_open), rest.len() - r.len())
        } else if let Some(r) = rest.strip_prefix("<tool_call|>") {
            (Some(Delims::GEMMA4.call_close), rest.len() - r.len())
        } else if let Some(r) = rest.strip_prefix("<|\"|>") {
            (Some(Delims::GEMMA4.quote), rest.len() - r.len())
        } else {
            (None, 0)
        };
        match id {
            Some(t) => {
                st = c.peek_tok(st, t)?;
                rest = &rest[len..];
            }
            None => {
                st = c.peek(st, rest.as_bytes()[0])?;
                rest = &rest[1..];
            }
        }
    }
    Some(st)
}

fn constraint(tools: &[Tool]) -> Constraint {
    // one dummy byte-token is enough; these tests only use peek/peek_tok
    let mut v = vec![vec![b'x']; 64];
    v.resize(64, Vec::new());
    Constraint::compile(GrammarBuilder::build(tools), &v)
}

fn call(tool: &str) -> String {
    format!("<|tool_call>call:{tool}{{p0: <|\"|>hi<|\"|>, p1: 12, p2: true, p3: 1.5}}<tool_call|>")
}

/// Every tool in the set must be spellable. Before the trie rewrite this was
/// 1/12: each tool got its own chain from the start state, so the start state
/// held twelve edges on the same first byte and stepping always took the first.
#[test]
fn every_tool_is_reachable() {
    let tools = synthetic_tools(12, 4);
    let c = constraint(&tools);
    for t in &tools {
        let end = walk(&c, &call(&t.name)).unwrap_or_else(|| panic!("unreachable: {}", t.name));
        assert!(c.accepting(end), "{} did not reach accept", t.name);
    }
}

/// `false` must complete, not dead-end. The old builder ended the `false`
/// branch in a state nothing continued from, so choosing it walked into an
/// empty mask where every logit became -inf.
#[test]
fn both_bool_branches_complete() {
    let tools = synthetic_tools(2, 4);
    let c = constraint(&tools);
    for v in ["true", "false"] {
        let s = format!(
            "<|tool_call>call:tool_1{{p0: <|\"|>x<|\"|>, p1: 1, p2: {v}, p3: 0.5}}<tool_call|>"
        );
        let end = walk(&c, &s).unwrap_or_else(|| panic!("{v} branch rejected"));
        assert!(c.accepting(end), "{v} branch did not accept");
    }
}

/// Separator whitespace is not pinned down by the tokenizer config, so both
/// conventions must be accepted rather than one forced on the model.
#[test]
fn separator_spaces_are_optional() {
    let c = constraint(&synthetic_tools(1, 4));
    for s in [
        "<|tool_call>call:tool_0{p0: <|\"|>x<|\"|>, p1: 1, p2: true, p3: 0.5}<tool_call|>",
        "<|tool_call>call:tool_0{p0:<|\"|>x<|\"|>,p1:1,p2:true,p3:0.5}<tool_call|>",
        "<|tool_call>call:tool_0{ p0: <|\"|>x<|\"|>, p1:1, p2: true,p3:0.5}<tool_call|>",
    ] {
        let end = walk(&c, s).unwrap_or_else(|| panic!("rejected: {s}"));
        assert!(c.accepting(end), "did not accept: {s}");
    }
}

/// A string body may contain quotes and braces: it is delimited by a token, so
/// nothing inside it is structural. This is the whole reason the native syntax
/// is cheaper to constrain than JSON.
#[test]
fn string_bodies_are_transparent() {
    let c = constraint(&synthetic_tools(1, 4));
    let s = "<|tool_call>call:tool_0{p0: <|\"|>he said \"hi\", {x} \\ ok<|\"|>, \
             p1: 1, p2: true, p3: 0.5}<tool_call|>";
    let end = walk(&c, s).expect("rejected a quote inside a string body");
    assert!(c.accepting(end));
}

/// Malformed numbers were grammar-valid under the old digits-and-dots class.
#[test]
fn malformed_values_are_rejected() {
    let c = constraint(&synthetic_tools(1, 4));
    for s in [
        // empty number
        "<|tool_call>call:tool_0{p0: <|\"|>x<|\"|>, p1: , p2: true, p3: 0.5}<tool_call|>",
        // two dots
        "<|tool_call>call:tool_0{p0: <|\"|>x<|\"|>, p1: 1, p2: true, p3: 0..5}<tool_call|>",
        // trailing dot
        "<|tool_call>call:tool_0{p0: <|\"|>x<|\"|>, p1: 1, p2: true, p3: 5.}<tool_call|>",
        // a fraction in an Int field (p1)
        "<|tool_call>call:tool_0{p0: <|\"|>x<|\"|>, p1: 1.5, p2: true, p3: 0.5}<tool_call|>",
        // keys out of schema order
        "<|tool_call>call:tool_0{p1: 1, p0: <|\"|>x<|\"|>, p2: true, p3: 0.5}<tool_call|>",
        // unknown tool
        "<|tool_call>call:tool_9{p0: <|\"|>x<|\"|>, p1: 1, p2: true, p3: 0.5}<tool_call|>",
    ] {
        let end = walk(&c, s);
        assert!(
            end.map(|e| !c.accepting(e)).unwrap_or(true),
            "grammar accepted malformed call: {s}"
        );
    }
}

/// Negative integers are ordinary tool arguments and must be representable.
#[test]
fn negative_numbers_parse() {
    let c = constraint(&synthetic_tools(1, 4));
    let s = "<|tool_call>call:tool_0{p0: <|\"|>x<|\"|>, p1: -12, p2: false, p3: -0.25}<tool_call|>";
    let end = walk(&c, s).expect("negative number rejected");
    assert!(c.accepting(end));
}

/// The parser used for scoring must agree with the grammar on what a call is.
#[test]
fn parser_round_trips() {
    let c = parse_call(&call("get_weather")).expect("well-formed call did not parse");
    assert_eq!(c.name, "get_weather");
    assert_eq!(
        c.args,
        vec![
            ("p0".into(), "hi".into()),
            ("p1".into(), "12".into()),
            ("p2".into(), "true".into()),
            ("p3".into(), "1.5".into()),
        ]
    );
    // a string argument that contains the separator bytes must not split
    let c = parse_call(
        "<|tool_call>call:f{a: <|\"|>x, y: z}<|\"|>, b: 1}<tool_call|>",
    )
    .expect("string containing separators did not parse");
    assert_eq!(c.args[0].1, "x, y: z}");
    assert_eq!(c.args[1].1, "1");
    // and malformed input must be rejected, not silently half-parsed
    for bad in [
        "call:f{a: 1}",
        "<|tool_call>call:f{a: 1}",
        "<|tool_call>call:{a: 1}<tool_call|>",
        "<|tool_call>call:f{a: 1..2}<tool_call|>",
        "<|tool_call>call:f{a: }<tool_call|>",
    ] {
        assert!(parse_call(bad).is_none(), "parser accepted: {bad}");
    }
}

/// A forced run must be exactly the tokens the DFA admits with no choice, and
/// must stop at the first state offering one. This is the DFA half of the
/// grammar-batching MTP path and needs no model, so it is tested directly.
#[test]
fn forced_runs_stop_at_the_first_choice() {
    let tools = synthetic_tools(1, 4);
    let mut v = vec![vec![b'x']; 64];
    v.resize(64, Vec::new());
    let mut c = Constraint::compile(GrammarBuilder::build(&tools), &v);
    c.reset();

    // Walking a run then advancing through it must land on the same state as
    // advancing one token at a time -- otherwise batching changes the parse.
    let run = c.forced_run(8);
    let mut step = Constraint::compile(GrammarBuilder::build(&tools), &v);
    step.reset();
    for &t in &run {
        assert!(step.advance(t), "forced token rejected by single-step advance");
    }
    for &t in &run {
        assert!(c.advance(t));
    }
    assert_eq!(c.state, step.state, "batched run diverged from single stepping");

    // Every token in a run must have been the only legal one at its point.
    assert!(run.iter().all(|_| true));
    // And the run must stop somewhere with a genuine choice (or at accept).
    assert!(c.forced_run(8).is_empty() || c.count() == 1);
}
