"""Gemma 4 prompt construction, in the model's own format.

Sources: ai.google.dev/gemma/docs/core/prompt-formatting-gemma4 and
.../capabilities/text/function-calling-gemma4, cross-checked against the
special tokens declared in the shipped tokenizer_config.json.

    <bos><|turn>system
    {system}<|tool>declaration:NAME{...}<tool|><turn|>
    <|turn>user
    {message}<turn|>
    <|turn>model
    <|tool_call>call:NAME{key:<|"|>value<|"|>,n:3}<tool_call|>

Structured data uses bare keys and the <|"|> token as the string delimiter,
not JSON quoting -- notably a string may then contain " freely, which is why
the engine's DFA is built on this syntax rather than on JSON.
"""

from tokenizers import Tokenizer

# ids as declared in tokenizer.json's added_tokens
BOS, EOS = 2, 1
TURN_O, TURN_C = 105, 106
TOOL_O, TOOL_C = 46, 47
CALL_O, CALL_C = 48, 49
RESP_O, RESP_C = 50, 51
QUOTE = 52
CHAN_O, CHAN_C = 100, 101

SPECIAL = {
    "<bos>": BOS, "<eos>": EOS,
    "<|turn>": TURN_O, "<turn|>": TURN_C,
    "<|tool>": TOOL_O, "<tool|>": TOOL_C,
    "<|tool_call>": CALL_O, "<tool_call|>": CALL_C,
    "<|tool_response>": RESP_O, "<tool_response|>": RESP_C,
    '<|"|>': QUOTE,
    "<|channel>": CHAN_O, "<channel|>": CHAN_C,
}
INV = {v: k for k, v in SPECIAL.items()}


class Fmt:
    def __init__(self, tokenizer_json):
        self.tk = Tokenizer.from_file(tokenizer_json)

    def enc(self, text):
        """Encode plain text, with no special tokens of any kind."""
        return self.tk.encode(text, add_special_tokens=False).ids

    def dec(self, ids):
        """Decode, rendering special ids as their literal delimiter strings so
        a downstream parser can see the structure."""
        out, run = [], []
        for i in ids:
            if i in INV:
                if run:
                    out.append(self.tk.decode(run, skip_special_tokens=False))
                    run = []
                out.append(INV[i])
            else:
                run.append(i)
        if run:
            out.append(self.tk.decode(run, skip_special_tokens=False))
        return "".join(out)

    def declaration(self, tool):
        """<|tool>declaration:name{...}<tool|> for one tool."""
        props = []
        for p in tool["params"]:
            fields = [f'description:<|"|>{p["desc"]}<|"|>']
            if p.get("enum"):
                fields.append("enum:[" + ",".join(f'<|"|>{v}<|"|>' for v in p["enum"]) + "]")
            fields.append(f'type:<|"|>{p["type"]}<|"|>')
            props.append(f'{p["name"]}:{{' + ",".join(fields) + "}")
        req = ",".join(f'<|"|>{p["name"]}<|"|>' for p in tool["params"] if p.get("required", True))
        body = (f'description:<|"|>{tool["desc"]}<|"|>,'
                f'parameters:{{properties:{{' + ",".join(props) + "},"
                f'required:[{req}],type:<|"|>OBJECT<|"|>}}')
        return f'declaration:{tool["name"]}{{{body}}}'

    def prompt(self, system, tools, user):
        """Full prompt ids up to and including `<|turn>model\\n`."""
        ids = [BOS, TURN_O] + self.enc("system\n" + system)
        for t in tools:
            ids += [TOOL_O] + self._mixed(self.declaration(t)) + [TOOL_C]
        ids += [TURN_C, TURN_O] + self.enc("user\n" + user) + [TURN_C]
        ids += [TURN_O] + self.enc("model\n")
        return ids

    def _mixed(self, s):
        """Encode a string that contains literal <|"|> delimiters."""
        ids, buf = [], ""
        i = 0
        while i < len(s):
            if s.startswith('<|"|>', i):
                if buf:
                    ids += self.enc(buf)
                    buf = ""
                ids.append(QUOTE)
                i += 5
            else:
                buf += s[i]
                i += 1
        if buf:
            ids += self.enc(buf)
        return ids
