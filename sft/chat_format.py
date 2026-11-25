"""
Chat format shared by SFT data preparation, SFT training and the AmirGPT server
(single source of truth, so training and inference can never disagree).

Karpathy, video [3:55:46]: "if you wanted to talk to the model we have to fine-tune
it into the chat format ... we're just swapping out a data set into a data set that
is a lot more conversational and there's a user assistant user assistant kind of
structure and we just fine-tune on it and then we basically fill in the user tokens
and we sample the assistant tokens".

Special tokens: the pretrained model's vocabulary is padded from 50,257 to 50,304
(video [2:07:26]) for speed, which leaves 47 never-used ids. We give four of them a
meaning; their embeddings are learned during SFT. No architecture change is needed.

  <|endoftext|> 50256  start of a conversation (the same document delimiter as pretraining)
  <|user|>      50257  start of a user message
  <|assistant|> 50258  start of an assistant message
  <|system|>    50259  start of a system message
  <|end|>       50260  end of any message (generation stops here)

A conversation is encoded as
  <|endoftext|> [<|system|> text <|end|>] <|user|> text <|end|> <|assistant|> text <|end|> ...
Message text is encoded with encode_ordinary, so a user who types "<|assistant|>"
gets ordinary text tokens and cannot forge a role token.

Loss mask: 1 for the tokens the model must learn to PRODUCE -- the assistant's text
and its closing <|end|> -- and 0 for everything else (system and user turns, role
tokens). Masks are aligned with the token they label; the trainer shifts them.
"""

import tiktoken

enc = tiktoken.get_encoding("gpt2")

EOT = 50256
USER = 50257
ASSISTANT = 50258
SYSTEM = 50259
END = 50260
ROLE_TOKENS = {"user": USER, "assistant": ASSISTANT, "system": SYSTEM}
SPECIAL_NAMES = {EOT: "<|endoftext|>", USER: "<|user|>", ASSISTANT: "<|assistant|>", SYSTEM: "<|system|>", END: "<|end|>"}
VOCAB_SIZE = 50304
# ids the assistant should never emit while answering (unused padding + other role markers)
BANNED_WHILE_ANSWERING = tuple([EOT, USER, ASSISTANT, SYSTEM] + list(range(END + 1, VOCAB_SIZE)))

DEFAULT_SYSTEM_PROMPT = ""


def encode_message(role, content):
    return [ROLE_TOKENS[role]] + enc.encode_ordinary(content) + [END]


def encode_conversation(messages, train_on=("assistant",)):
    """messages: list of {"role": "system"|"user"|"assistant", "content": str}.
    Returns (tokens, mask): mask[i] == 1 iff tokens[i] is something the model should learn to produce."""
    tokens, mask = [EOT], [0]
    for m in messages:
        role = m["role"]
        if role not in ROLE_TOKENS:
            raise ValueError(f"unknown role {role!r}")
        body = enc.encode_ordinary(m["content"])
        tokens += [ROLE_TOKENS[role]] + body + [END]
        learn = 1 if role in train_on else 0
        mask += [0] + [learn] * len(body) + [learn]
    return tokens, mask


def render_prompt(messages, system_prompt=None, max_tokens=1024, reserve=256):
    """Tokens for a generation prompt: the conversation so far + "<|assistant|>".
    If it does not fit in max_tokens - reserve, the OLDEST user/assistant turns are dropped
    (the system message is always kept). Returns the token list."""
    sys_msgs = [m for m in messages if m["role"] == "system"]
    if system_prompt:
        sys_msgs = [{"role": "system", "content": system_prompt}]
    turns = [m for m in messages if m["role"] != "system"]
    budget = max_tokens - reserve
    head = [EOT] + [t for m in sys_msgs for t in encode_message("system", m["content"])]
    encoded = [encode_message(m["role"], m["content"]) for m in turns]
    while True:
        body = [t for e in encoded for t in e]
        toks = head + body + [ASSISTANT]
        if len(toks) <= budget or len(encoded) <= 1:
            break
        encoded = encoded[1:]  # drop the oldest turn
    if len(toks) > budget:  # a single giant message: keep its tail
        toks = toks[:len(head)] + toks[-(budget - len(head)):]
    return toks


def decode(ids):
    """Decode ids to text, rendering our special tokens by name (for debugging)."""
    out, buf = [], []
    for i in ids:
        if i in SPECIAL_NAMES or i >= 50257:
            if buf:
                out.append(enc.decode(buf))
                buf = []
            out.append(SPECIAL_NAMES.get(i, f"<|unused_{i}|>"))
        else:
            buf.append(i)
    if buf:
        out.append(enc.decode(buf))
    return "".join(out)
