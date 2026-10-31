"""Measure decode and prefill speed of a running Strata server (OpenAI-compatible).

Decode speed is taken from the engine log, not from the HTTP stream: the
server flushes content in waves, so "time to first content chunk" is not the
end of the prompt read and would inflate the result.
"""
import json
import re
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090"
OUT = sys.argv[2] if len(sys.argv) > 2 else None
LOG = sys.argv[3] if len(sys.argv) > 3 else "strata-iq3_xxs-12gb.log"

# strata serve: prompt 7063 tokens = 0 reused + 7063 read in 11445 ms (617.1 tok/s), 64 generated in 1391 ms (46.0 tok/s), ...
PROMPT_RE = re.compile(
    r"prompt (\d+) tokens = .*?read in (\d+) ms \(([\d.]+) tok/s\), "
    r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)")


def engine_last():
    """The last completed prompt/generation line in the engine log."""
    try:
        with open(LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 262144))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    lines = [ln for ln in tail.splitlines() if "prompt " in ln and "generated in" in ln]
    if not lines:
        return None
    m = PROMPT_RE.search(lines[-1])
    if not m:
        return None
    return {
        "prompt_tokens": int(m.group(1)),
        "prefill_ms": int(m.group(2)),
        "prefill_tok_s": float(m.group(3)),
        "generated": int(m.group(4)),
        "decode_ms": int(m.group(5)),
        "decode_tok_s": float(m.group(6)),
    }


def stream_chat(prompt, max_tokens):
    body = json.dumps({
        "model": "bench",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    chunks = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                if (ch.get("delta") or {}).get("content"):
                    chunks += 1
    total = time.perf_counter() - t0
    return {
        "wall_s": round(total, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens") or chunks,
        "engine": engine_last(),
    }


long_text = "The quick brown fox jumps over the lazy dog. " * 700

results = {}
print("warmup ...", flush=True)
stream_chat("Say hello in one short sentence.", 32)
results["decode"] = stream_chat(
    "Explain in plain words why a GPU is faster than a CPU for matrix multiplication.", 256)
print("decode:", results["decode"], flush=True)
results["prefill"] = stream_chat(
    "Summarise the following text in two sentences.\n\n" + long_text, 64)
print("prefill:", results["prefill"], flush=True)
results["code"] = stream_chat(
    "Write a Python function that merges two sorted lists into one sorted list, "
    "then a second function that finds the longest common subsequence of two strings. "
    "Include docstrings and a few assertions.", 256)
print("code:", results["code"], flush=True)

print(json.dumps(results, indent=2))
if OUT:
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
