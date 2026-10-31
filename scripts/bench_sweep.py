"""Prompt-length sweep for a running Strata server, in the community-bench shape.

Fresh prompts (a unique nonce per request, so nothing is reused from the
prompt cache), 256 generated tokens each. Decode and prefill throughput are
read from the engine log, which is the engine's own measurement.
"""
import json
import random
import re
import string
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090"
OUT = sys.argv[2] if len(sys.argv) > 2 else "bench-sweep.json"
LOG = sys.argv[3] if len(sys.argv) > 3 else "strata-iq3_xxs-12gb.log"
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 1
SIZES = [int(x) for x in (sys.argv[5].split(",") if len(sys.argv) > 5 else ["1024", "4096", "32768", "131072"])]

PROMPT_RE = re.compile(
    r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
    r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)")
WARN_RE = re.compile(r"WARNING: only (\d+) MiB of VRAM free")

FILLER = ("The quick brown fox jumps over the lazy dog while the cat watches "
          "from the windowsill. ")


def engine_last():
    """The last completed prompt/generation line in the engine log."""
    try:
        with open(LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 400000))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    lines = [ln for ln in tail.splitlines() if "prompt " in ln and "generated in" in ln]
    for ln in reversed(lines):
        m = PROMPT_RE.search(ln)
        if m:
            return {
                "prompt_tokens": int(m.group(1)),
                "reused": int(m.group(2)),
                "read": int(m.group(3)),
                "prefill_ms": int(m.group(4)),
                "prefill_tok_s": float(m.group(5)),
                "generated": int(m.group(6)),
                "decode_ms": int(m.group(7)),
                "decode_tok_s": float(m.group(8)),
            }
    return None


def vram_warnings():
    """Free-VRAM warnings in the tail of the log (the driver-paging guard)."""
    try:
        with open(LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 400000))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    return [int(m.group(1)) for m in WARN_RE.finditer(tail)]


def nonce():
    return "".join(random.choice(string.ascii_lowercase) for _ in range(12))


def chat(prompt, max_tokens):
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
    usage = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
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
    return {"wall_s": round(time.perf_counter() - t0, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "engine": engine_last()}


def median(vals):
    vals = sorted(vals)
    n = len(vals)
    return vals[n // 2] if n % 2 else round((vals[n // 2 - 1] + vals[n // 2]) / 2, 1)


results = {}
print("warmup ...", flush=True)
chat("Say hello in one short sentence.", 32)

for size in SIZES:
    rows = []
    for rep in range(REPS):
        # A unique leading nonce per request defeats any prefix reuse.
        words = max(1, int(size / 17))
        prompt = (f"[nonce {nonce()}] Summarise the following text in two "
                  f"sentences.\n\n{FILLER * words}")
        r = chat(prompt, 256)
        e = r["engine"] or {}
        rows.append(r)
        print(f"{size} rep{rep}: tokens={r['prompt_tokens']} "
              f"prefill={e.get('prefill_tok_s')} decode={e.get('decode_tok_s')} "
              f"reused={e.get('reused')}", flush=True)
        time.sleep(2)
    ok = [r["engine"] for r in rows if r["engine"] and r["engine"]["reused"] == 0]
    results[size] = {
        "runs": rows,
        "n": len(ok),
        "prompt_tokens": [r["prompt_tokens"] for r in rows],
        "prefill_tok_s": median([o["prefill_tok_s"] for o in ok]) if ok else None,
        "decode_tok_s": median([o["decode_tok_s"] for o in ok]) if ok else None,
    }

results["vram_warnings_mib_free"] = sorted(set(vram_warnings()))
print(json.dumps({k: v for k, v in results.items()
                  if k in ("vram_warnings_mib_free",) or isinstance(v, dict) and "n" in v},
                 indent=2))
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
