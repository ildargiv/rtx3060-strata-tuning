"""Rebuild a run-by-run table from the engine log.

For every `engine started` line we record the config that mattered, then attach
the first measured generation and the expert-cache hit rate that follows it, up
to the next start. Print-only: nothing is invented, missing pieces say so.
"""
import re
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "strata-iq3_xxs-12gb.log"

START_RE = re.compile(r"\[strata\] (\S+ \S+) engine started: .*?strata\.exe (.*)$", re.S)
CTX_RE = re.compile(r"--max-context (\d+)")
RES_RE = re.compile(r"--vram-reserve-mib (\d+)")
FRAC_RE = re.compile(r"--pcie-frac ([0-9.]+)")
MINP_RE = re.compile(r"--spec-min-p ([0-9.]+)")
GEN_RE = re.compile(
    r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
    r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")
HIT_RE = re.compile(r"decode expert cache hit rate: ([\d.]+)% .*\((\d+) hits / (\d+) lookups\); "
                    r"(\d+) more read by the GPU over PCIe or from another GPU \(([\d.]+)% of all (\d+) routed\)")
SLOTS_RE = re.compile(r"expert cache (\d+) slots, ([\d.]+) GiB of VRAM")
FREE_RE = re.compile(r"MiB of VRAM free with everything loaded")
PROBE_RE = re.compile(r"PCIe probe: ([\d.]+) GB/s .*-> pcie_frac ([\d.]+)")
SHARE_RE = re.compile(r"PCIe share: the CPU pool takes (\d+) us per missed expert .*-> pcie_frac ([\d.]+)")
ARENA_RE = re.compile(r"expert arena: locked (\S+) MiB")
PINRE_RE = re.compile(r"KV streaming: (\d+) of (\d+) cells per QSA layer in VRAM, the K/V in ([\d.]+) GiB")

lines = open(LOG, "rb").read().decode("utf-8", "replace").splitlines()

runs = []
cur = None
for ln in lines:
    m = START_RE.search(ln)
    if m:
        stamp, args = m.group(1), m.group(2)
        cur = {
            "stamp": stamp,
            "max_context": (CTX_RE.search(args) or [None, "?"])[1],
            "reserve": (RES_RE.search(args) or [None, "-"])[1],
            "pcie_frac": (FRAC_RE.search(args) or [None, "auto"])[1],
            "spec_min_p": (MINP_RE.search(args) or [None, "?"])[1],
            "kv": "q4_0" if "--kv q4_0" in args else ("int8" if "--kv int8" in args else "?"),
            "prior": "1.5" if cur is None else "?",
            "slots": None, "free": None, "probe": None, "shares": [],
            "arena": None, "kvram": None, "hits": [], "measured": [],
        }
        runs.append(cur)
        continue
    if cur is None:
        continue
    if "expert cache" in ln and SLOTS_RE.search(ln) and cur["slots"] is None:
        cur["slots"] = SLOTS_RE.search(ln).groups()
    if "VRAM free with everything loaded" in ln and cur["free"] is None:
        cur["free"] = ln.split("serve: ")[1].split(" ")[0]
    if "PCIe probe" in ln and cur["probe"] is None:
        cur["probe"] = PROBE_RE.search(ln).groups()
    if "PCIe share:" in ln and SHARE_RE.search(ln):
        cur["shares"].append(SHARE_RE.search(ln).groups())
    if "expert arena: locked" in ln and cur["arena"] is None:
        cur["arena"] = ARENA_RE.search(ln).group(1)
    if "KV streaming:" in ln and PINRE_RE.search(ln) and cur["kvram"] is None:
        cur["kvram"] = PINRE_RE.search(ln).groups()
    if " generated in " in ln and GEN_RE.search(ln):
        g = GEN_RE.search(ln)
        cur["measured"].append({
            "prompt": int(g.group(1)), "reused": int(g.group(2)),
            "prefill": float(g.group(5)), "generated": int(g.group(6)),
            "decode": float(g.group(8)),
            "accepted": int(g.group(9)), "offered": int(g.group(10)),
        })
    if "decode expert cache hit rate" in ln and HIT_RE.search(ln):
        h = HIT_RE.search(ln)
        cur["hits"].append({"rate": float(h.group(1)), "pcie_pct": float(h.group(5))})

PRIOR_ON = {"2026-10-31 13:38:09", "2026-10-31 13:40:09",
            "2026-10-31 13:41:27", "2026-10-31 13:43:22"}


def median(vals):
    vals = sorted(vals)
    n = len(vals)
    if not n:
        return None
    return vals[n // 2] if n % 2 else round((vals[n // 2 - 1] + vals[n // 2]) / 2, 1)


print(f"{'when':19} {'ctx':7} {'res':5} {'pcie':6} {'minp':5} {'prior':6} "
      f"{'slots':6} {'free':5} {'short decode':13} {'7k prefill':11} {'hit%':6} {'pcie miss%':11}")
for r in runs:
    short = [m for m in r["measured"] if m["generated"] >= 200 and m["prompt"] < 200]
    long = [m for m in r["measured"] if m["prompt"] > 5000 and m["prompt"] < 9000]
    hits = [h["rate"] for h in r["hits"]]
    pcie = [h["pcie_pct"] for h in r["hits"]]
    print(f"{r['stamp']:19} {str(r['max_context']):7} {str(r['reserve']):5} "
          f"{str(r['pcie_frac']):6} {str(r['spec_min_p']):5} "
          f"{'1.5' if r['stamp'] in PRIOR_ON else 'off':6} "
          f"{str((r['slots'] or ['?','?'])[0]):6} {str(r['free']):5} "
          f"{str(median([m['decode'] for m in short])) + ' n=' + str(len(short)):13} "
          f"{str(median([m['prefill'] for m in long])):11} "
          f"{str(median(hits)):6} {str(median(pcie)):11}")

print()
for r in runs:
    if r["shares"]:
        print(f"{r['stamp']}: share {' -> '.join(s[1] for s in r['shares'])} "
              f"(CPU us/expert: {', '.join(s[0] for s in r['shares'])})")
    if r["arena"]:
        print(f"{r['stamp']}: arena locked {r['arena']} MiB; KV {r['kvram'][2]} GiB RAM "
              f"({r['kvram'][1]} cells, {r['kvram'][0]} resident); probe "
              f"{(r['probe'] or ['?','?'])[0]} GB/s -> {(r['probe'] or ['?','?'])[1]}")
    drafts = [(m["accepted"], m["offered"]) for m in r["measured"]
              if m["generated"] >= 200 and m["prompt"] < 200]
    if drafts:
        print(f"{r['stamp']}: drafts accepted/offered on the 256-token runs: {drafts}")
