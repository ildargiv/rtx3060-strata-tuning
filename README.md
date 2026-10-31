# Tuning Strata on one RTX 3060 12 GB: 44 → 61 tok/s decode at 262K context

Our own measurements of [Strata](https://github.com/Niko1221/Strata) (engine 0.1.54, app 0.2.8)
running Qwen3.8-Flash-Next **IQ3_XXS** on one GeForce RTX 3060 12 GB in a Ryzen 7 5700G desktop,
Windows 10. Every number here was produced on that machine. Raw data is in `results/`,
the scripts that made it are in `scripts/`.

Русская версия — в конце файла.

## Result

| Config (all at `--max-context 262144`) | Decode tok/s | Prefill tok/s | Expert-cache hit | Expert misses over PCIe |
| --- | ---: | ---: | ---: | ---: |
| Starting point, 131K context, `--pcie-frac 0.35`, `--spec-min-p 0.70` | 47.8 | 692 | — | — |
| 262K context, same other settings | 44.2 | 617 | — | — |
| + `--spec-min-p 0.5` | 44.0 | 618 | — | — |
| **+ `STRATA_ROUTE_PRIOR=1.5`** (chosen fast config) | **61.0** (code: 63.1) | 619 | 91.6% | 1.0% |
| fast config, `--vram-reserve-mib 1408` (safer VRAM guard) | 58.6 | 611 | — | — |
| fast config, `--prefill 7168` (later verified, see step 8) | 62.7 (32K) / 57.8 (131K) | 684.7 (32K) / 668.7 (131K) | — | — |

Decode is the engine's own figure from its log line (see Methodology). Prefill is the 7,063-token request.
The jump from 44 to 61 came from one setting, `STRATA_ROUTE_PRIOR=1.5`. Everything else moved the
number by less than the run-to-run noise we measured.

Sweep on the chosen fast config (fresh prompts, 256 output tokens, two repetitions each; ranges are min–max):

| Prompt tokens | Prefill tok/s (median) | Decode tok/s (median, range) |
| ---: | ---: | ---: |
| 1,153 | 233.8 | 62.2 (59.9–64.6) |
| 4,393 | 539.1 | 69.7 (65.9–73.5) |
| 34,759 | 640.9 | 56.1 (55.5–56.7) |
| 138,853 | 641.0 | 60.9 (59.8–62.0) |

Same sweep, later on the same day, with `--prefill 7168` in place of `--prefill auto` (`results/raw/bench-3060-262k-prefill7168.json`):

| Prompt tokens | Prefill tok/s (median) | Decode tok/s (median, range) |
| ---: | ---: | ---: |
| 1,152 | 228.6 | 62.0 (61.1–62.9) |
| 4,392 | 534.4 | 59.2 (53.7–64.7) |
| 34,759 | **684.7** | 62.7 (58.1–67.3) |
| 138,853 | **668.7** | 57.8 (57.4–58.2) |

Prefill at 34,759 and 138,853 tokens improved by 6.8% and 4.3%; the two short prompts moved by 2% and 1%,
which is noise. Decode moved in both directions between sizes (32K better, 131K worse), so we do not attribute
any of the decode movement to this flag. See step 8.

## Hardware (exact)

| Part | Value |
| --- | --- |
| Motherboard | ASUS PRIME X570-P, BIOS 3801 |
| CPU | AMD Ryzen 7 5700G, 8 cores / 16 threads (Zen 3, integrated Radeon, display attached here) |
| RAM | 2 × 32 GB Patriot "3200 C16" kit, **running at 3000 MT/s** (DOCP not enabled) |
| GPU | ASUS GeForce RTX 3060 12 GB, PCI ID `10de:2504`, subsystem `1043:881d`, 170 W limit |
| GPU driver | 616.92 |
| OS | Windows 10 Pro 19045, Secure Boot off |
| Storage | model files on a local SSD |
| Strata | app 0.2.8, engine 0.1.54, CUDA 13 runtime from the bundled Python environment |
| Model | Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, 42.9 GB of experts, MTP draft layer (820 MiB VRAM) |

Measured on this machine: PCIe host→device 13.4 GB/s (engine probe). CPU expert pool: 7 workers on
8 physical cores; 68–74 µs per missed expert. Under the decode load the GPU held 1935 MHz,
133–135 W, 57 °C at 99% utilisation — it was not throttling.

## How we tuned it

We started from a config that already existed and changed one thing at a time, restarting the
server and benchmarking each step. The steps, in order:

1. **Measured the starting point** at 131K context: 47.8 tok/s decode, 692 tok/s prefill, 2232 expert-cache slots.
2. **Raised the context to the model's native 262,144** with `--max-context 262144`. The model was trained
   on that length, so no RoPE scaling is used. KV is streamed: `--kv q4_0 --kv-resident 32768` keeps
   32,768 cells per layer in VRAM and the rest (1.69 GiB at 262K, vs 0.84 GiB at 131K) in pinned RAM.
   Result: 44.2 tok/s. The drop from 47.8 is **not isolated**: the 262K change also made the engine
   choose smaller prompt chunks (7168 → 6400 tokens), which explains the prefill drop (692 → 617), and we did
   not run a separate control for the decode drop.
3. **Removed the hardcoded `--pcie-frac 0.35`** so the engine could tune the PCIe share from its own
   CPU measurements. It probed 0.37, then refined to 0.25 and 0.20 from the measured CPU cost per
   missed expert (72 µs, then 68 µs). At the same 1536 reserve, hardcoded 0.35 gave 45.5 tok/s and auto gave
   46.0 — within noise. We keep the auto rule: it measures this machine instead of relying on a fixed number.
4. **Set `--spec-min-p 0.5`** (Strata's own setup default; we had 0.70). Offered drafts per 256 tokens went
   from 106 to 221 and accepted drafts from 91 to 134 (acceptance rate fell from 86% to 61%). Decode did not
   change (44.2 → 44.0). Kept for parity with the default.
5. **Turned on `STRATA_ROUTE_PRIOR=1.5`** (environment variable in the config's `env` block). The router
   prefers experts already resident in the VRAM cache. Expert-cache hit rate went from 61.6% to 91.6%;
   misses over PCIe from 6.3% to 1.0%. Decode 44 → 61 (code prompt: 63.1). This is the change that matters.
6. **Tested the VRAM reserve.** Strata warns when free VRAM is under 1024 MiB because the driver may page
   GPU memory to system RAM. With `--vram-reserve-mib 1200` the card sits at 845 MiB free and the warning
   fires. At 1408 it sits at 1053 MiB, no warning, but the expert cache shrinks from 2089 to 1963 slots and
   decode drops 61.0 → 58.6. Our sweep dipped to 689 MiB free during the 131K run (the guard fired five times
   during the sweep); decode held near 60 tok/s and we saw no paging, but this is the setting we would change first
   on a different driver or a card that also drives a display.
7. **Kept the conversation cache** (`--conversation-cache-mib 6144 --conversation-cache-min-tokens 2048
   --conversation-cache-park-gain 256`). The log shows it parking 38,009 tokens in 344 ms and refusing to park
   short requests. We did not benchmark it separately.

8. **Set `--prefill 7168`** so the chunk size stays at 7,168 tokens instead of the 6,400 the engine picked at
   262K context. This recovers part of the prefill we had at 131K context: 34,759 tokens went 640.9 → **684.7**
   tok/s and 138,853 tokens went 641.0 → **668.7** tok/s, measured the same way as the sweep above
   (`0 reused`, two repetitions). Decode moved with the noise, not with the setting: 62.7 tok/s at 32K and
   57.8 at 131K, versus 56.1 and 60.9 at the auto chunk. The engine kept the same expert-cache size
   (2,093–2,096 slots) and the same VRAM situation (742–749 MiB free after the slots, same warnings).
   Keeping `--prefill auto` was never the reason for the 262K prefill drop — the chunk size was, and fixing
   the chunk recovers most of the gap while leaving the context the same.

Two candidate settings we found in the engine's documentation are **still untested**: `--kv int8` (better
long-context quality, roughly +1.7 GiB pinned RAM, smaller VRAM cache). It is listed as a next step.

## What did not work, and our mistakes

These are in the report on purpose; several cost us time.

- **Our first benchmark was wrong.** It measured decode from the time to the first streamed chunk. The server
  flushes text in waves, so it reported **103.3 and 128.6 tok/s** where the engine's own log said 47.8 and 45.5.
  The two files are kept in `results/methodology-failures/` as the example of a measurement that looks plausible.
  Fix: read decode and prefill from the engine's log line (see Methodology).
- **A restart script reported READY while the old server was still running.** Once its child process lost the
  port race, the previous config kept serving. Several restarts overlapped. Each restart now needs a check of the
  newest `engine started` line, the PID on port 8090, and `n_ctx` from `/v1/models`.
- **A Windows PowerShell 5.1 parse failure.** A script with a non-ASCII path (Cyrillic folder name) failed to
  parse under the default code page. The restart script must stay ASCII-only and use `$PSScriptRoot`.
- **Removing `--vision` from the args did nothing.** The config also has a separate `vision` block, which adds
  the flag again, so that restart was identical to the one before it. Vision stayed on for all measurements.
- **`--spec-min-p 0.5` increased drafts by ~2× and changed nothing.** The verify window was not the limit;
  the expert cache was. We learned that from the hit-rate line, not from the speed.
- **I initially blamed the VRAM reserve for a 7% decode loss.** The data does not support that: with the prior
  off, reserve 1536 and 1200 gave 44.4 and 44.2. The reserve only mattered once the prior was on. The loss from
  131K to 262K is still unexplained, see step 2.
- **Cache-slot counts moved with unrelated changes.** The same reserve gave 2232 slots at 131K and 2089 at 262K,
  so comparing slot counts across context sizes needs care.
- **Run-to-run noise is real.** The starting config measured 49.6 tok/s once (a single longer request, earlier
  the same day) and 47.8 later. Any difference under about 3 tok/s on this machine should be treated as noise
  until repeated.
- **Every sweep was measured while our own agent was using the same server.** The decode numbers show it: two
  repetitions of the same 34,759-token prompt in one run came out 57.6 and 53.1 tok/s. The prefill figures are
  stable enough to compare (within 1% run to run); decode comparisons are not, and we should not quote a decode
  difference as a finding until it is repeated on an idle server with four repetitions per size.

## Methodology

- **Decode and prefill come from the engine's own log line**, not from the HTTP stream:
  `prompt 69 tokens = 0 reused + 69 read in 1016 ms (67.9 tok/s), 256 generated in 5793 ms (44.2 tok/s), drafts accepted 91 of 106`.
  `scripts/bench_3060.py` and `scripts/engine_log_table.py` parse it.
- **Unique nonce at the start of each prompt** (`scripts/bench_sweep.py`) so no prefix can be reused. The
  check is that the line says `0 reused`.
- **Warm-up request discarded**, then the measured requests. The expert cache warms during a session, so the
  first request is not comparable with the fifth.
- **GPU sampled during the run** with `nvidia-smi` to confirm the card was not the limit.
- **Sample size is small.** Most rows are one or two requests. The sweep has two repetitions per size. Treat
  differences under about 3 tok/s as noise.

## Connection to DeepInfer's practices

[DeepSeek's DeepInfer](https://github.com/deepseek-ai/DeepInfer) does not run on this hardware: it targets
NVIDIA SM100 and Ascend 950, Linux, CUDA 13, and its README says a KV cache system is not in the open-source
release. We did not run any of its code. We used its feature list as a checklist of what determines inference
cost, and mapped each item to a Strata setting we could measure here:

| DeepInfer practice | Strata setting we used | What we measured here |
| --- | --- | --- |
| Confidence-scheduled verification, draft length cap | `--spec 4 --spec-min-p 0.5` | Drafts offered 106 → 221 per 256 tokens; decode unchanged |
| MoE expert placement (EP / EPLB) | `--expert-cache auto` + `STRATA_ROUTE_PRIOR=1.5` | Hit rate 61.6% → 91.6%; decode 44 → 61 |
| KV cache management | `--kv q4_0 --kv-resident 32768`, `--conversation-cache-mib 6144` | 262K window with 1.69 GiB KV in RAM; 38,009 tokens parked in 344 ms |
| Asynchronous scheduling | sm_86 defaults (`STRATA_ADAPT_NOWAIT`, `STRATA_ADAPT_LAG`) and `--prefill 7168` | Chunk size 6,400 → 7,168 tokens; prefill +6.8% at 34K and +4.3% at 138K |
| Deterministic measurement | greedy decoding, log-based timings, nonce prompts | Makes the numbers comparable across restarts |

DeepInfer's prefill/decode disaggregation and data-parallel load balancing have no single-GPU equivalent,
so the analogy stops there.

## Reproduce

1. Copy `configs/strata-fast.json` (or `strata-agent.json`) and replace the `<STRATA>` and `<DATA>`
   placeholders with your install and model paths. The configs bind to `127.0.0.1`; we ran ours with LAN
   access, which is a security decision for you to make.
2. Start the server elevated on Windows (the engine locks the expert arena in RAM and needs the privilege).
3. Check the newest `engine started` line in the log and `GET /v1/models` → `n_ctx` must read `262144`.
4. Run `python scripts/bench_3060.py http://127.0.0.1:8090 out.json <engine.log>` and
   `python scripts/bench_sweep.py http://127.0.0.1:8090 sweep.json <engine.log> 2 1024,4096,32768,131072`.
5. Print the per-run table with `python scripts/engine_log_table.py <engine.log>`.

## Not in this report

- Results from other people's machines (we did not compare against them in the tables).
- Any GPU firmware or BIOS modification. These tables are software settings only. We did run a UEFI
  in-RAM unlock on this card (`10de:2504`, read-only preflight confirmed first, then one transaction); its
  measured effect on these numbers is not in these tables yet, see "Unlock" below.
- Quality evaluation of the model under each setting. `STRATA_ROUTE_PRIOR=1.5` is documented by Strata
  as able to repeat text and weaken long agentic runs. We did not evaluate that ourselves.

## Unlock attempt: what ran, and what it did not do

We ran the pre-boot unlock on this card (upstream project, GA106 `10de:2504` on an ASUS PRIME X570-P).
Read-only preflight on a new stick gave four agreeing cold-boot reports, the fourth with every gate
confirmed (`complete=1`, `result=pass`). One transaction then ran successfully: Bus Master enabled and
removed again, driver released, firmware stage loaded and DMA'd, selector write, PCIe FLR,
`FINAL status=0 stage=transaction attempted=1 driver_release=1 clean=1`.

Five later boots were refused on purpose (`attempted=0`, `clean=1`). Four of them are the guard working as
designed: the card came up warm, the selector registers read as zero, and the app does not write to a card it
cannot confirm is in the state the profile was built from. The fifth is worth reporting because the reason
changed. After a full power cycle the card read `SS0=0x13414010`, where the profile records `SS0=0x53414010`
from four earlier cold boots: one bit (bit 30) different, everything else identical — same chip, PLM, GFW,
WPR2, `SS1=6`, `fb_mb=0x3000`, BAR0 size and capability value. We do not know which of the two explanations
is true: that the bit varies on its own between cold boots (the upstream profile for this card masks bit 30
out of its SS0 gate), or that our transaction changed a latch that survived the power cycle. Either way the
guard stopped it, and nothing was written.

### What the throughput runs did and did not capture

All three runs below are the same config and the same sweep (engine log, fresh prompts, `0 reused`, two
repetitions). Files: `results/raw/bench-3060-262k-prefill7168.json`, `bench-3060-262k-post-run04.json`,
`bench-3060-262k-post-run09.json`.

| Prompt tokens | Prefill tok/s: baseline / run A / run B | Decode tok/s: baseline / run A / run B |
| ---: | ---: | ---: |
| 1,152 | 228.6 / 231.1 / 225.9 | 62.0 / 62.8 / 53.8 |
| 4,392 | 534.4 / 546.3 / 540.9 | 59.2 / 61.5 / 63.5 |
| 34,759 | 684.7 / 694.4 / 694.3 | 62.7 / 61.2 / 55.4 |
| 138,852 | 668.7 / 670.3 / 669.9 | 57.8 / 51.8 / 59.5 |

baseline = the sweep that established `--prefill 7168` (step 8); run A = the Windows session that followed
the successful transaction (15:36 boot); run B = the Windows session after the fifth, SS0-refused boot
(16:15 boot, nothing written).

**Run A did not carry the transaction, and we should have written this before publishing it.** Between the
transaction (15:31) and that Windows boot (15:35:50) four more UEFI boots happened, all refused, and each one
re-initialises the card before Windows starts. Run A therefore compares a baseline with a card of unknown
state, not unlocked with locked. Run B is the clean control: full power cycle, transaction refused, nothing
written. Its prefill is within 1% of the baseline at every size except 34,759 (694.3 against 684.7), which is
the same value run A produced.

Decode is not usable at this sample size: inside run B, two repetitions of the same prompt size differed by
7.7 tok/s, and our own agent session was sending traffic to the server during every sweep. Nothing in these
numbers separates a real effect from noise, and the honest summary is that no Windows measurement has been
taken yet that provably carries a completed transaction. The clean A/B is still owed.

## Next steps

- A clean unlock A/B: cold boot, one transaction, then Windows immediately, then the same sweep. On this
  machine that needs the SS0 gate to accept the value the card now reports on a cold boot (the profile
  currently records four cold boots that all read `0x53414010`, and the fifth refused).
- More repetitions before drawing any conclusion about decode: four repetitions per size, server idle, no
  agent traffic during the sweep.
- `--kv int8` versus `q4_0` at 262K.
- BIOS DOCP to run the RAM at its rated 3200 MT/s, since the CPU expert pool is memory-bound.
- A proper quality check of `STRATA_ROUTE_PRIOR=1.5` on long agentic tasks before recommending it for that use.

## License

Text, scripts and configs: MIT. Model weights, packs and NVIDIA components keep their own licenses.

---

# Русская версия (кратко)

**Результат.** Strata (движок 0.1.54) с моделью Qwen3.8-Flash-Next IQ3_XXS на одной RTX 3060 12 GB,
Ryzen 7 5700G, Windows 10, 64 GB RAM, контекст 262 144 токена.
Декодирование выросло с **44 до 61 tok/s** одной переменной `STRATA_ROUTE_PRIOR=1.5`: доля попаданий в кэш экспертов
поднялась с 61.6% до 91.6%, промахов через PCIe стало 1.0% вместо 6.3%. Остальное (`--spec-min-p 0.5`,
авто-доля PCIe, резерв VRAM) дало изменения в пределах шума. Prefill 640–641 tok/s на 34K и 138K токенах;
с `--prefill 7168` (фиксированный размер чанка вместо подобранного движком 6400) стало 684.7 и 668.7 tok/s,
то есть +6.8% и +4.3%. Скорость генерации при этом не изменилась в лучшую сторону.

**Разблокировка.** Preflight (только чтение) отработал на четырёх холодных запусках, четвёртый — со всеми
подтверждёнными гейтами (`complete=1`). Одна транзакция прошла: Bus Master включён/снят, драйвер прошивки
отпущен, FWSEC поднят, FLR сделан, итог `attempted=1 driver_release=1 clean=1`. Пять последующих запусков
были остановлены намеренно (`attempted=0`): четыре — потому что карта была в тёплом состоянии, пятый — по
новой причине, см. ниже.

Замеры скорости в Windows: baseline (228.6 / 534.4 / 684.7 / 668.7), прогон после транзакции
(231.1 / 546.3 / 694.4 / 670.3) и контрольный прогон после холодного запуска с остановленной транзакцией
(225.9 / 540.9 / 694.3 / 669.9 — ничего записано не было). Контрольный prefill совпадает с baseline
с точностью 1%. Но прогон «после транзакции» не может относиться к разблокировке: между транзакцией (15:31)
и загрузкой Windows (15:35:50) было ещё четыре запуска флешки, все остановленные, и каждый заново
инициализирует карту. То есть у нас пока **нет ни одного замера в Windows, который доказанно несёт
завершённую транзакцию**. Генерация вообще непригодна: два повтора одной длины в одном прогоне дали
7.7 tok/s разброса, и сервер всё время нагружался нашим же агентом.

Пятый отказ — важное наблюдение: после полного выключения питания карта читает `SS0=0x13414010`, тогда как
в профиле записано `0x53414010` с четырёх предыдущих холодных запусков. Отличается ровно один бит (бит 30),
всё остальное идентично: тот же chip, PLM, GFW, WPR2, `SS1=6`, `fb_mb=0x3000`, BAR0, capability. Две
гипотезы, и мы не знаем, какая верна: бит сам меняется между холодными запусками (в upstream-профиле этой
карты бит 30 из гейта SS0 исключён) либо транзакция изменила фиксируемый бит, который пережил выключение
питания. В любом случае защита сработала, ничего не записано.

**Железо.** ASUS PRIME X570-P (BIOS 3801), Ryzen 7 5700G (8 ядер/16 потоков), 2×32 GB Patriot 3200 C16 (работают на 3000 MT/s),
RTX 3060 12 GB ASUS (`10de:2504`, `1043:881d`), драйвер 616.92, Windows 10 Pro 19045.

**Что неудачно.** Первый бенчмарк измерял декодирование по SSE-потоку и показывал 103–128 tok/s вместо 45–48 из лога движка.
Перезапуск сервера несколько раз выдавал READY, пока старый сервер держал порт. Предположение о потере 7% из-за резерва VRAM
не подтвердилось. Падение с 131K до 262K контекста пока не объяснено изолированным тестом.

**Что дальше.** Чистое сравнение разблокировки: холодный запуск, одна транзакция, затем сразу Windows и тот же
прогон; для этого на нашей машине нужно принять то значение `SS0`, которое карта читает на холодном запуске
сейчас (профиль записывает четыре запуска по `0x53414010`, а пятый запуск отказал по иному значению). Затем
`--kv int8` против `q4_0` при 262K, DOCP для ОЗУ, и отдельно проверка качества `STRATA_ROUTE_PRIOR=1.5`
на длинных агентских задачах — прежде чем рекомендовать его для них.
