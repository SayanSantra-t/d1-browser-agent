# Duckets LFM3-1.7B-BrowserAgent-GGUF vs d1 Harness — Live Comparison

Date: 2026-11-17 · Model: `Duckets/LFM3-1.7B-BrowserAgent-GGUF` (Q8_0, 1.85 GB, llama-server build 12428)
Agent: `gguf_agent_live.py` — same 5 live tasks, same DOM extractor (`dom_extractor.EXTRACTOR_JS`),
same verification (`success_url_substr` + title keywords) as `run_iterations_v2.run_live_task_v2`.
d1 baselines: iteration 20 sign-off logs (`iteration_logs/iter_20.txt`).

## Headline results (all verified=True)

| Task | d1-omni-600M | d1-2B | LFM3-1.7B-BrowserAgent-GGUF |
|---|---|---|---|
| L1 OBS download | 4 steps · 466 ms | 4 steps · 575 ms | **1 step · 769 ms** (direct nav obs.studio → obsproject.com) |
| L2 PyPI requests | 2 steps · 988 ms | 2 steps · 438 ms | **1 step · 873 ms** (direct nav to project URL) |
| L3 GitHub llama.cpp | 2 steps · 846 ms | 2 steps · 597 ms | **1 step · 678 ms** (direct nav to repo URL) |
| L4 YouTube wifi-security | 2 steps · 925 ms | 3 steps · 507 ms | 3 steps · 913 ms (searched; **better on-topic video**: "Every Wi Fi Security Setting Explained" vs d1's "Want a Career in Web Security?") |
| L5 Wikipedia adiabatic | 2 steps · 848 ms | 1 step · 0 ms* | 1 step · 1030 ms |
| **Totals** | 12 steps · 5/5 | 12 steps · 5/5 | **7 steps · 5/5** |

\* d1-2B L5 finished via the arrival check before its first model call (0 ms latency).

## Latency basis
- d1 numbers = engine decision timing from the iter-20 logs (same tasks, live).
- GGUF numbers = llama-server HTTP round trip per decision (prompt 348–632 tok, gen 15–73 tok, temperature 0).
- All three are the same order of magnitude (0.4–1.2 s/decision). A 1.7B Q8 on a 4 GB RTX 3050
  is **not** the latency problem one might expect, because candidate menus stay under ~650 tokens.
  On heavy pages (1K+ token snapshots) expect 2–4 s/step.

## What the GGUF agent does that the d1 models structurally cannot
1. **Direct URL navigation from world knowledge** — it one-shot L1–L3 by typing the URL it
   already knows. d1 has no navigate primitive and must always search-and-click (hence 2–4 steps).
2. **Free-form query construction** — it chose its own search text ("wifi security",
   "adiabatic process") instead of relying on the d1 `extract_query` head.
3. **Native `search` action** with its own text, from its BrowserOS-neo MCP training format:
   `{"action": "search", "ref": "e1", "text": "adiabatic process"}`.

## Raw reliability vs hardened reliability (important)
- First raw run: **1/5 verified**. Failure modes observed:
  - Hallucinated URL (`obsstudio.com` — a squatter page) burned all steps on L1 (run 1).
  - Ref confusion on YouTube: picked non-input elements (`e11`/`e15`, a `<button>`) for typing
    and looped 7 steps (the neo training format numbers refs differently than our extractor).
  - Wikipedia language drift to `es.wikipedia.org` (the same failure mode d1-600M showed in iter 19).
  - Raw `Locator.fill()` crashed on non-input targets twice.
- After three executor fixes (navigation-settle retries, native `search` mapping, is-input
  routing to the page search box): **5/5 verified, stable across reruns**.
- The d1 harness carries 20 iterations of equivalent hardening: safety gate, loop breakers,
  visited-filtering, robust click, deterministic arrival checks, human escalation.

## Verdict
- **Best raw browser agent (general tasks, quota-free): Duckets LFM3-1.7B-BrowserAgent-GGUF.**
  Same 5/5 on these navigation tasks in fewer steps, free-form actions, world-knowledge
  shortcuts, and it found the more on-topic video. This is the model that keeps Antigravity
  quota for real coding work. Needs an executor with the guards listed above (they are in
  `gguf_agent_live.py`).
- **Best embedded decision core for the d1 harness: keep d1-omni-600M.** The GGUF cannot run
  the d1 offline suites (different architecture: chat generation vs `system_one` one-pass
  heads), and inside the harness its advantages (URL priors, free text) are already covered
  better by the deterministic policy pipeline — at lower latency and ~0.7 GB less VRAM.
- **d1-2B**: no longer earning its keep vs the 600M (identical scores, 2–2.6× slower decisions,
  +230 MB VRAM). Safe to drop; keep on disk only for Span-Read extraction experiments.

## Reproduce
1. `llama-server -m C:\Users\sayan\models\lfm3-1.7b-browseragent\lfm3-1.7b-browseragent-604a-Q8_0.gguf --port 9931 -ngl 99 -c 8192`
2. BrowserOS neo CDP on `127.0.0.1:9110`
3. `python gguf_agent_live.py --tasks L1,L2,L3,L4,L5`
4. Raw per-task records: `gguf_compare_results.json`; parser/format tests: `_test_gguf_parser.py`.
