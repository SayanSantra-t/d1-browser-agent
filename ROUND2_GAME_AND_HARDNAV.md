# Round 2 — Flappy Bird Control + Hard Navigation (V3)

Date: 2026-11-17 · Contenders: d1-omni-600M, d1-2B, Duckets LFM3-1.7B-BrowserAgent-GGUF (Q8_0, llama-server).
Baselines/artifacts: `flappy_clone.html` (deterministic seeded clone), `game_agent_live.py`,
`hard_nav_compare.py`, `game_results_*.json`, `hard_nav_results_*.json`.

---

## Part 1 — Flappy Bird (closed-loop control, identical seed/pipes/cadence for every brain)

Beat = 4 frames; observation = same state text for all brains (bird_y, vy, gap center, distance).

| Brain | Score | Death | Decisions | Avg decision | Notes |
|---|---|---|---|---|---|
| **Reflex heuristic** (baseline) | **4** | pipe | 156 | ~0 ms | momentum rule: never flap while rising |
| d1-omni-600M (v1: plain flap/wait) | 0 | pipe | 61 | 53 ms | flapped constantly, pinned at ceiling (bird_y=16) |
| d1-omni-600M (v2: state-in-descriptions) | 0 | ground | 42 | 46 ms | over-waited after retune |
| d1-2B | 0 | pipe | 61 | 225 ms | same ceiling-pin as 600M v1 |
| GGUF (raw prompt) | 0 | ground | 11 | 219 ms | waited through the opening free-fall (first pipe spawns at frame 60; ground death at 44) |
| GGUF (+lethal-ground prompt, no-pipe phrasing) | 0 | pipe | 61 | 213 ms | survives the opening, then ceiling-pins |
| GGUF (+momentum rules in prompt) | 0 | pipe | 61 | 271 ms | still cannot apply the rule beat-by-beat |

**Verdict: none of the model brains can play.** Only the hand-coded control rule scores.
The barrier is NOT latency (GGUF decided at 213–271 ms — same order as d1 heads) — it is
**decision quality**: correct play needs a momentum/deadband control law ("never flap while
rising; flap when falling at/below gap center"), and neither the `system_one` classifier heads
nor a 1.7B chat agent can execute that loop reliably, even when the rule is spelled out in
the prompt. The GGUF was the most *steerable* brain (behavior changed with every prompt
variant); the d1 heads mostly stayed locked on "flap".

---

## Part 2 — Hard navigation V3 (same verification standard as V2 live: final-URL substring + title keywords)

Tasks: H1 deep path (python.org/downloads), H2 multi-hop (repo **/issues** page),
H3 typo ("Albert Einsteen"), H4 negation ("...not the heat capacity page").

| Task | d1-omni-600M | d1-2B | GGUF |
|---|---|---|---|
| H1 python downloads | ✅ 2 steps (`/downloads/windows`) | ✅ 2 steps (`/downloads/`) | ✅ 2 steps (clicked through to a real 3.15.2 release page) |
| H2 llama.cpp issues | ❌ 3 steps — arrival check false-fired on a third-party mirror (`issues.ecosyste.ms`, bot-check page); caught by the external URL check | ❌ 2 steps — query extractor dropped the "issues page" intent; stopped one hop early at the repo root | ✅ 2 steps — landed exactly on `llama.cpp/issues` |
| H3 Albert Einsteen | ❌ 5 steps — landed on "Albert Einstein **College of Medicine**" (wrong page; loose substring verifier also passed it — strict rescoring = FAIL) | ❌ 6 steps — *reached the exact Einstein article* but the typo broke its internal arrival check, never self-terminated, ran out of steps | ✅ 1 step — corrected the typo itself, exact article |
| H4 negation | ✅ 2 steps (engine's `parse_negations`) | ✅ 1 step | ✅ 1 step |
| **Strict tally** | **2/4** | **2/4** | **4/4** |

### Verifier lessons (feed back into the harness backlog)
1. **Substring URL checks accept prefix extensions** — `wiki/Albert_Einstein` matched
   `.../Albert_Einstein_College_of_Medicine`. Needs an exact-title / path-boundary option.
2. **The external URL verifier is earning its keep** — it caught 600M's H2 false-positive on
   a mirror domain that spoofed the token-overlap arrival check.
3. **Intent words must survive query extraction** — 2B's H2 loss came from "issues page"
   being stripped before the arrival check could require `/issues`.

---

## Round-2 bottom line
- **Game (closed-loop control): nobody beats the dumb reflex. Model brains all 0.** Small
  LLM-ish decision models are not control loops — this is exactly the kind of task to leave
  to a hand-coded policy (or a real RL agent), not to buy a bigger model for.
- **Hard navigation: GGUF 4/4 vs 2/4, 2/4.** The 1.7B BrowserAgent's world knowledge (typo
  correction, exact deep URLs) and free-text reasoning win as tasks get harder. The d1
  stack's failures are all in its deterministic layers (arrival checks, query extraction) —
  fixable, but each fix is a policy patch, while the GGUF generalizes for free.
- Combined with Round 1 (V2 live 5/5 both, GGUF 5/5 in fewer steps): **the Duckets
  LFM3-1.7B-BrowserAgent is the better general browser agent on every harder test**; the
  d1-600M remains the faster embedded brain inside the harness's pipeline.
