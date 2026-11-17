# Production Decision-Model Browser Harness — Handover & 20-Iteration Execution Log

**Project Path:** `C:\Users\sayan\OneDrive\Desktop\projects\d1-browser-agent`  
**Status:** **ALL 20 ITERATIONS COMPLETED — PRODUCTION READY (v2.0, V1+V2 suites green)**

**Supported Models (Model-Agnostic Architecture):**
- **`d1-2B`** (`C:\Users\sayan\models\d1-2B`, 4-bit NF4 via `bitsandbytes`, `~1,336–2,464 MB` VRAM) — **Recommended Default**
- **`d1-omni-600M`** (`C:\Users\sayan\models\d1-omni-600M`, FP16, `~1,120 MB` VRAM) — **Ultra-Low-Latency Lightweight Option**
- Extensible via `MODEL_PRESETS` / custom paths in `DecisionModelEngine` for any `system_one()` decision model (`d1-3B`, `d1-Omni-2B`, fine-tuned checkpoints).

**Hardware & Browser Environment:**
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM, Ampere `sm_86`)
- **Runtime:** Python 3.12, PyTorch CUDA, `transformers 5.15.0`, `bitsandbytes`, `playwright`, `customtkinter`
- **Live Browser:** BrowserOS neo (`BrowserClaw\Application\chrome.exe`) on CDP `http://127.0.0.1:9110`

---

## Production Harness Architecture (`d1-browser-agent`)

| File | Role & Production Guarantees |
| :--- | :--- |
| **`dom_extractor.py`** | Rich semantic DOM extractor: resolves SPA card titles (`#video-title`, `#channel-title`, `h1..h4`, `a[title]`), strips ad slots (`ytd-ad-slot-renderer`, promoted cards) and generic CTAs (`Watch`, `READ MORE`), tags regions (`[header]`, `[main]`, `[nav]`, `[modal]`), and builds `Choice` criteria menus. |
| **`d1_engine.py`** | Model-agnostic `DecisionModelEngine`: registers `'axial'` RoPE + SDPA `_fused = False` fallbacks for Windows Ampere GPUs; derives execution actions deterministically from element semantics (`type_and_search`, `click`, `scroll_down`, `stop`, `request_human`, `abstain`); handles single-candidate fast paths (`len == 1`); extracts search queries via `type="text"` Span-Read (`d1-2B`) with NLP fallback (`600M`); provides calibrated `idk` abstention, `noul` safety checks (with `<1B` keyword floor), and `system_one_batch()`. |
| **`agent_orchestrator.py`** | Deterministic state-machine orchestrator: consecutive-scroll circuit breaker (`max_consecutive_scrolls = 2`) with `Videos` tab / top-video escalation; visited `(url, role:label)` loop guard; portal-homepage distractor filtering; topic-aware completion verification (`check_goal_reached`); JSON trajectory recording in `trajectories/`. |
| **`d1_browser_gui.py`** | CustomTkinter desktop controller wired directly to `DecisionModelEngine` + `DecisionAgentOrchestrator` with live GPU model switcher (`d1-2B` / `d1-omni-600M`), presets, and live element probability inspector. |
| **`run_iterations.py`** | Automated 10-iteration verification & live BrowserOS neo regression runner (logs saved in `iteration_logs/iter_1.txt` .. `iter_10.txt`). |
| **`test_suite_v2.py`** | V2 hard suite: 28 engine decision cases (`BROWSER_CASES_V2`), 6 orchestrator guard cases (`ORCH_CASES_V2`), 16 harder query-extraction prompts (`QUERY_CASES_V2`), 10 safety cases (`SAFETY_CASES_V2`), 5 live CDP tasks (`LIVE_TASKS_V2`). |
| **`run_iterations_v2.py`** | Automated iterations 11–20 runner: V2 suites, V1 regression, live CDP tasks with per-task crash isolation (logs in `iteration_logs/iter_11.txt` .. `iter_20.txt`). |
| **`_stub_check.py`** | No-GPU sanity harness: replaces the model with a deliberately-wrong stub to prove the deterministic policy pipeline works independently of model quality (25/28 V2 engine + 6/6 guards). |

---

## Iterations 11–20: V2 Hard-Suite Hardening (Self-Directed Extension)

The second campaign designed a **harder V2 test suite** and drove both models to green
without regressing the original V1 suite. The core upgrade is a **deterministic policy
pipeline** in `d1_engine.select_element_action` — ordered locks that bypass the model when
the decision is unambiguous (zero forward pass):

| # | Policy | Fires when |
|---|--------|-----------|
| 1 | `login_captcha_human_escalation` | credential/captcha fields + account goal → `request_human` |
| 2 | `modal_gate_affirmative` / `modal_optout_dismiss` | age-gate affirmatives; cookie/newsletter opt-out dismissal; informational-only overlays (terms/privacy links) dropped; modal restriction for blocked content |
| 3 | temporal/ordinal video lock | "oldest/newest/first/Nth video" over dated video links |
| 4 | completion-stop | "watch till the end" goals on watch pages → `opt_done` (moved BEFORE target locks in Iter 16) |
| 5 | sort-intent lock | "popular/trending/most viewed/top rated" → sort buttons over re-clicked tabs |
| 6 | version-pin lock | "version x.y.z" literal match |
| 7 | unique full-string target lock | entire extracted query appears contiguously in exactly one candidate |
| 8 | multi-input disambiguation | 2+ inputs, no non-input overlap → best-scoring input |
| 9 | results-page input suppression | `?q=` / `/results` / `Special:Search` URLs suppress re-search |
| 10 | zero-overlap abstention | no candidate shares any topic token → `idk` |
| 9b | premature-stop veto | on watch pages, `opt_done` removed unless a topic token corroborates the title/last click |
| — | post-model override | token-overlap override with IDF-style non-discriminative-token filtering; download-intent aggregator-domain penalty |

### Iteration 11: V2 Hard-Suite Baseline
- **Status:** COMPLETED (`iteration_logs/iter_11.txt`)
- **Results:** Built `test_suite_v2.py` (28+6+16+10+5 cases) and `run_iterations_v2.py`; baseline scored the un-hardened pipeline and exposed the failure classes the next iterations fixed.

### Iteration 12: Deterministic Target-Lock, Results-Input Suppression, Channel Pruning
- **Status:** COMPLETED (`iteration_logs/iter_12.txt`)
- **Results:** d1-2B V2 **28/28 (100%)**, d1-omni-600M **26/28** after Policies 6–9 + `_postclean_span` gap-fill/wrapper-strip.

### Iteration 13: Modal Policy, Login/Captcha Escalation, Multi-Input Disambiguation
- **Status:** COMPLETED (`iteration_logs/iter_13.txt`)
- **Results:** Both models V2 **28/28**; guards 6/6 (2B) and 5/6 (600M — V2-O6 picked `opt_scroll`, fixed by removing scroll when the orchestrator narrows to a search box).

### Iteration 14: Query Extraction V2 Hardening
- **Status:** COMPLETED (`iteration_logs/iter_14.txt`)
- **Results:** V2 Query **16/16** both models (negations, quoted strings, `page for the` site-qualifier retention, numbered pins).

### Iteration 15: Safety V2 + Orchestrator Guard Regression
- **Status:** COMPLETED (`iteration_logs/iter_15.txt`)
- **Results:** V2 Safety **10/10** and Guards **6/6** on both models; safety keyword floor applied to all models with expanded destructive keywords (cancel/refund/withdraw/terminate/empty trash/permanently).

### Iteration 16: Stress & Latency Regression (V1 Backward Compat)
- **Status:** COMPLETED (`iteration_logs/iter_16.txt`)
- **Fixes:** V1-07 completion-stop ordering + Policy 5 sort-intent lock; V1-09 `_OPTOUT_CONTEXT_RE` cookie pluralization; V1-11 sort-intent bypass in the orchestrator's channel filter.
- **Results:** V1 Decision Joint **16/16 on BOTH models** (was 13/16 and 14/16), zero V1/V2 regressions.

### Iterations 17–19: Live BrowserOS neo (V2 Live Tasks L1–L5)
- **Status:** COMPLETED (`iteration_logs/iter_17.txt` .. `iter_19.txt`)
- **Live hardening fixes:** token-overlap arrival detection (host+path match in `check_goal_reached`); informational-modal-link dropping; portal-home regex tolerating query strings; visited-fallback scroll/escalate loop guard; `_robust_click` (normal → JS dispatch → scroll+force) so overlay-intercepted clicks never crash a run; per-task exception isolation in the runner; watch-page suppression of subscribe/notification buttons; premature-stop veto; action-verb tokens excluded from video completion; `page does not exist` placeholder filtering (en + it); results-page detection covering `Special:Search`/`?search=`; host-token exclusion in topic matching (no more Italian-Wikipedia drift); empty-results re-search narrowing; channel-page topic-aware video narrowing; channel-corroborated video completion; `<site> page for the <topic>` site-qualifier retention.
- **Results:** all 5 live tasks verified on both models (10/10 runs).

### Iteration 20: Final Sign-Off
- **Status:** COMPLETED (`iteration_logs/iter_20.txt`)
- **Final Scores:**

| Suite | d1-omni-600M | d1-2B |
|---|---|---|
| V2 Decision (28) | **28/28** | **28/28** |
| V2 Orchestrator Guards (6) | **6/6** | **6/6** |
| V2 Query (16) | **16/16** | **16/16** |
| V2 Safety (10) | **10/10** | **10/10** |
| V1 Decision Joint (16) | **16/16** | **16/16** |
| V1 Query (12) | 12/12 | 11/12 |
| V1 Safety (6) | **6/6** | **6/6** |
| Live V2 tasks L1–L5 | **5/5 verified** | **5/5 verified** |

The only non-perfect cell (V1 Query 11/12 on 2B) is a pre-existing Span-Read exact-match
gap documented since Iteration 1; the deterministic extractor produces the correct query.

---

## Complete 10-Iteration Execution & Fix Log

### Baseline (Pre-Iteration 1)
- **Status:** COMPLETED
- **Findings:** `d1-omni-600M` scored `3/16 (19%)` joint element+action; `d1-2B` scored `6/16 (38%)` joint and looped 6x on `opt_scroll` on `@itea-greencode` due to unlabeled YouTube links and no scroll circuit breaker.

### Iteration 1: Core Production Harness Build + Offline 16-Case & 12-Prompt Regression
- **Status:** COMPLETED (`iteration_logs/iter_1.txt`)
- **Changes & Results:** Built `dom_extractor.py`, `d1_engine.py`, and `agent_orchestrator.py`. Deterministic action binding boosted joint accuracy to **`8/16 (50%)` on `600M`** and **`11/16 (69%)` on `2B`**; query extraction reached **`12/12 (100%)`** on `600M` NLP fallback and **`7/12`** exact Span-Read on `2B`.

### Iteration 2: Live BrowserOS neo Test #1 — YouTube Channel Open (`"open techchip youtube channel"`)
- **Status:** COMPLETED (`iteration_logs/iter_2.txt`)
- **Changes & Results:** `d1-2B` succeeded in **2 steps** (`548.1 ms` avg $\rightarrow$ `https://www.youtube.com/@techchipnet`). Fixed false-positive `or is_channel` stop check in `check_goal_reached()` and added `has_search_box` + `has_modal` routing hints.

### Iteration 3: Live BrowserOS neo Test #2 — Multi-Step Channel + Latest Video (`"open greencode and play its latest video"`)
- **Status:** COMPLETED (`iteration_logs/iter_3.txt`)
- **Changes & Results:** `d1-2B` succeeded end-to-end (`https://www.youtube.com/watch?v=GuGwFUnsRx0` — `CareSpeak | Assistive Communication Using Hand & Eye Gestures | SIH 2026`). Hardened `check_goal_reached()` to verify `target_query` on video pages, filtered portal-homepage distractors when `not has_direct_match`, and disabled `opt_scroll` on channel pages when `video_link` candidates are already visible.

### Iteration 4: Live BrowserOS neo Test #3 — Topic-Specific Search (`"find that video where techchip talks about kali linux"`)
- **Status:** COMPLETED (`iteration_logs/iter_4.txt`)
- **Changes & Results:** Both models extracted `'techchip kali linux'` cleanly (`d1-2B` via Span-Read). Isolated YouTube ad/banner generic CTA links (`Watch`, `READ MORE`) and search-history leaks; patched `dom_extractor.py` to exclude ad slots/CTAs and `agent_orchestrator.py` to require topic-keyword overlap for video completion.

### Iteration 5: Live BrowserOS neo Test #4 — Wikipedia Knowledge Lookup (`"find out how flash attention works on wikipedia"`)
- **Status:** COMPLETED (`iteration_logs/iter_5.txt`)
- **Changes & Results:** `d1-omni-600M` navigated `wikipedia.org` $\rightarrow$ searched `'flash attention'` $\rightarrow$ clicked `FlashAttention` $\rightarrow$ landed on **`https://en.wikipedia.org/wiki/Transformer_(deep_learning)#FlashAttention`**. Caught `ValueError: a choice needs criteria with at least two options` when candidate filtering left 1 item; added `len(elements) == 1` fast path in `d1_engine.py`.

### Iteration 6: Live BrowserOS neo Test #5 — GitHub Repository Navigation (`"open the llama.cpp repository"`)
- **Status:** COMPLETED (`iteration_logs/iter_6.txt`)
- **Changes & Results:** `d1-2B` extracted `'llama.cpp'` via Span-Read, clicked GitHub's header quick-search trigger (`Open quick search dialog, type / to search`), typed `'llama.cpp'` into `[modal] search_input: Search or jump to...`, and clicked `ggml-org/llama.cpp` $\rightarrow$ **`https://github.com/ggml-org/llama.cpp`**. Added article stripping (`the/a/an`) + `repository/repo` suffix stripping to `extract_search_text()` and header search-trigger prioritization on GitHub.

### Iteration 7: Edge-Case Hardening — Scroll Circuit Breaker & Calibrated Abstention (`idk`)
- **Status:** COMPLETED (`iteration_logs/iter_7.txt`)
- **Changes & Results:** Verified Scroll Circuit Breaker escalates to `tab: Videos` (`scroll_circuit_breaker_videos_tab`) on both `600M` and `2B`. Replaced unsupported `abstain_below` kwarg in `SystemOneModel.system_one()` with model-agnostic probability renormalization over `probs`, achieving **`chosen=idk (conf=85.5%, act=abstain)`** on `d1-2B`.

### Iteration 8: Safety Guardrail (`noul`) & Multi-Tab Batch Throughput (`system_one_batch`)
- **Status:** COMPLETED (`iteration_logs/iter_8.txt`)
- **Changes & Results:** `d1-2B` scored **`6/6 (100%)`** on destructive-action safety detection (`P(risky)=84.3%–95.6%` on delete/pay/password removal vs `0.9%–1.2%` on safe clicks) and ran a **6-tab batch in `213.6 ms` (`35.6 ms/tab`)** (`80.6 ms` / `13.4 ms/tab` on `600M`). Added a deterministic destructive-keyword floor in `d1_engine.check_action_safety()` for `<1B` models (`not self.supports_span_read`) so `600M` also enforces `6/6` safety.

### Iteration 9: Desktop GUI Integration (`d1_browser_gui.py`) & Smoke Verification
- **Status:** COMPLETED (`iteration_logs/iter_9.txt`)
- **Changes & Results:** Upgraded `d1_browser_gui.py` to use `DecisionModelEngine` + `DecisionAgentOrchestrator` + `dom_extractor.EXTRACTOR_JS` with a live model switcher (`d1-2B` / `d1-omni-600M`), presets, and probability inspector; verified clean imports and initialization.

### Iteration 10: Final End-to-End Regression & Production Sign-Off
- **Status:** COMPLETED (`iteration_logs/iter_10.txt`)
- **Summary:** Final offline + live BrowserOS neo regression completed across both `d1-omni-600M` and `d1-2B`. All core modules, trajectory logs (`trajectories/`), and iteration reports (`iteration_logs/`) are ready for production use or handover.
