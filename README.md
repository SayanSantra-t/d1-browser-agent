# D1 Browser Agent: System-1 Autonomous Browser Navigation

A high-speed, zero-token forward-pass autonomous browser agent architecture powered by compact foundation models on NVIDIA CUDA for real-time web interaction, DOM accessibility-tree parsing, and low-latency decision orchestration.

---

## 🌟 Key Features

- **System-1 Decision Loop:** Sub-100ms forward-pass decision inference utilizing compact multi-modal foundation models (`d1-omni-600M` FP16, `d1-2B` 4-bit NF4).
- **Deterministic Policy Pipeline:** Ordered zero-forward-pass locks (login/captcha human escalation, modal dismissal, temporal/ordinal video locks, completion-stop, sort-intent, version-pin, full-string target lock, multi-input disambiguation, results-page suppression, zero-overlap abstention, premature-stop veto) that bypass the model whenever the decision is unambiguous — proven model-independent by a stub-model check.
- **DOM Accessibility Tree Parsing:** Surgical extraction of actionable UI elements with persistent coordinate and ref mapping, ad-slot/CTA filtering, and region tagging (`[header]`, `[main]`, `[nav]`, `[modal]`).
- **Guarded Orchestrator:** Consecutive-scroll circuit breaker, visited-element loop guard with scroll/escalate fallback, portal & results-page candidate narrowing, token-overlap arrival verification, and layered robust clicks (normal → JS dispatch → scroll+force) for hostile overlay layouts.
- **Safety Guardrails:** `noul` risk classification plus a deterministic destructive-keyword floor so even sub-1B models block delete/pay/credential actions, escalating to `request_human`.
- **Two Test Suites (V1 + V2):** 16-case V1 suite plus a harder 28-case V2 decision suite, 6 orchestrator guards, 16 query-extraction prompts, 10 safety cases, and 5 live BrowserOS neo CDP tasks — all green on both models (see `task.md` for the 20-iteration fix log).
- **Interactive Control GUI:** CustomTkinter desktop control surface with live GPU model switcher and element-probability inspector.
- **Benchmarking Suite:** Integrated GPU latency and decision throughput benchmarking harness.

---

## 🏗️ Architecture

```
User Goal / Task
       │
       ▼
┌──────────────────────────────────────────────┐
│       Agent Orchestrator                     │
│  - State Machine & Step Execution            │
│  - Action History & Trajectory Tracking      │
└──────────────────────┬───────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌──────────────────┐       ┌───────────────────┐
│ Browser Adapter  │       │ D1 Decision Engine│
│ - DOM Tree Snap  │       │ - PyTorch / CUDA  │
│ - Element Refs   │       │ - Candidate Match │
└────────┬─────────┘       └─────────┬─────────┘
         │                           │
         └─────────────┬─────────────┘
                       ▼
            Executed Browser Action
            (Click / Type / Scroll)
```

---

## 📂 Project Structure

```
d1-browser-agent/
├── agent_orchestrator.py                # Guarded decision loop, loop guards, arrival verification
├── dom_extractor.py                     # Semantic DOM extraction with ad/CTA filtering + region tags
├── d1_engine.py                         # System-1 decision engine + deterministic policy pipeline
├── d1_browser_gui.py                    # CustomTkinter desktop GUI and probability inspector
├── browser_adapter.py                   # Browser snapshot adapter and DOM element parser
├── benchmark_gpu.py                     # GPU latency and decision throughput benchmarking
├── test_suite_v2.py                     # V2 hard suite (28 decision + 6 guards + 16 query + 10 safety + 5 live)
├── compare_600m_vs_2b.py                # V1 suite source (16 cases + 12 query + 6 safety)
├── run_iterations.py                    # V1-era iteration runner (iter_1..iter_10)
├── run_iterations_v2.py                 # V2 iteration runner (iter_11..iter_20, live CDP + crash isolation)
├── _stub_check.py                       # No-GPU sanity check of the deterministic pipeline
├── iteration_logs/                      # iter_1.txt .. iter_20.txt execution & fix logs
├── trajectories/                        # Recorded JSON action trajectories
├── test_perfect_flow.py                 # End-to-end integration test harness
├── launch_d1_gui.bat                    # Windows quick-launcher script
├── GGUF_VS_D1_COMPARISON.md              # Round 1 benchmark results (GGUF vs d1 models)
├── flappy_clone.html                    # Instrumented Flappy Bird for control benchmark
├── game_agent_live.py                   # Game harness with reflex/D1Brain/GGUFBrain brains
├── ROUND2_GAME_AND_HARDNAV.md            # Round 2: Flappy Bird control + hard-nav V3 results
├── hard_nav_compare.py                  # Hard navigation V3 test runner (4 cases)
├── hard_nav_results_600M.json           # Results JSON for d1-omni-600M (hard-nav V3)
├── hard_nav_results_2B.json             # Results JSON for d1-2B (hard-nav V3)
├── hard_nav_results_gguf.json           # Results JSON for GGUF (hard-nav V3)
└── GGUF_VS_D1_COMPARISON.md              # Reference benchmark for Round 1 to Round 2
```

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.10+
- PyTorch with CUDA support
- customtkinter (GUI)
- playwright (for BrowserOS neo integration)
- numpy

### 2. Installation
```bash
git clone https://github.com/SayanSantra-t/d1-browser-agent.git
cd d1-browser-agent
pip install -r requirements.txt
```

### 3. Run Benchmark
```bash
python benchmark_gpu.py
```

### 4. Launch Desktop Interface
```bash
python d1_browser_gui.py
```

### 5. Run the Test Suites
```bash
python _stub_check.py                 # no-GPU: deterministic policy pipeline + guards
python run_iterations_v2.py 11..20    # V2 suites, V1 regression, live BrowserOS neo tasks
```
Live tasks require BrowserOS neo running with CDP at `http://127.0.0.1:9110`.

---

## 📜 License
MIT License. Developed by Sayan Santra.
