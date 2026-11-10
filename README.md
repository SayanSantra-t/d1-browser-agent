# D1 Browser Agent: System-1 Autonomous Browser Navigation

A high-speed, zero-token forward-pass autonomous browser agent architecture powered by compact foundation models on NVIDIA CUDA for real-time web interaction, DOM accessibility-tree parsing, and low-latency decision orchestration.

---

## 🌟 Key Features

- **System-1 Decision Loop:** Sub-100ms forward-pass decision inference utilizing compact multi-modal foundation models (`d1-omni-600M`).
- **DOM Accessibility Tree Parsing:** Surgical extraction of actionable UI elements with persistent coordinate and ref mapping.
- **Hardware-Accelerated Inference:** Native NVIDIA CUDA integration for instant zero-shot action selection without external API round-trips.
- **Interactive Control GUI:** PySide6 desktop control surface featuring real-time telemetry, action logs, and live browser viewport streaming.
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
├── agent_orchestrator.py    # Autonomous decision loop and step execution
├── browser_adapter.py       # Browser snapshot adapter and DOM element parser
├── d1_engine.py             # System-1 decision engine powered by PyTorch/CUDA
├── d1_browser_gui.py        # PySide6 desktop GUI and telemetry visualizer
├── benchmark_gpu.py         # GPU latency and decision throughput benchmarking
├── test_perfect_flow.py     # End-to-end integration test harness
└── launch_d1_gui.bat        # Windows quick-launcher script
```

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.10+
- PyTorch with CUDA support
- PySide6, Transformers

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

---

## 📜 License
MIT License. Developed by Sayan Santra.
