# dino_agent_live.py: Chrome Dino closed-loop control benchmark.

Deterministic instrumented clone (dino_clone.html, seeded obstacle sequence, identical for
every brain) driven over Playwright CDP in a real BrowserOS tab. Each decision beat the
brain sees the SAME observation text and chooses JUMP or WAIT; the world then steps 4 frames.

Brains:
  reflex  — dumb heuristic baseline (calibrates the environment)
  600M    — d1-omni-600M system_one heads (jump/wait as a 2-candidate choice)
  2B      — d1-2B system_one heads
  gguf    — Duckets LFM3-1.7B-BrowserAgent via llama-server (JSON chat contract)

Usage: python dino_agent_live.py --brain reflex|600M|2B|gguf [--seed 1234]

FIXED: Corrected escape character issues in system prompt
      Updated reflex and d1 brains for Dino game physics
      Added proper JSON escaping in prompts