"""game_agent_live.py: Flappy Bird closed-loop control benchmark.

Deterministic instrumented clone (flappy_clone.html, seeded pipe sequence, identical for
every brain) driven over Playwright CDP in a real BrowserOS tab. Each decision beat the
brain sees the SAME observation text and chooses FLAP or WAIT; the world then steps 4 frames.

Brains:
  reflex  — dumb heuristic baseline (calibrates the environment)
  600M    — d1-omni-600M system_one heads (flap/wait as a 2-candidate choice)
  2B      — d1-2B system_one heads
  gguf    — Duckets LFM3-1.7B-BrowserAgent via llama-server (JSON chat contract)

Usage: python game_agent_live.py --brain reflex|600M|2B|gguf [--seed 1234]
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

from playwright.sync_api import sync_playwright

CDP_ENDPOINT = "http://127.0.0.1:9110"
GAME_URL = "file:///" + os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "flappy_clone.html")).replace("\\", "/")

OBS_TEMPLATE = (
    "bird_y={bird_y} (0=top, 640=ground) | vy={vy} (+=falling) | "
    "next_pipe_gap_center_y={gap_y} | horizontal distance to pipe={gap_dist}px | score={score}"
)

GOAL_TEXT = "keep the flappy bird alive: flap to pass through the pipe gaps without hitting pipes or the ground"


# ---------------------------------------------------------------------------
# Brains
# ---------------------------------------------------------------------------

class ReflexBrain:
    """Heuristic baseline: flap when predicted altitude drops below the gap center."""
    name = "reflex"

    def __init__(self):
        self.latencies = []

    def decide(self, obs: dict) -> str:
        t0 = time.perf_counter()
        gap_y = obs["gap_y"]
        if gap_y is None:
            action = "flap" if obs["bird_y"] > 320 else "wait"
        elif obs["vy"] < -1.0:
            action = "wait"  # already rising on momentum — flapping again overshoots the gap
        elif obs["bird_y"] >= gap_y - 7:
            action = "flap"  # falling at/below the flap point: impulse apex stays inside the gap
        else:
            action = "wait"
        self.latencies.append((time.perf_counter() - t0) * 1000)
        return action


class D1Brain:
    """d1 system_one heads: the game is a 2-candidate choice (flap vs wait)."""
    def __init__(self, model_name: str):
        from d1_engine import DecisionModelEngine
        self.model_name = model_name
        self.name = f"d1-{model_name.split('-')[-1].lower()}"  # d1-600m / d1-2b
        self.engine = DecisionModelEngine(model_name)
        self.latencies = []

    def load(self):
        self.engine.load()

    def unload(self):
        self.engine.unload()

    def decide(self, obs: dict) -> str:
        gap_y = obs["gap_y"]
        if gap_y is None:
            below, rising = obs["bird_y"] > 320, False
        else:
            below = obs["bird_y"] >= gap_y - 7
            rising = obs["vy"] < -1.0
        if below and not rising:
            flap_desc = (f"action: FLAP now - bird_y={obs['bird_y']} is below the pipe gap "
                         f"center_y={gap_y}, staying low means hitting the bottom pipe")
            wait_desc = f"action: WAIT - keep falling away from the gap center_y={gap_y} and miss it"
        elif rising:
            flap_desc = (f"action: FLAP again while already rising - will overshoot above "
                         f"the gap center_y={gap_y} and clip the top pipe")
            wait_desc = (f"action: WAIT - already rising on momentum toward the gap "
                         f"center_y={gap_y}, do not add impulse")
        else:
            flap_desc = (f"action: FLAP early - bird_y={obs['bird_y']} is already above the "
                         f"gap center_y={gap_y}, the impulse will overshoot the gap")
            wait_desc = (f"action: WAIT - falling gently toward the gap center_y={gap_y}, "
                         f"still aligned, no impulse needed")
        candidates = {"e_flap": flap_desc, "e_wait": wait_desc}
        try:
            decision = self.engine.select_element_action(
                goal=GOAL_TEXT,
                current_url="file:///flappy_clone.html",
                page_title=OBS_TEMPLATE.format(**obs),
                elements=candidates,
                target_query="flap only when falling below the gap center",
                history=[],
            )
            chosen = decision.get("chosen_element", "e_wait")
            lat = decision.get("latency_ms", 0.0)
        except Exception:
            chosen, lat = "e_wait", 0.0
        self.latencies.append(max(lat, 0.0))
        return "flap" if chosen == "e_flap" else "wait"


class GGUFBrain:
    """Duckets GGUF via llama-server chat completions."""
    name = "gguf"

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.latencies = []

    def decide(self, obs: dict) -> str:
        if obs["gap_y"] is None:
            obs_txt = (f"bird_y={obs['bird_y']} (0=top, 640=ground) | vy={obs['vy']} (+=falling) | "
                       "no pipe visible yet | score=0")
        else:
            obs_txt = OBS_TEMPLATE.format(**obs)
        payload = json.dumps({
            "messages": [
                {"role": "system", "content": (
                    "You are playing Flappy Bird. flap = strong upward jump (the bird "
                    "rises about 56px per flap); wait = keep falling. The ground and "
                    "pipes are lethal. Rules of thumb: NEVER flap while already rising "
                    "fast (vy < -1) - momentum carries you; flap only when falling at or "
                    "below the pipe gap center (bird_y >= gap_y - 7); with no pipe "
                    "visible yet, flap to hold bird_y around 300. "
                    'Reply with ONLY one JSON object: {"action": "flap"} or {"action": "wait"}.'
                )},
                {"role": "user", "content": obs_txt + "\nOne JSON action:"},
            ],
            "max_tokens": 24,
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
            text = (data["choices"][0]["message"]["content"] or "").lower()
        except Exception:
            text = ""
        self.latencies.append((time.perf_counter() - t0) * 1000)
        return "flap" if '"flap"' in text else "wait"


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(page, brain, seed: int, frames_per_decision: int = 4,
                max_decisions: int = 220, log=print) -> dict:
    page.goto(GAME_URL, wait_until="load")
    page.wait_for_timeout(600)
    page.evaluate(f"window.flappy.reset({seed})")

    decisions, actions = 0, []
    t_start = time.perf_counter()
    while decisions < max_decisions:
        obs = page.evaluate("window.flappy.getState()")
        if not obs["alive"]:
            break
        action = brain.decide(obs)
        actions.append(action)
        if action == "flap":
            page.evaluate("window.flappy.flap()")
        page.evaluate(f"window.flappy.step({frames_per_decision})")
        decisions += 1
        if decisions % 25 == 0:
            log(f"    [{brain.name}] decision {decisions}: score={obs['score']} bird_y={obs['bird_y']}")
    wall_s = time.perf_counter() - t_start

    final = page.evaluate("window.flappy.getState()")
    result = {
        "brain": brain.name, "seed": seed, "score": final["score"],
        "alive_at_cap": final["alive"], "cause": final["cause"] or ("ran_out_of_decisions" if final["alive"] else ""),
        "decisions": decisions, "frames": final["frames"],
        "wall_time_s": round(wall_s, 1),
        "avg_decision_ms": round(sum(brain.latencies) / len(brain.latencies), 1) if brain.latencies else 0.0,
        "actions_tail": "".join("F" if a == "flap" else "." for a in actions[-40:]),
    }
    log(f"  [{brain.name}] score={result['score']} cause={result['cause']} "
        f"decisions={decisions} wall={result['wall_time_s']}s avg_decision={result['avg_decision_ms']}ms")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", required=True, choices=["reflex", "600M", "2B", "gguf"])
    ap.add_argument("--base-url", default=os.environ.get("LLAMA_BASE_URL", "http://127.0.0.1:9931"))
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.brain == "reflex":
        brain = ReflexBrain()
    elif args.brain == "600M":
        brain = D1Brain("d1-omni-600M")
    elif args.brain == "2B":
        brain = D1Brain("d1-2B")
    else:
        brain = GGUFBrain(args.base_url)

    if isinstance(brain, D1Brain):
        brain.load()

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(CDP_ENDPOINT)
        ctx = browser.contexts[0]
        page = ctx.new_page()
        try:
            result = run_episode(page, brain, args.seed)
        finally:
            try:
                page.close()
            except Exception:
                pass

    if isinstance(brain, D1Brain):
        brain.unload()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"game_results_{brain.name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
