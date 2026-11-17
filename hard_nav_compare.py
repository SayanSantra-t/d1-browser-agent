"""hard_nav_compare.py: V3 hard-navigation cases — same standard, three brains.

Harder than the V2 live set: deep destination paths (python.org/downloads, repo /issues),
a misspelled entity (Einsteen), and a negated alternative ("...not the heat capacity page").
Same verification as run_iterations_v2: success_url_substr in final URL (+ title keywords).

Usage: python hard_nav_compare.py --brain gguf|600M|2B
"""

from __future__ import annotations

import argparse
import json
import os

from playwright.sync_api import sync_playwright

from agent_orchestrator import DecisionAgentOrchestrator
from d1_engine import DecisionModelEngine
from gguf_agent_live import LlamaClient, run_task, goal_verified
from run_iterations_v2 import run_live_task_v2  # noqa: F401  (verification parity reference)

CDP_ENDPOINT = "http://127.0.0.1:9110"

HARD_TASKS = [
    {
        "name": "H1. Deep path: official Python 3 downloads page",
        "start_url": "https://www.google.com",
        "goal": "open the downloads page for python 3 from the official python website",
        "success_url_substr": "python.org/downloads",
        "max_steps": 6,
    },
    {
        "name": "H2. Multi-hop: llama.cpp repository ISSUES page",
        "start_url": "https://www.google.com",
        "goal": "open the issues page of the ggml-org llama.cpp repository on github",
        "success_url_substr": "github.com/ggml-org/llama.cpp/issues",
        "max_steps": 7,
    },
    {
        "name": "H3. Typo robustness: Albert Einsteen",
        "start_url": "https://www.wikipedia.org",
        "goal": "open the wikipedia article about Albert Einsteen",
        "success_url_substr": "wiki/Albert_Einstein",
        "max_steps": 6,
    },
    {
        "name": "H4. Negation: adiabatic process, NOT heat capacity",
        "start_url": "https://www.wikipedia.org",
        "goal": "open the wikipedia article about the adiabatic process, not the heat capacity page",
        "success_url_substr": "wiki/Adiabatic_process",
        "max_steps": 6,
    },
]


def run_d1(model_name: str, log=print) -> dict:
    results = {}
    engine = DecisionModelEngine(model_name)
    engine.load()
    orch = DecisionAgentOrchestrator(engine=engine)
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(CDP_ENDPOINT)
        ctx = browser.contexts[0]
        page = ctx.new_page()
        try:
            for task in HARD_TASKS:
                log(f"\n>>> {task['name']} [{model_name}]")
                try:
                    res = orch.run_cdp_task(page=page, start_url=task["start_url"],
                                            goal=task["goal"], max_steps=task["max_steps"],
                                            log_fn=log)
                except Exception as err:
                    log(f"  CRASH {type(err).__name__}: {str(err)[:150]}")
                    res = {"success": False, "reason": f"crash: {type(err).__name__}",
                           "steps": 0, "avg_latency_ms": 0.0,
                           "final_url": "", "final_title": "", "trajectory": []}
                verified = bool(res["success"] and goal_verified(task, res["final_url"], res["final_title"]))
                results[task["name"]] = {**res, "verified": verified}
                log(f"  Result [{model_name}]: verified={verified} | steps={res['steps']} | "
                    f"url={res['final_url'][:90]}")
        finally:
            try:
                page.close()
            except Exception:
                pass
    engine.unload()
    return results


def run_gguf(base_url: str, log=print) -> dict:
    client = LlamaClient(base_url)
    if not client.wait_healthy():
        raise SystemExit("llama-server not healthy — start it first.")
    results = {}
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(CDP_ENDPOINT)
        ctx = browser.contexts[0]
        page = ctx.new_page()
        try:
            for task in HARD_TASKS:
                try:
                    r = run_task(page, task, client, log)
                    r["verified"] = bool(r["success"] and goal_verified(task, r["final_url"], r["final_title"]))
                    results[task["name"]] = r
                except Exception as err:
                    log(f"  CRASH {type(err).__name__}: {str(err)[:150]}")
                    results[task["name"]] = {"verified": False, "reason": f"crash: {type(err).__name__}"}
        finally:
            try:
                page.close()
            except Exception:
                pass
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", required=True, choices=["600M", "2B", "gguf"])
    ap.add_argument("--base-url", default=os.environ.get("LLAMA_BASE_URL", "http://127.0.0.1:9931"))
    args = ap.parse_args()

    if args.brain == "gguf":
        results = run_gguf(args.base_url)
    else:
        results = run_d1("d1-omni-600M" if args.brain == "600M" else "d1-2B")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"hard_nav_results_{args.brain}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== HARD NAV SUMMARY [{}]===".format(args.brain))
    for name, r in results.items():
        print(f"{name}: verified={r.get('verified')} steps={r.get('steps')} "
              f"reason={str(r.get('reason'))[:60]} url={r.get('final_url', '')[:70]}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
