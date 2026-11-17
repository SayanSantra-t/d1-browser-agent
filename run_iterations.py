"""run_iterations.py: Executes targeted iteration tests (1..10) across both d1-omni-600M and d1-2B
for the production Decision-Model Browser Harness.
"""

from __future__ import annotations
import json
import os
import re
import sys
import time
import traceback
from playwright.sync_api import sync_playwright

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from d1_engine import DecisionModelEngine, extract_search_text
from agent_orchestrator import DecisionAgentOrchestrator
from compare_600m_vs_2b import BROWSER_CASES, QUERY_CASES, SAFETY_CASES

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iteration_logs")
os.makedirs(LOG_DIR, exist_ok=True)


def make_logger(iter_num: int):
    path = os.path.join(LOG_DIR, f"iter_{iter_num}.txt")
    if os.path.exists(path):
        os.remove(path)

    def log(msg: str):
        print(msg, flush=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    return log, path


def infer_meta_from_desc(desc: str) -> dict:
    d = desc.lower()
    is_inp = "search_input" in d or "textbox" in d or "searchbox" in d
    role = "search_input" if is_inp else ("tab" if d.startswith("tab:") else "link")
    return {"is_input": is_inp, "role": role}


def run_offline_suite_for_model(engine: DecisionModelEngine, log):
    log(f"\n--- Offline 16-Case Decision Suite: {engine.model_label} ---")
    elem_ok = 0
    joint_ok = 0
    lats = []
    failures = []

    for case in BROWSER_CASES:
        meta = {k: infer_meta_from_desc(v) for k, v in case["elements"].items()}
        dec = engine.select_element_action(
            goal=case["goal"],
            current_url=case["url"],
            page_title=case["title"],
            elements=case["elements"],
            elements_meta=meta,
            history=case.get("history", []),
        )
        lats.append(dec["latency_ms"])
        valid_elems = case.get("alt_elems", [case["expected_elem"]])
        e_hit = dec["chosen_element"] in valid_elems
        a_hit = dec["action_type"] == case["expected_act"]
        elem_ok += int(e_hit)
        if e_hit and a_hit:
            joint_ok += 1
        else:
            failures.append(
                f"  [FAIL] {case['name']}: got=({dec['chosen_element']} @ {dec['element_confidence']:.1%}, {dec['action_type']}) "
                f"expected=({case['expected_elem']}, {case['expected_act']})"
            )

    avg_lat = sum(lats) / len(lats) if lats else 0.0
    log(
        f"  Decision Score: Element={elem_ok}/{len(BROWSER_CASES)} ({elem_ok/len(BROWSER_CASES):.0%}), "
        f"Joint={joint_ok}/{len(BROWSER_CASES)} ({joint_ok/len(BROWSER_CASES):.0%}), AvgLat={avg_lat:.1f}ms"
    )
    for f in failures:
        log(f)

    # Query extraction evaluation
    q_hits = 0
    q_lats = []
    q_fails = []
    for prompt, expected in QUERY_CASES:
        res = engine.extract_query(prompt)
        q_lats.append(res["latency_ms"])
        got_norm = re.sub(r"\s+", " ", res["query"].strip().lower())
        exp_norm = re.sub(r"\s+", " ", expected.strip().lower())
        if got_norm == exp_norm:
            q_hits += 1
        else:
            q_fails.append(f"  [Q-FAIL] '{prompt}' -> got='{res['query']}' ({res['method']}) vs exp='{expected}'")

    log(
        f"  Query Extraction: {q_hits}/{len(QUERY_CASES)} ({q_hits/len(QUERY_CASES):.0%}) | "
        f"AvgLat={(sum(q_lats)/len(q_lats) if q_lats else 0.0):.1f}ms"
    )
    for qf in q_fails:
        log(qf)

    return {"elem_ok": elem_ok, "joint_ok": joint_ok, "q_hits": q_hits, "avg_lat": avg_lat}


def run_live_cdp_both_models(start_url: str, goal: str, log, max_steps: int = 7, close_tabs: bool = True):
    results = {}
    for model_name in ["d1-omni-600M", "d1-2B"]:
        log(f"\n>>> LIVE CDP TEST [{model_name}] | Goal: '{goal}' | Start: {start_url}")
        engine = DecisionModelEngine(model_name)
        engine.load()
        orch = DecisionAgentOrchestrator(engine=engine)
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp("http://127.0.0.1:9110")
            ctx = browser.contexts[0]
            page = ctx.new_page()
            try:
                res = orch.run_cdp_task(
                    page=page,
                    start_url=start_url,
                    goal=goal,
                    max_steps=max_steps,
                    log_fn=log,
                )
                results[model_name] = res
                log(
                    f"  Result [{model_name}]: success={res['success']} | steps={res['steps']} | "
                    f"avg_lat={res['avg_latency_ms']}ms | reason={res['reason']} | url={res['final_url']}"
                )
            finally:
                if close_tabs:
                    try:
                        page.close()
                    except Exception:
                        pass
        engine.unload()
    return results


def run_iter_1(log):
    log("=== ITERATION 1: Offline 16-Case Decision + 12-Prompt Query Regression (Both Models) ===")
    for model_name in ["d1-omni-600M", "d1-2B"]:
        eng = DecisionModelEngine(model_name)
        eng.load()
        run_offline_suite_for_model(eng, log)
        eng.unload()


def run_iter_2(log):
    log("=== ITERATION 2: Live BrowserOS neo Test #1 — YouTube Channel Open ===")
    run_live_cdp_both_models("https://www.youtube.com", "open techchip youtube channel", log, max_steps=6)


def run_iter_3(log):
    log("=== ITERATION 3: Live BrowserOS neo Test #2 — Multi-Step Channel + Latest Video ===")
    run_live_cdp_both_models("https://www.youtube.com", "open greencode and play its latest video", log, max_steps=7)


def run_iter_4(log):
    log("=== ITERATION 4: Live BrowserOS neo Test #3 — Topic-Specific Video Search ===")
    run_live_cdp_both_models(
        "https://www.youtube.com",
        "find that video where techchip talks about kali linux",
        log,
        max_steps=7,
    )


def run_iter_5(log):
    log("=== ITERATION 5: Live BrowserOS neo Test #4 — Wikipedia Search & Article Open ===")
    run_live_cdp_both_models(
        "https://www.wikipedia.org",
        "find out how flash attention works on wikipedia",
        log,
        max_steps=6,
    )


def run_iter_6(log):
    log("=== ITERATION 6: Live BrowserOS neo Test #5 — GitHub Repository Navigation ===")
    run_live_cdp_both_models(
        "https://github.com/trending",
        "open the llama.cpp repository",
        log,
        max_steps=6,
    )


def run_iter_7(log):
    log("=== ITERATION 7: Edge Cases — Obstruction, Loop Avoidance & Calibrated Abstention ===")
    for model_name in ["d1-omni-600M", "d1-2B"]:
        eng = DecisionModelEngine(model_name)
        eng.load()
        orch = DecisionAgentOrchestrator(engine=eng, max_consecutive_scrolls=2)
        # Test 1: Consecutive scroll circuit breaker on channel page
        fake_channel_items = [
            {"ref": "e1", "role": "tab", "is_input": False, "label": "Home", "description": "[main] tab: Home"},
            {"ref": "e2", "role": "tab", "is_input": False, "label": "Videos", "description": "[main] tab: Videos"},
            {"ref": "e3", "role": "button", "is_input": False, "label": "Subscribe", "description": "[header] button: Subscribe"},
        ]
        orch.consecutive_scrolls = 2
        esc = orch.decide_from_candidates(
            goal="open greencode and play its latest video",
            target_query="greencode",
            url="https://www.youtube.com/@GreenCode",
            title="GreenCode - YouTube",
            raw_items=fake_channel_items,
        )
        log(f"  [{model_name}] Scroll Circuit Breaker -> ref={esc['chosen_element']} ({esc.get('description')}) escalated={esc.get('escalated')}")

        # Test 2: Abstention when target missing
        abs_dec = eng.select_element_action(
            goal="click the Download APK button",
            current_url="https://example.com",
            page_title="About Page",
            elements={
                "e1": "[main] link: About Us",
                "e2": "[main] link: Contact Support",
                "idk": "I don't know",
            },
            abstain_below=0.5,
        )
        log(
            f"  [{model_name}] Abstention Test -> chosen={abs_dec['chosen_element']} "
            f"(conf={abs_dec['element_confidence']:.1%}, act={abs_dec['action_type']})"
        )
        eng.unload()


def run_iter_8(log):
    log("=== ITERATION 8: Safety Guardrail (noul) & Multi-Tab Batch Throughput ===")
    for model_name in ["d1-omni-600M", "d1-2B"]:
        eng = DecisionModelEngine(model_name)
        eng.load()
        safe_hits = 0
        for desc, exp_risky in SAFETY_CASES:
            res = eng.check_action_safety(desc)
            ok = res["is_risky"] == exp_risky
            safe_hits += int(ok)
            log(f"  [{model_name}] Safety '{desc[:40]}' -> risky={res['is_risky']} (P={res['risk_probability']:.1%}) [{'PASS' if ok else 'FAIL'}]")
        log(f"  [{model_name}] Safety Score: {safe_hits}/{len(SAFETY_CASES)}")

        reqs = [
            (
                {"user_goal": f"find docs tab {i}", "url": f"https://example.com/t{i}"},
                {"pick": {"type": "choice", "instructions": "Pick docs link", "criteria": {"e1": "link: Home", "e2": "link: API Docs"}}},
            )
            for i in range(6)
        ]
        t0 = time.time()
        outs = eng.batch_decide(reqs)
        lat = (time.time() - t0) * 1000.0
        picks = [o["answers"]["pick"]["choice"] for o in outs]
        log(f"  [{model_name}] 6-Tab Batch: {lat:.1f}ms ({lat/6:.1f}ms/tab) | picks={picks}")
        eng.unload()


def run_iter_9(log):
    log("=== ITERATION 9: GUI & Orchestrator Integration Smoke Check ===")
    import d1_browser_gui as gui_mod
    assert hasattr(gui_mod, "D1BrowserApp"), "D1BrowserApp missing"
    eng = DecisionModelEngine("d1-2B")
    assert eng.model_label == "d1-2B"
    orch = DecisionAgentOrchestrator(engine=eng)
    assert orch.max_consecutive_scrolls == 2
    log("  GUI module & production engine/orchestrator imports verified cleanly.")


def run_iter_10(log):
    log("=== ITERATION 10: Final End-to-End Sign-Off (Offline + Live BrowserOS neo on Both Models) ===")
    for model_name in ["d1-omni-600M", "d1-2B"]:
        eng = DecisionModelEngine(model_name)
        eng.load()
        run_offline_suite_for_model(eng, log)
        eng.unload()
    run_live_cdp_both_models("https://www.youtube.com", "open techchip youtube channel", log, max_steps=6)


ITERS = {
    1: run_iter_1,
    2: run_iter_2,
    3: run_iter_3,
    4: run_iter_4,
    5: run_iter_5,
    6: run_iter_6,
    7: run_iter_7,
    8: run_iter_8,
    9: run_iter_9,
    10: run_iter_10,
}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    log, path = make_logger(n)
    try:
        ITERS[n](log)
        log(f"\n=== ITERATION {n} FINISHED SUCCESSFULLY ===")
    except Exception as e:
        log(f"\n=== ITERATION {n} ERROR: {e} ===")
        log(traceback.format_exc())
