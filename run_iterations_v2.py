"""run_iterations_v2.py: Iterations 11-20 — harder-suite regression hardening
for the production Decision-Model Browser Harness.

Usage: python run_iterations_v2.py <n>   (n = 11..20)
Logs:  iteration_logs/iter_11.txt .. iter_20.txt
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
from test_suite_v2 import (
    BROWSER_CASES_V2,
    ORCH_CASES_V2,
    QUERY_CASES_V2,
    SAFETY_CASES_V2,
    LIVE_TASKS_V2,
)
from compare_600m_vs_2b import BROWSER_CASES, QUERY_CASES, SAFETY_CASES

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iteration_logs")
os.makedirs(LOG_DIR, exist_ok=True)

MODELS = ["d1-omni-600M", "d1-2B"]


def make_logger(iter_num: int):
    path = os.path.join(LOG_DIR, f"iter_{iter_num}.txt")
    if os.path.exists(path):
        os.remove(path)

    def log(msg: str):
        print(msg, flush=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    return log, path


def infer_meta_v2(desc: str) -> dict:
    """Richer meta inference than v1: parses '[region] role: label' descriptions."""
    d = desc.strip()
    region = ""
    m = re.match(r"^\[([a-z]+)\]\s*(.*)$", d)
    if m:
        region, d = m.group(1), m.group(2)
    role, _, label = d.partition(":")
    role = role.strip().lower()
    label = label.strip()
    is_input = role in ("search_input", "textbox", "searchbox", "combobox")
    return {"is_input": is_input, "role": role or "link", "region": region, "label": label}


# ---------------------------------------------------------------------------
# Offline suites
# ---------------------------------------------------------------------------

def run_engine_suite_v2(engine: DecisionModelEngine, log):
    log(f"\n--- Offline V2 Hard Suite ({len(BROWSER_CASES_V2)} cases): {engine.model_label} ---")
    elem_ok = joint_ok = 0
    lats = []
    failures = []
    for case in BROWSER_CASES_V2:
        meta = {k: infer_meta_v2(v) for k, v in case["elements"].items()}
        dec = engine.select_element_action(
            goal=case["goal"],
            current_url=case["url"],
            page_title=case["title"],
            elements=case["elements"],
            elements_meta=meta,
            history=case.get("history", []),
            abstain_below=case.get("abstain_below"),
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
                f"  [FAIL] {case['name']}: got=({dec['chosen_element']} @ {dec['element_confidence']:.1%}, "
                f"{dec['action_type']}) expected=({case['expected_elem']}, {case['expected_act']})"
            )
    avg_lat = sum(lats) / len(lats) if lats else 0.0
    log(
        f"  V2 Decision Score: Element={elem_ok}/{len(BROWSER_CASES_V2)} ({elem_ok/len(BROWSER_CASES_V2):.0%}), "
        f"Joint={joint_ok}/{len(BROWSER_CASES_V2)} ({joint_ok/len(BROWSER_CASES_V2):.0%}), AvgLat={avg_lat:.1f}ms"
    )
    for f in failures:
        log(f)
    return {"elem_ok": elem_ok, "joint_ok": joint_ok, "avg_lat": avg_lat}


def run_query_suite_v2(engine: DecisionModelEngine, log):
    log(f"\n--- Offline V2 Query Suite ({len(QUERY_CASES_V2)} prompts): {engine.model_label} ---")
    hits = 0
    lats = []
    fails = []
    for prompt, expected in QUERY_CASES_V2:
        res = engine.extract_query(prompt)
        lats.append(res["latency_ms"])
        got = re.sub(r"\s+", " ", res["query"].strip().lower())
        exp = re.sub(r"\s+", " ", expected.strip().lower())
        if got == exp:
            hits += 1
        else:
            fails.append(f"  [Q-FAIL] '{prompt}' -> got='{res['query']}' ({res['method']}) vs exp='{expected}'")
    log(
        f"  V2 Query Extraction: {hits}/{len(QUERY_CASES_V2)} ({hits/len(QUERY_CASES_V2):.0%}) | "
        f"AvgLat={(sum(lats)/len(lats) if lats else 0.0):.1f}ms"
    )
    for f in fails:
        log(f)
    return {"q_hits": hits}


def run_safety_suite_v2(engine: DecisionModelEngine, log):
    log(f"\n--- Offline V2 Safety Suite ({len(SAFETY_CASES_V2)} cases): {engine.model_label} ---")
    hits = 0
    for desc, exp_risky in SAFETY_CASES_V2:
        res = engine.check_action_safety(desc)
        ok = res["is_risky"] == exp_risky
        hits += int(ok)
        if not ok:
            log(f"  [S-FAIL] '{desc}' -> risky={res['is_risky']} (P={res['risk_probability']:.1%}) expected_risky={exp_risky}")
    log(f"  V2 Safety Score: {hits}/{len(SAFETY_CASES_V2)}")
    return {"s_hits": hits}


def run_orch_suite_v2(engine: DecisionModelEngine, log):
    log(f"\n--- Offline V2 Orchestrator Guard Suite ({len(ORCH_CASES_V2)} cases): {engine.model_label} ---")
    hits = 0
    for case in ORCH_CASES_V2:
        orch = DecisionAgentOrchestrator(engine=engine)
        orch.consecutive_scrolls = case.get("consecutive_scrolls", 0)
        for h in case.get("history", []):
            orch.action_history.append(h)
        for v in case.get("visited", []):
            orch.clicked_history.add(v)
        exp = case["expect"]
        tq = case.get("target_query")
        if tq is None:
            tq = extract_search_text(case["goal"])
        dec = orch.decide_from_candidates(
            goal=case["goal"],
            target_query=tq,
            url=case["url"],
            title=case["title"],
            raw_items=case["items"],
        )
        ok = True
        why = []
        if "escalated" in exp and dec.get("escalated") != exp["escalated"]:
            ok = False
            why.append(f"escalated={dec.get('escalated')} != {exp['escalated']}")
        if "desc_contains" in exp and exp["desc_contains"].lower() not in dec.get("description", "").lower():
            ok = False
            why.append(f"desc='{dec.get('description')}' lacks '{exp['desc_contains']}'")
        if "not_ref" in exp and dec.get("chosen_element") == exp["not_ref"]:
            ok = False
            why.append(f"re-picked visited ref {exp['not_ref']}")
        if "is_input_pick" in exp and dec.get("action_type") != "type_and_search":
            ok = False
            why.append(f"act={dec.get('action_type')} != type_and_search")
        if "action" in exp and dec.get("action_type") != exp["action"]:
            ok = False
            why.append(f"act={dec.get('action_type')} != {exp['action']}")
        hits += int(ok)
        status = "PASS" if ok else "FAIL"
        log(
            f"  [{status}] {case['name']} -> ref={dec.get('chosen_element')} act={dec.get('action_type')} "
            f"esc={dec.get('escalated')} | {dec.get('description','')[:60]} {('; ' + '; '.join(why)) if why else ''}"
        )
    log(f"  V2 Orchestrator Guard Score: {hits}/{len(ORCH_CASES_V2)}")
    return {"o_hits": hits}


def run_v1_regression(engine: DecisionModelEngine, log):
    log(f"\n--- V1 Regression (16 cases): {engine.model_label} ---")
    def infer_meta(desc):
        d = desc.lower()
        is_inp = "search_input" in d or "textbox" in d or "searchbox" in d
        role = "search_input" if is_inp else ("tab" if d.startswith("tab:") else "link")
        return {"is_input": is_inp, "role": role}

    elem_ok = joint_ok = 0
    for case in BROWSER_CASES:
        meta = {k: infer_meta(v) for k, v in case["elements"].items()}
        dec = engine.select_element_action(
            goal=case["goal"],
            current_url=case["url"],
            page_title=case["title"],
            elements=case["elements"],
            elements_meta=meta,
            history=case.get("history", []),
        )
        valid = case.get("alt_elems", [case["expected_elem"]])
        e_hit = dec["chosen_element"] in valid
        a_hit = dec["action_type"] == case["expected_act"]
        elem_ok += int(e_hit)
        joint_ok += int(e_hit and a_hit)
        if not (e_hit and a_hit):
            log(
                f"  [V1-FAIL] {case['name']}: got=({dec['chosen_element']} @ {dec['element_confidence']:.1%}, "
                f"{dec['action_type']}) expected=({case['expected_elem']}, {case['expected_act']})"
            )
    q_hits = 0
    for prompt, expected in QUERY_CASES:
        res = engine.extract_query(prompt)
        got = re.sub(r"\s+", " ", res["query"].strip().lower())
        exp = re.sub(r"\s+", " ", expected.strip().lower())
        q_hits += int(got == exp)
    s_hits = sum(int(engine.check_action_safety(d)["is_risky"] == r) for d, r in SAFETY_CASES)
    log(
        f"  V1 Decision Joint={joint_ok}/{len(BROWSER_CASES)} ({joint_ok/len(BROWSER_CASES):.0%}) | "
        f"Query={q_hits}/{len(QUERY_CASES)} | Safety={s_hits}/{len(SAFETY_CASES)}"
    )
    return {"v1_joint": joint_ok, "v1_q": q_hits, "v1_s": s_hits}


# ---------------------------------------------------------------------------
# Live CDP
# ---------------------------------------------------------------------------

def run_live_task_v2(task: dict, log, models=None):
    results = {}
    for model_name in (models or MODELS):
        log(f"\n>>> LIVE {task['name']} [{model_name}] | Goal: '{task['goal']}'")
        engine = DecisionModelEngine(model_name)
        engine.load()
        orch = DecisionAgentOrchestrator(engine=engine)
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp("http://127.0.0.1:9110")
            ctx = browser.contexts[0]
            page = ctx.new_page()
            try:
                try:
                    res = orch.run_cdp_task(
                        page=page,
                        start_url=task["start_url"],
                        goal=task["goal"],
                        max_steps=task.get("max_steps", 6),
                        log_fn=log,
                    )
                except Exception as task_err:
                    # One live crash (overlay intercepts, nav race, ...) must not kill
                    # the whole iteration — record it and keep going.
                    log(f"  Result [{model_name}]: CRASH {type(task_err).__name__}: {str(task_err)[:200]}")
                    res = {
                        "success": False,
                        "verified": False,
                        "reason": f"crash: {type(task_err).__name__}",
                        "steps": 0,
                        "avg_latency_ms": 0.0,
                        "final_url": page.url,
                        "final_title": "",
                        "trajectory": [],
                    }
                url_ok = task["success_url_substr"].lower() in res["final_url"].lower()
                title_kw_ok = True
                if res["success"] and task.get("extra_check_title_keywords"):
                    tl = res["final_title"].lower()
                    title_kw_ok = any(k in tl for k in task["extra_check_title_keywords"])
                verified = bool(res["success"] and url_ok and title_kw_ok)
                results[model_name] = {**res, "verified": verified}
                log(
                    f"  Result [{model_name}]: success={res['success']} verified={verified} | steps={res['steps']} | "
                    f"avg_lat={res['avg_latency_ms']}ms | reason={res['reason']} | url={res['final_url']}"
                )
            finally:
                try:
                    page.close()
                except Exception:
                    pass
        engine.unload()
    return results


# ---------------------------------------------------------------------------
# Iterations 11-20
# ---------------------------------------------------------------------------

def run_iter_11(log):
    log("=== ITERATION 11: V2 Hard-Suite Baseline (26 Decision + 6 Guard + 16 Query + 10 Safety, both models) ===")
    for model_name in MODELS:
        eng = DecisionModelEngine(model_name)
        eng.load()
        run_engine_suite_v2(eng, log)
        run_orch_suite_v2(eng, log)
        run_query_suite_v2(eng, log)
        run_safety_suite_v2(eng, log)
        eng.unload()


def run_iter_12(log):
    log("=== ITERATION 12: Deterministic Target-Lock, Results-Input Suppression, Channel Pruning, Completion Hint ===")
    for model_name in MODELS:
        eng = DecisionModelEngine(model_name)
        eng.load()
        run_engine_suite_v2(eng, log)
        eng.unload()


def run_iter_13(log):
    log("=== ITERATION 13: Modal Policy, Login/Captcha Human Escalation, Multi-Input Disambiguation, Negation ===")
    for model_name in MODELS:
        eng = DecisionModelEngine(model_name)
        eng.load()
        run_engine_suite_v2(eng, log)
        run_orch_suite_v2(eng, log)
        eng.unload()


def run_iter_14(log):
    log("=== ITERATION 14: Query Extraction V2 Hardening (negations, quoted, 'page for the', numbered pins) ===")
    for model_name in MODELS:
        eng = DecisionModelEngine(model_name)
        eng.load()
        run_query_suite_v2(eng, log)
        eng.unload()


def run_iter_15(log):
    log("=== ITERATION 15: Safety V2 + Orchestrator Guard Regression ===")
    for model_name in MODELS:
        eng = DecisionModelEngine(model_name)
        eng.load()
        run_safety_suite_v2(eng, log)
        run_orch_suite_v2(eng, log)
        eng.unload()


def run_iter_16(log):
    log("=== ITERATION 16: Stress & Latency Regression (dense candidates + batch throughput) ===")
    for model_name in MODELS:
        eng = DecisionModelEngine(model_name)
        eng.load()
        run_engine_suite_v2(eng, log)
        run_v1_regression(eng, log)
        eng.unload()


def run_iter_17(log):
    log("=== ITERATION 17: Live BrowserOS neo — Google -> Official OBS Studio Download ===")
    run_live_task_v2(LIVE_TASKS_V2[0], log)


def run_iter_18(log):
    log("=== ITERATION 18: Live BrowserOS neo — PyPI requests + GitHub llama.cpp ===")
    run_live_task_v2(LIVE_TASKS_V2[1], log)
    run_live_task_v2(LIVE_TASKS_V2[2], log)


def run_iter_19(log):
    log("=== ITERATION 19: Live BrowserOS neo — YouTube Topic Video + Wikipedia Article ===")
    run_live_task_v2(LIVE_TASKS_V2[3], log)
    run_live_task_v2(LIVE_TASKS_V2[4], log)


def run_iter_20(log):
    log("=== ITERATION 20: FINAL SIGN-OFF — Full V1+V2 Offline Regression + All 5 Live Tasks (both models) ===")
    summary = {}
    for model_name in MODELS:
        eng = DecisionModelEngine(model_name)
        eng.load()
        r1 = run_engine_suite_v2(eng, log)
        r2 = run_orch_suite_v2(eng, log)
        r3 = run_query_suite_v2(eng, log)
        r4 = run_safety_suite_v2(eng, log)
        r5 = run_v1_regression(eng, log)
        summary[model_name] = {**r1, **r2, **r3, **r4, **r5}
        eng.unload()
    for task in LIVE_TASKS_V2:
        run_live_task_v2(task, log)
    log("\n=== FINAL SUMMARY ===")
    for m, s in summary.items():
        log(
            f"  {m}: V2 Joint={s['joint_ok']}/{len(BROWSER_CASES_V2)} | Guard={s['o_hits']}/{len(ORCH_CASES_V2)} | "
            f"V2 Query={s['q_hits']}/{len(QUERY_CASES_V2)} | V2 Safety={s['s_hits']}/{len(SAFETY_CASES_V2)} | "
            f"V1 Joint={s['v1_joint']}/{len(BROWSER_CASES)} | V1 Query={s['v1_q']}/{len(QUERY_CASES)} | "
            f"V1 Safety={s['v1_s']}/{len(SAFETY_CASES)}"
        )


ITERS_V2 = {
    11: run_iter_11,
    12: run_iter_12,
    13: run_iter_13,
    14: run_iter_14,
    15: run_iter_15,
    16: run_iter_16,
    17: run_iter_17,
    18: run_iter_18,
    19: run_iter_19,
    20: run_iter_20,
}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    log, path = make_logger(n)
    try:
        ITERS_V2[n](log)
        log(f"\n=== ITERATION {n} FINISHED SUCCESSFULLY ===")
    except Exception as e:
        log(f"\n=== ITERATION {n} ERROR: {e} ===")
        log(traceback.format_exc())
