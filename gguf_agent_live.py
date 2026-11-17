"""gguf_agent_live.py: Live BrowserOS neo comparison — Duckets LFM3-1.7B-BrowserAgent GGUF
(llama-server OpenAI API) vs the d1 harness baselines from iteration 20.

Runs the SAME live tasks (test_suite_v2.LIVE_TASKS_V2), the SAME DOM extractor
(dom_extractor.EXTRACTOR_JS), and the SAME verification (success_url_substr in
final URL + extra title keywords) that run_iterations_v2.run_live_task_v2 uses
for d1-omni-600M / d1-2B — only the decision head is swapped for llama-server
chat completions with a neo-style JSON tool-call contract.

Usage:  python gguf_agent_live.py [--tasks L1,L2] [--base-url http://127.0.0.1:9931]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
import urllib.error

from playwright.sync_api import sync_playwright

from dom_extractor import EXTRACTOR_JS
from test_suite_v2 import LIVE_TASKS_V2

CDP_ENDPOINT = "http://127.0.0.1:9110"
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gguf_compare_results.json")

SYSTEM_PROMPT = """You are a browser automation agent driving a real Chrome page. Each turn you receive the goal, the current URL, the page title, actions already taken, and a numbered menu of interactive elements with refs (eN). Choose exactly ONE next action.

Respond with ONLY one JSON object, no prose, no markdown fences:
{"action": "click", "ref": "eN"}       click the element with that ref
{"action": "type", "ref": "eN", "text": "..."}   click the input, type text, press Enter
{"action": "scroll"}                    scroll the page down
{"action": "done"}                      the goal is already fully achieved on this page

Rules: pick a ref that exists in the menu. If the page has a search input, strongly prefer typing the search into it; only navigate directly when you are certain of the exact URL. Never invent refs. Never repeat an action already in the history."""


# ---------------------------------------------------------------------------
# llama-server client
# ---------------------------------------------------------------------------

class LlamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def chat(self, user_msg: str, max_tokens: int = 120, temperature: float = 0.0) -> dict:
        payload = json.dumps({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        lat_ms = (time.perf_counter() - t0) * 1000.0
        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {}) or {}
        return {"text": text, "latency_ms": lat_ms,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0)}

    def wait_healthy(self, timeout_s: float = 10.0) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=timeout_s) as resp:
                return resp.status == 200
        except Exception:
            return False


def parse_action(raw_text: str, valid_refs: set) -> dict:
    """Robustly pull the action JSON out of a model reply.

    Accepts the shapes the Duckets model was trained to emit (neo-style tool
    calls) plus the plain contract:  {"name"/"tool"/"action": ..., ...}.
    """
    text = raw_text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?|```", "", text).strip()

    candidate_jsons = []
    for m in re.finditer(r"\{", text):
        depth, start = 0, m.start()
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate_jsons.append(text[start:i + 1])
                    break
    candidate_jsons.append(text)  # last resort: try to lenient-parse the whole thing

    for cand in candidate_jsons:
        try:
            obj = json.loads(cand)
        except Exception:
            cleaned = re.sub(r",\s*([}\]])", r"\1", cand)  # trailing commas
            try:
                obj = json.loads(cleaned)
            except Exception:
                continue
        if not isinstance(obj, dict):
            continue

        name = str(obj.get("name") or obj.get("tool") or obj.get("action") or "").lower()
        args = obj.get("arguments") or obj.get("args") or obj
        if not isinstance(args, dict):
            args = {}

        # Observed Duckets native shape: the element ref itself rides in
        # "tool"/"name" ({"tool": "e12", "args": {"action": "link", ...}}).
        if re.fullmatch(r"(?:e\d+|opt_[a-z_]+)", name):
            ref_candidate = name
            inner = str(args.get("action") or args.get("kind") or "").lower()
            if "scroll" in inner:
                return {"kind": "scroll"}
            if inner in ("done", "stop", "finish") or "done" in inner:
                return {"kind": "done"}
            if "type" in inner or "fill" in inner:
                return {"kind": "type", "ref": ref_candidate, "text": str(args.get("value") or "")}
            if not valid_refs or ref_candidate in valid_refs:
                return {"kind": "click", "ref": ref_candidate}
            return {"kind": "scroll"}

        kind = str(args.get("kind") or args.get("action") or name).lower()
        ref = str(args.get("ref") or args.get("element") or obj.get("ref") or "")
        txt = str(args.get("value") or args.get("text") or obj.get("text") or "")

        if "navigate" in name or "navigate" in kind:
            return {"kind": "navigate", "url": txt or str(args.get("url") or "")}
        if "search" in kind or "search" in name:
            # Native Duckets action: search with the query in value/text
            return {"kind": "type", "ref": ref, "text": txt or str(args.get("value") or "")}
        if "type" in kind or "fill" in kind or name in ("type", "fill"):
            return {"kind": "type", "ref": ref, "text": txt}
        if "scroll" in kind or name == "scroll":
            return {"kind": "scroll"}
        if kind in ("done", "stop", "finish", "complete") or name in ("done", "stop", "finish"):
            return {"kind": "done"}
        if "click" in kind or name == "act":
            if ref.startswith("opt_scroll"):
                return {"kind": "scroll"}
            if ref.startswith("opt_done"):
                return {"kind": "done"}
            if ref.startswith("e") and (not valid_refs or ref in valid_refs):
                return {"kind": "click", "ref": ref}
            # act without a usable ref on a search input is a lost cause -> scroll
            return {"kind": "scroll"}

    # No JSON at all: salvage "click e12" style prose, else scroll to make progress
    m = re.search(r"\be(\d+)\b", text)
    if m and f"e{m.group(1)}" in valid_refs:
        return {"kind": "click", "ref": f"e{m.group(1)}"}
    return {"kind": "unparsed"}


# ---------------------------------------------------------------------------
# Action execution (mirrors the d1 harness timings/fallbacks)
# ---------------------------------------------------------------------------

def robust_click(page, ref: str, log) -> bool:
    loc = page.locator(f'[data-d1-ref="{ref}"]')
    if loc.count() == 0:
        log(f"    ref {ref} stale/missing")
        return False
    try:
        loc.first.click(timeout=8000)
        return True
    except Exception as err:
        log(f"    normal click failed ({type(err).__name__}), trying JS fallback")
    try:
        loc.first.evaluate("el => el.click()")
        return True
    except Exception as err:
        log(f"    JS click failed ({type(err).__name__}), trying scroll+force")
    try:
        loc.first.scroll_into_view_if_needed(timeout=5000)
        loc.first.click(timeout=8000, force=True)
        return True
    except Exception as err:
        log(f"    click failed permanently: {err}")
        return False


def goal_verified(task: dict, url: str, title: str) -> bool:
    url_ok = task["success_url_substr"].lower() in url.lower()
    title_kw_ok = True
    if task.get("extra_check_title_keywords"):
        title_kw_ok = any(k in title.lower() for k in task["extra_check_title_keywords"])
    return url_ok and title_kw_ok


def settle_call(fn, retries: int = 4, delay_s: float = 1.5):
    """Playwright calls raise 'Execution context was destroyed' while a page
    navigates underneath us (redirect chains, error pages). Retry until stable."""
    last = None
    for _ in range(retries):
        try:
            return fn()
        except Exception as err:
            last = err
            time.sleep(delay_s)
    raise last


def run_task(page, task: dict, client: LlamaClient, log) -> dict:
    log(f"\n>>> LIVE {task['name']} [LFM3-1.7B-BrowserAgent-GGUF] | Goal: '{task['goal']}'")
    page.bring_to_front()
    page.goto(task["start_url"], wait_until="domcontentloaded")
    page.wait_for_timeout(2200)

    history: list = []
    latencies: list = []
    tok_prompt = tok_completion = 0
    trajectory: list = []
    success = False
    reason = "max_steps_reached"
    max_steps = task.get("max_steps", 6)

    for step in range(1, max_steps + 1):
        url, title = settle_call(lambda: (page.url, page.title()))
        if goal_verified(task, url, title):
            success = True
            reason = f"Goal URL/title verified at step {step}"
            log(f"[Step {step}] DONE: {reason}")
            break

        raw_items = settle_call(lambda: page.evaluate(EXTRACTOR_JS, {"maxCandidates": 20}))
        menu_lines = [f"{it['ref']} {it['description']}" for it in raw_items]
        menu_lines.append("opt_scroll scroll the page down to reveal more elements")
        menu_lines.append("opt_done the goal is already fully achieved on this page")
        valid_refs = {it["ref"] for it in raw_items} | {"opt_scroll", "opt_done"}

        user_msg = (
            f"Goal: {task['goal']}\n"
            f"Current URL: {url}\n"
            f"Page title: {title}\n"
            f"Actions already taken: {json.dumps(history) if history else 'none'}\n"
            f"Element menu:\n" + "\n".join(menu_lines) + "\n"
            f"One JSON action:"
        )

        try:
            reply = client.chat(user_msg)
        except Exception as err:
            reason = f"llama-server error: {type(err).__name__}: {str(err)[:120]}"
            log(f"[Step {step}] {reason}")
            break

        latencies.append(reply["latency_ms"])
        tok_prompt += reply["prompt_tokens"]
        tok_completion += reply["completion_tokens"]
        action = parse_action(reply["text"], valid_refs)
        kind = action["kind"]

        log(f"[Step {step}] {url} -> {kind} {action.get('ref','')} | "
            f"raw={reply['text'][:90]!r} | {reply['latency_ms']:.0f}ms "
            f"(prompt {reply['prompt_tokens']} tok, gen {reply['completion_tokens']} tok)")
        trajectory.append({"step": step, "url": url, "title": title,
                           "action": action, "raw_reply": reply["text"][:200],
                           "latency_ms": round(reply["latency_ms"], 1)})

        if kind == "unparsed":
            history.append("unparseable reply; model output ignored")
            page.wait_for_timeout(800)
            continue

        if kind == "done":
            success = goal_verified(task, page.url, page.title())
            reason = "model_selected_done (verified)" if success else "model_selected_done but goal URL not reached"
            break

        if kind == "scroll":
            page.mouse.wheel(0, 700)
            history.append("Scrolled down")
            page.wait_for_timeout(1500)
            continue

        if kind == "type":
            text = action.get("text", "") or task["goal"]
            ref = action.get("ref", "")
            loc = page.locator(f'[data-d1-ref="{ref}"]')
            target = None
            if ref and loc.count() > 0:
                try:
                    tag = loc.first.evaluate("el => el.tagName.toLowerCase()")
                    editable = loc.first.evaluate(
                        "el => !!el.isContentEditable || !!el.closest('[contenteditable]')")
                    if tag in ("input", "textarea", "select") or editable:
                        robust_click(page, ref, log)
                        target = loc.first
                    else:
                        log(f"    ref {ref} is a <{tag}>, not an input — routing to page search input")
                except Exception as err:
                    log(f"    ref {ref} tag probe failed: {err}")
            if target is None:
                target = page.locator('input[name="search_query"], input[type="search"], '
                                      'input[type="text"], input:not([type])').first
                try:
                    target.click(timeout=5000)
                except Exception as err:
                    log(f"    generic input click failed: {err}")
            typed = False
            try:
                target.fill(text, timeout=8000)
                target.press("Enter", timeout=8000)
                typed = True
            except Exception as err:
                # fill() rejects comboboxes/custom widgets and stale refs —
                # raw keyboard input works on anything focusable.
                log(f"    fill failed ({type(err).__name__}), typing via keyboard")
                try:
                    page.keyboard.type(text, delay=25)
                    page.keyboard.press("Enter")
                    typed = True
                except Exception as err2:
                    log(f"    keyboard typing failed: {err2}")
            history.append(f"Typed '{text}' and searched" if typed else f"Type failed for '{text}'")
            page.wait_for_timeout(2800)
            continue

        if kind == "navigate":
            target = action.get("url", "")
            if target:
                if not target.startswith("http"):
                    target = "https://" + target
                page.goto(target, wait_until="domcontentloaded")
                history.append(f"Navigated to {target}")
                page.wait_for_timeout(2200)
                continue
            history.append("navigate with empty url; ignored")
            continue

        # default: click
        ref = action.get("ref", "")
        meta = next((it for it in raw_items if it["ref"] == ref), None)
        if robust_click(page, ref, log):
            history.append(f"Clicked {meta['description'] if meta else ref}")
        page.wait_for_timeout(2800)

    wall_s = sum(latencies) / 1000.0
    result = {
        "model": "LFM3-1.7B-BrowserAgent-GGUF",
        "goal": task["goal"],
        "success": success,
        "reason": reason,
        "steps": len(trajectory),
        "avg_decision_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "total_decision_time_s": round(wall_s, 1),
        "prompt_tokens_total": tok_prompt,
        "completion_tokens_total": tok_completion,
        "final_url": settle_call(lambda: page.url),
        "final_title": settle_call(lambda: page.title()),
        "verified": bool(success and goal_verified(task, page.url, page.title())),
        "trajectory": trajectory,
    }
    log(f"  Result [LFM3-1.7B-GGUF]: success={result['success']} verified={result['verified']} | "
        f"steps={result['steps']} | avg_lat={result['avg_decision_latency_ms']}ms | "
        f"reason={result['reason']} | url={result['final_url']}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("LLAMA_BASE_URL", "http://127.0.0.1:9931"))
    ap.add_argument("--tasks", default="L1,L2,L3,L4,L5")
    args = ap.parse_args()

    client = LlamaClient(args.base_url)
    if not client.wait_healthy():
        raise SystemExit(f"llama-server at {args.base_url} is not healthy — start it first.")

    wanted = {t.strip().upper() for t in args.tasks.split(",")}
    tasks = [t for t in LIVE_TASKS_V2 if t["name"].split(".")[0].strip() in wanted]

    all_results = {}
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(CDP_ENDPOINT)
        ctx = browser.contexts[0]
        page = ctx.new_page()
        try:
            for task in tasks:
                try:
                    all_results[task["name"]] = run_task(page, task, client, print)
                except Exception as task_err:
                    print(f"  CRASH {type(task_err).__name__}: {str(task_err)[:200]}")
                    all_results[task["name"]] = {"success": False, "verified": False,
                                                 "reason": f"crash: {type(task_err).__name__}",
                                                 "steps": 0, "avg_decision_latency_ms": 0.0,
                                                 "total_decision_time_s": 0.0,
                                                 "prompt_tokens_total": 0, "completion_tokens_total": 0,
                                                 "final_url": "", "final_title": "", "trajectory": []}
        finally:
            try:
                page.close()
            except Exception:
                pass

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== GGUF LIVE SUMMARY ===")
    for name, r in all_results.items():
        print(f"{name}: verified={r.get('verified')} steps={r.get('steps')} "
              f"avg_lat={r.get('avg_decision_latency_ms')}ms url={r.get('final_url','')[:80]}")
    print(f"\nSaved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
