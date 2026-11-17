"""live_browseros_test.py: Live side-by-side BrowserOS neo test for d1-omni-600M vs d1-2B.
Connects to active BrowserOS neo via CDP (127.0.0.1:9110), brings tabs to foreground,
and executes real multi-step browser tasks with full telemetry.
"""

from __future__ import annotations
import gc
import os
import re
import sys
import time
import torch
from playwright.sync_api import sync_playwright
from transformers import AutoModel, BitsAndBytesConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

if "axial" not in ROPE_INIT_FUNCTIONS:
    def _axial_rope_init(config, device=None, seq_len=None, **kwargs):
        base = config.rope_parameters["rope_theta"]
        spatial_dim = config.hidden_size // config.num_attention_heads // 2
        inv_freq = 1.0 / (
            base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float, device=device) / spatial_dim)
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["axial"] = _axial_rope_init


def _patch_d1_2b_attention():
    for mod in list(sys.modules.values()):
        if getattr(mod, "__name__", "").endswith(".attention") and hasattr(mod, "_fused"):
            mod._fused = lambda q: False


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from d1_engine import extract_search_text
from d1_browser_gui import EXTRACTOR_JS

MODEL_600M = r"C:\Users\sayan\models\d1-omni-600M"
MODEL_2B = r"C:\Users\sayan\models\d1-2B"
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_test_results.txt")

_orig_print = print


def print(*args, **kwargs):
    kwargs["flush"] = True
    _orig_print(*args, **kwargs)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        _orig_print(*args, file=f, **kwargs)


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_600m():
    free_gpu()
    t0 = time.time()
    m = AutoModel.from_pretrained(
        MODEL_600M, trust_remote_code=True, dtype=torch.float16, device_map="cuda"
    )
    m.eval()
    return m, time.time() - t0, torch.cuda.max_memory_allocated() / (1024 ** 2)


def load_2b():
    free_gpu()
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    m = AutoModel.from_pretrained(
        MODEL_2B, trust_remote_code=True, quantization_config=bnb, device_map="auto"
    )
    m.eval()
    _patch_d1_2b_attention()
    return m, time.time() - t0, torch.cuda.max_memory_allocated() / (1024 ** 2)


def run_live_task(
    model,
    model_label: str,
    is_2b: bool,
    start_url: str,
    goal: str,
    max_steps: int = 8,
):
    print(f"\n{'='*72}")
    print(f"LIVE BROWSEROS TEST | {model_label}")
    print(f"Goal: '{goal}' | Start: {start_url}")
    print(f"{'='*72}")

    wants_video = bool(
        re.search(r"\b(?:video|vid|clip|watch|play|stream|first|latest)\b", goal, flags=re.IGNORECASE)
    )

    # Extract search query: use native Span-Read on d1-2B (zero regex!), regex on 600M
    if is_2b:
        t_q = time.time()
        with torch.no_grad():
            q_out = model.system_one(
                goal,
                {
                    "query": {
                        "type": "text",
                        "instructions": "Extract the exact channel name, topic, or search query to type into the search box.",
                    }
                },
            )
        q_lat = (time.time() - t_q) * 1000.0
        target_query = (q_out["answers"]["query"]["text"] or "").strip() or extract_search_text(goal)
        print(f"[Query Extraction] d1-2B Span-Read -> '{target_query}' ({q_lat:.1f} ms)")
    else:
        target_query = extract_search_text(goal)
        print(f"[Query Extraction] Regex -> '{target_query}'")

    step_latencies = []
    final_url = start_url
    final_title = ""
    success = False

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://127.0.0.1:9110")
        ctx = browser.contexts[0]
        page = ctx.new_page()
        page.bring_to_front()
        page.goto(start_url, wait_until="domcontentloaded")
        time.sleep(2.5)

        history = []
        clicked_refs = set()

        for step in range(1, max_steps + 1):
            page.bring_to_front()
            url = page.url
            title = page.title()
            final_url = url
            final_title = title

            is_video_page = "/watch" in url
            is_results_page = "/results" in url or "search?" in url
            is_channel_page = "/@" in url or "/channel/" in url or "/c/" in url

            print(f"\n--- Step {step} ---")
            print(f"URL:   {url}")
            print(f"Title: {title}")

            # Deterministic completion check
            if step > 1 and not is_results_page:
                clean_target = re.sub(r"[^a-zA-Z0-9]", "", target_query.lower())
                clean_url = re.sub(r"[^a-zA-Z0-9]", "", url.lower())
                clean_title = re.sub(r"[^a-zA-Z0-9]", "", title.lower())
                if wants_video and is_video_page:
                    print(f"[SUCCESS] Video playing in {step - 1} actions! ({title})")
                    success = True
                    break
                if not wants_video and clean_target and (
                    clean_target in clean_url or clean_target in clean_title or is_channel_page
                ):
                    print(f"[SUCCESS] Target channel/page reached in {step - 1} actions! ({title})")
                    success = True
                    break

            raw_items = page.evaluate(EXTRACTOR_JS)
            if is_channel_page and wants_video:
                vid_items = [
                    i
                    for i in raw_items
                    if "video_link" in i.get("description", "")
                ]
                if vid_items:
                    raw_items = vid_items

            # Mask previously clicked non-input elements to prevent loops
            filtered_items = [
                it
                for it in raw_items
                if it.get("is_input") or (url, it["description"]) not in clicked_refs
            ]
            if not filtered_items:
                filtered_items = raw_items

            elements_map = {it["ref"]: it for it in filtered_items}
            criteria = {it["ref"]: it["description"] for it in filtered_items}
            criteria["opt_scroll"] = "action: Scroll down the page to reveal more content"
            if is_video_page or (is_channel_page and not wants_video):
                criteria["opt_done"] = "action: Target is already open on screen, stop"

            # v1 Harness Improvement: Single element choice only (deterministic action binding)
            state = {
                "user_goal": goal,
                "target_query": target_query,
                "current_url": url,
                "current_page_title": title,
                "action_history": " -> ".join(history[-4:]) if history else "Start",
            }
            if is_channel_page and wants_video:
                instr = f"We are on '{target_query}' channel. Pick the latest video link to play."
            elif is_results_page:
                instr = f"We are on search results for '{target_query}'. Pick the matching channel or video link (not search box)."
            else:
                instr = f"Goal: '{goal}'. Target: '{target_query}'. Pick the search box or matching link."

            t0 = time.time()
            with torch.no_grad():
                out = model.system_one(
                    state,
                    {
                        "target_element": {
                            "type": "choice",
                            "instructions": instr,
                            "criteria": criteria,
                        }
                    },
                )
            lat_ms = (time.time() - t0) * 1000.0
            step_latencies.append(lat_ms)

            ans = out["answers"]["target_element"]
            chosen_ref = ans["choice"]
            conf = ans["confidence"]
            chosen_desc = criteria.get(chosen_ref, "unknown")

            # Deterministic action derivation from chosen element
            elem_info = elements_map.get(chosen_ref, {})
            is_input = elem_info.get("is_input", False) or "search_input" in chosen_desc.lower()

            if chosen_ref == "opt_done":
                derived_act = "stop"
            elif chosen_ref == "opt_scroll":
                derived_act = "scroll_down"
            elif is_input:
                derived_act = "type_and_search"
            else:
                derived_act = "click"

            print(
                f"Decision: {chosen_ref} ({chosen_desc}) -> Action: {derived_act} | "
                f"Conf: {conf:.1%} | Latency: {lat_ms:.1f} ms"
            )

            if derived_act == "stop":
                print(f"[SUCCESS] Model selected stop at step {step}.")
                success = True
                break
            elif derived_act == "scroll_down":
                page.mouse.wheel(0, 650)
                history.append("Scrolled down")
                time.sleep(1.5)
            elif derived_act == "type_and_search":
                loc = page.locator(f'[data-d1-ref="{chosen_ref}"]')
                if loc.count() == 0:
                    loc = page.locator('input[name="search_query"], input[type="text"], input[type="search"]').first
                loc.click()
                loc.fill(target_query)
                loc.press("Enter")
                history.append(f"Searched '{target_query}'")
                time.sleep(3.0)
            else:
                clicked_refs.add((url, chosen_desc))
                loc = page.locator(f'[data-d1-ref="{chosen_ref}"]')
                if loc.count() > 0:
                    loc.first.click()
                    history.append(f"Clicked {chosen_desc}")
                else:
                    print(f"[Warn] Ref {chosen_ref} disappeared, refreshing...")
                time.sleep(3.0)

        final_url = page.url
        final_title = page.title()
        page.bring_to_front()

    avg_lat = sum(step_latencies) / len(step_latencies) if step_latencies else 0.0
    print(f"\n>>> SUMMARY ({model_label}): Success={success} | Steps={len(step_latencies)} | Avg Decision Latency={avg_lat:.1f}ms")
    print(f"    Final URL:   {final_url}")
    print(f"    Final Title: {final_title}")
    return success, len(step_latencies), avg_lat, final_url, final_title


def main():
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    try:
        # 1. Run live task with d1-omni-600M
        m600, l600, v600 = load_600m()
        print(f"Loaded d1-omni-600M (FP16) in {l600:.2f}s | VRAM: {v600:.1f} MB")
        run_live_task(
            m600,
            "d1-omni-600M (FP16)",
            is_2b=False,
            start_url="https://www.youtube.com",
            goal="open techchip youtube channel",
        )
        del m600
        free_gpu()

        # 2. Run live multi-step task with d1-2B (4-bit NF4)
        m2b, l2b, v2b = load_2b()
        print(f"\nLoaded d1-2B (4-bit NF4) in {l2b:.2f}s | VRAM: {v2b:.1f} MB")
        run_live_task(
            m2b,
            "d1-2B (4-bit NF4)",
            is_2b=True,
            start_url="https://www.youtube.com",
            goal="open greencode and play its latest video",
        )
        del m2b
        free_gpu()
        print("\n=== LIVE BROWSEROS TEST COMPLETE ===")
    except Exception as e:
        import traceback
        print("LIVE TEST ERROR:", e)
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
