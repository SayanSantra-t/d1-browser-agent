"""d1_browser_gui.py: CustomTkinter GUI for D1-Omni System-1 Browser Automation.
Features:
- Dynamic stopping (System-1 D1 stop decision + Target reached verification, no fixed 9-step limits)
- Robust typing system with automated keyword/intent extraction and optional override
- Sub-second System-1 zero-token forward pass latency on CUDA (~23ms)
- Direct Chrome DevTools Protocol (CDP) bridge to BrowserOS neo
"""

from __future__ import annotations
import os
import sys
import re
import time
import threading
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk

# Ensure package path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from d1_engine import D1DecisionEngine, extract_search_text
from browser_adapter import BrowserSnapshotAdapter

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

EXTRACTOR_JS = """
() => {
    document.querySelectorAll('[data-d1-ref]').forEach(el => el.removeAttribute('data-d1-ref'));
    const candidates = [];
    const seen = new Set();
    let counter = 1;
    
    // 1. Prioritize search/text inputs
    const inputs = document.querySelectorAll(
        'input[type="text"], input[type="search"], input:not([type]), textarea, [role="searchbox"], [role="combobox"]'
    );
    for (const el of inputs) {
        const rect = el.getBoundingClientRect();
        if (rect.width < 10 || rect.height < 10) continue;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden') continue;
        
        const ref = `e${counter++}`;
        el.setAttribute('data-d1-ref', ref);
        let label = el.getAttribute('placeholder') || el.getAttribute('aria-label') || el.getAttribute('title') || el.getAttribute('name') || el.value || 'Search query input box';
        candidates.push({
            ref: ref,
            tag: el.tagName.toLowerCase(),
            is_input: true,
            description: `search_input: ${label.trim()}`
        });
        if (candidates.length >= 2) break;
    }

    // 2. Collect visible links, buttons, channels from main content
    let selector = '#contents button, #contents a[href], ytd-search button, ytd-search a[href], [role="main"] button, [role="main"] a[href], main button, main a[href]';
    let elements = document.querySelectorAll(selector);
    if (!elements || elements.length < 5) {
        elements = document.querySelectorAll('button, a[href], [role="button"], [role="link"]');
    }

    for (const el of elements) {
        // Exclude sidebar navigation drawers
        if (el.closest('#guide, ytd-guide-renderer, #guide-content, nav:not([role="navigation"])')) continue;
        const rect = el.getBoundingClientRect();
        if (rect.width < 8 || rect.height < 8) continue;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue;
        if (el.getAttribute('aria-hidden') === 'true') continue;
        if (el.hasAttribute('data-d1-ref')) continue;
        if (el.parentElement && el.parentElement.closest('[data-d1-ref]')) continue;
        
        let label = el.innerText || el.textContent || '';
        label = label.trim().replace(/\\s+/g, ' ');
        if (!label) {
            label = el.getAttribute('aria-label') || el.getAttribute('title') || '';
        }
        if (label.length > 60) label = label.slice(0, 57) + '...';
        
        const tag = el.tagName.toLowerCase();
        let role = tag;
        if (tag === 'a') {
            role = 'link';
            const href = el.getAttribute('href') || '';
            if (href.startsWith('/@') || href.includes('/channel/') || href.includes('/c/')) {
                role = 'channel_link';
            } else if (href.includes('watch?v=')) {
                role = 'video_link';
            }
        }
        
        const desc = `${role}: ${label || 'unlabeled'}`;
        const key = desc.toLowerCase();
        if (seen.has(key)) continue;
        seen.add(key);

        const ref = `e${counter++}`;
        el.setAttribute('data-d1-ref', ref);
        
        candidates.push({
            ref: ref,
            tag: tag,
            is_input: false,
            description: desc
        });
        if (candidates.length >= 40) break;
    }
    return candidates;
}
"""

class D1BrowserApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("D1-Omni System-1 Browser Automation Controller")
        self.geometry("1100x780")
        self.minsize(950, 650)

        self.engine = None
        self.is_running = False

        self._build_ui()
        self._async_load_d1()

    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, height=60, corner_radius=0)
        header.pack(fill="x", side="top")
        
        ctk.CTkLabel(
            header,
            text="⚡ D1-Omni System-1 Browser Agent",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(side="left", padx=20, pady=12)

        self.lbl_status = ctk.CTkLabel(
            header,
            text="● Initializing D1 (CUDA)...",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#ffc107"
        )
        self.lbl_status.pack(side="right", padx=20)

        # Tabs
        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=15, pady=10)

        self.tab_auto = self.tabs.add("Autonomous Browser Agent")
        self.tab_inspector = self.tabs.add("Live Element Scorer")
        self.tab_info = self.tabs.add("Hardware & Architecture")

        self._build_auto_tab()
        self._build_inspector_tab()
        self._build_info_tab()

    def _build_auto_tab(self):
        frame = self.tab_auto

        # Inputs Card
        card = ctk.CTkFrame(frame)
        card.pack(fill="x", padx=10, pady=10)

        # Row 0: Target URL
        ctk.CTkLabel(card, text="Target URL:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=6, sticky="w")
        self.entry_url = ctk.CTkEntry(card, width=500, placeholder_text="https://www.youtube.com")
        self.entry_url.insert(0, "https://www.youtube.com")
        self.entry_url.grid(row=0, column=1, padx=10, pady=6, sticky="w")

        # Row 1: User Goal
        ctk.CTkLabel(card, text="User Goal:", font=ctk.CTkFont(weight="bold")).grid(row=1, column=0, padx=10, pady=6, sticky="w")
        self.entry_goal = ctk.CTkEntry(card, width=500, placeholder_text="open techchip youtube channel")
        self.entry_goal.insert(0, "open techchip youtube channel")
        self.entry_goal.grid(row=1, column=1, padx=10, pady=6, sticky="w")

        # Row 2: Search Query Override
        ctk.CTkLabel(card, text="Target Query (Opt):", font=ctk.CTkFont(weight="bold")).grid(row=2, column=0, padx=10, pady=6, sticky="w")
        self.entry_query = ctk.CTkEntry(card, width=500, placeholder_text="(Leave empty for automatic query extraction)")
        self.entry_query.grid(row=2, column=1, padx=10, pady=6, sticky="w")

        # Row 3: Presets & Controls
        chips_frame = ctk.CTkFrame(card, fg_color="transparent")
        chips_frame.grid(row=3, column=1, sticky="w", padx=10, pady=(0, 8))
        ctk.CTkLabel(chips_frame, text="Presets:").pack(side="left", padx=(0, 5))
        ctk.CTkButton(chips_frame, text="TechChip YouTube", width=130, height=24,
                      command=lambda: self._set_preset("https://www.youtube.com", "open techchip youtube channel", "techchip")).pack(side="left", padx=4)
        ctk.CTkButton(chips_frame, text="Wikipedia Search", width=120, height=24,
                      command=lambda: self._set_preset("https://www.wikipedia.org", "Search for Quantum Computing on Wikipedia", "Quantum Computing")).pack(side="left", padx=4)
        ctk.CTkButton(chips_frame, text="Google Liquid AI", width=120, height=24,
                      command=lambda: self._set_preset("https://www.google.com", "Search Liquid AI D1 on Google", "Liquid AI D1")).pack(side="left", padx=4)

        # Action Buttons
        btn_box = ctk.CTkFrame(card, fg_color="transparent")
        btn_box.grid(row=0, column=2, rowspan=3, padx=15, pady=8)

        self.btn_start = ctk.CTkButton(
            btn_box,
            text="▶ Start Agent",
            font=ctk.CTkFont(weight="bold"),
            fg_color="#28a745",
            hover_color="#218838",
            width=130,
            height=40,
            command=self.start_automation
        )
        self.btn_start.pack(pady=4)

        self.btn_stop = ctk.CTkButton(
            btn_box,
            text="⏹ Stop",
            font=ctk.CTkFont(weight="bold"),
            fg_color="#dc3545",
            hover_color="#c82333",
            width=130,
            height=32,
            state="disabled",
            command=self.stop_automation
        )
        self.btn_stop.pack(pady=4)

        # Log Output Feed
        ctk.CTkLabel(frame, text="Live System-1 Decision Execution Feed:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=15, pady=(5, 2))
        self.txt_log = ctk.CTkTextbox(frame, wrap="word", font=ctk.CTkFont(family="Consolas", size=12))
        self.txt_log.pack(fill="both", expand=True, padx=10, pady=5)
        self.txt_log.insert("end", "System ready. D1 is equipped with dynamic stopping and automated search query extraction.\n\n")

    def _set_preset(self, url: str, goal: str, query: str = ""):
        self.entry_url.delete(0, "end")
        self.entry_url.insert(0, url)
        self.entry_goal.delete(0, "end")
        self.entry_goal.insert(0, goal)
        self.entry_query.delete(0, "end")
        if query:
            self.entry_query.insert(0, query)

    def _build_inspector_tab(self):
        frame = self.tab_inspector
        top = ctk.CTkFrame(frame)
        top.pack(fill="x", padx=10, pady=10)

        ctk.CTkButton(top, text="🔍 Snapshot Active BrowserOS Tab", command=self.inspect_active_tab).pack(side="left", padx=10, pady=10)
        ctk.CTkLabel(top, text="Question/Goal:").pack(side="left", padx=10)
        self.entry_insp_q = ctk.CTkEntry(top, width=400)
        self.entry_insp_q.insert(0, "Which element should be clicked?")
        self.entry_insp_q.pack(side="left", padx=5)

        ctk.CTkButton(top, text="⚡ Score Elements with D1", fg_color="#007bff", command=self.score_elements_inspector).pack(side="left", padx=10)

        self.txt_inspector = ctk.CTkTextbox(frame, wrap="none", font=ctk.CTkFont(family="Consolas", size=12))
        self.txt_inspector.pack(fill="both", expand=True, padx=10, pady=10)

    def _build_info_tab(self):
        frame = self.tab_info
        info_box = ctk.CTkTextbox(frame, wrap="word", font=ctk.CTkFont(family="Consolas", size=13))
        info_box.pack(fill="both", expand=True, padx=15, pady=15)

        info_text = (
            "D1-OMNI SYSTEM-1 BROWSER AGENT ARCHITECTURE:\n"
            "==========================================================\n\n"
            "• Model: Liquid AI d1-omni-600M (587M Parameters)\n"
            "• Engine: Native PyTorch (CUDA 12.8) in-process execution\n"
            "• Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)\n"
            "• VRAM Footprint: ~1.2 GB (over 2.8 GB VRAM remains free)\n"
            "• Warm GPU Decision Latency: ~23.4 milliseconds per step!\n"
            "• Output Tokens: 0 (Instant, single forward pass, zero token generation delay)\n\n"
            "SYSTEM-1 DYNAMIC DECISION LOOP:\n"
            "----------------------------------------------------------\n"
            "1. BrowserOS neo is contacted via Chrome DevTools Protocol (CDP port 9110).\n"
            "2. In-page JavaScript maps live buttons/inputs into a numbered candidate dict.\n"
            "3. Query extractor parses user intent (e.g. 'open techchip youtube channel' -> 'techchip').\n"
            "4. D1 evaluates candidate elements using the 'Choice' primitive in ~23ms.\n"
            "5. Typing actions target input elements directly, followed by Enter.\n"
            "6. Dynamic stopping: When the target page/channel is reached or D1 indicates completion,\n"
            "   the agent halts immediately. No rigid, fixed-step loops!\n"
        )
        info_box.insert("1.0", info_text)
        info_box.configure(state="disabled")

    def _async_load_d1(self):
        def load():
            try:
                self.engine = D1DecisionEngine(device="cuda")
                self.engine.load()
                self.after(0, lambda: self.lbl_status.configure(text="● D1 Online (CUDA - 23ms)", text_color="#28a745"))
            except Exception as e:
                self.after(0, lambda: self.lbl_status.configure(text=f"● D1 Load Error: {e}", text_color="#dc3545"))

        threading.Thread(target=load, daemon=True).start()

    def log(self, text: str):
        self.txt_log.insert("end", text + "\n")
        self.txt_log.see("end")

    def start_automation(self):
        if not self.engine or not self.engine._is_loaded:
            messagebox.showwarning("Model Loading", "D1 model is still loading into GPU. Please wait a moment.")
            return

        self.is_running = True
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        
        url = self.entry_url.get().strip()
        goal = self.entry_goal.get().strip()
        manual_query = self.entry_query.get().strip()
        target_query = manual_query or extract_search_text(goal)

        threading.Thread(target=self._run_agent_thread, args=(url, goal, target_query), daemon=True).start()

    def stop_automation(self):
        self.is_running = False
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.log("\n[User requested Stop] Halting automation loop.")

    def _run_agent_thread(self, start_url: str, goal: str, target_query: str):
        wants_video = bool(re.search(r'\b(?:video|vid|clip|watch|play|stream|first|latest)\b', goal, flags=re.IGNORECASE))

        self.after(0, lambda: self.log(
            f"\n=======================================================\n"
            f"Starting D1 Agent: '{goal}'\n"
            f"Target Query: '{target_query}'\n"
            f"Sub-goal: {'Open and play video' if wants_video else 'Navigate to target page'}\n"
            f"Start URL: {start_url}\n"
            f"Mode: Dynamic Stopping (Tab Preserved in BrowserOS neo)\n"
            f"======================================================="
        ))
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.connect_over_cdp("http://127.0.0.1:9110")
                ctx = browser.contexts[0]
                page = ctx.new_page()

                self.after(0, lambda: self.log(f"Navigating to {start_url}..."))
                page.goto(start_url, wait_until="domcontentloaded")
                time.sleep(2.5)

                history = []
                step = 1
                max_safety_steps = 25

                while self.is_running and step <= max_safety_steps:
                    title = page.title()
                    url = page.url
                    self.after(0, lambda s=step, u=url, t=title: self.log(f"\n--- STEP {s} ---\nURL: {u}\nTitle: {t}"))

                    is_video_page = "/watch" in url
                    is_results_page = "/results" in url or "search?" in url
                    is_channel_page = ("/@" in url or "/channel/" in url or "/c/" in url)

                    # Dynamic Target Reached Check
                    if step > 1 and not is_results_page:
                        clean_target = re.sub(r'[^a-zA-Z0-9]', '', target_query.lower())
                        clean_url = re.sub(r'[^a-zA-Z0-9]', '', url.lower())
                        clean_title = re.sub(r'[^a-zA-Z0-9]', '', title.lower())

                        if wants_video:
                            if is_video_page:
                                self.after(0, lambda s=step, t=title: self.log(
                                    f"\n🎉 GOAL ACCOMPLISHED! Video '{t}' is playing in {s - 1} actions!\n"
                                    f"Tab left active in BrowserOS neo."
                                ))
                                try:
                                    page.bring_to_front()
                                except Exception:
                                    pass
                                break
                            elif is_channel_page:
                                self.after(0, lambda t=title: self.log(
                                    f"Channel '{t}' reached! Proceeding to open first/latest video..."
                                ))
                        else:
                            if clean_target and (clean_target in clean_url or clean_target in clean_title or is_channel_page):
                                self.after(0, lambda s=step, t=title: self.log(
                                    f"\n🎉 GOAL ACCOMPLISHED! Reached target '{t}' in {s - 1} actions!\n"
                                    f"Tab left active in BrowserOS neo."
                                ))
                                try:
                                    page.bring_to_front()
                                except Exception:
                                    pass
                                break

                    # 1. Extract elements
                    raw_items = page.evaluate(EXTRACTOR_JS)

                    # Filter candidates if on channel page and goal wants a video
                    if is_channel_page and wants_video:
                        video_items = [i for i in raw_items if "video_link" in i.get("description", "") or "play" in i.get("description", "").lower()]
                        if video_items:
                            raw_items = video_items

                    elements_map = {item["ref"]: item for item in raw_items}
                    criteria = {item["ref"]: item["description"] for item in raw_items}

                    # Add stop option only when at valid completion states (never on home or search results)
                    if is_video_page or (is_channel_page and not wants_video):
                        criteria["opt_done"] = f"action: Target is already open on screen, stop"

                    self.after(0, lambda c=len(criteria): self.log(f"Extracted {c} candidate elements from page."))

                    # 2. Score with D1 on GPU
                    t0 = time.time()
                    decision = self.engine.select_element_action(
                        goal=goal,
                        current_url=url,
                        page_title=title,
                        elements=criteria,
                        target_query=target_query,
                        history=history
                    )
                    lat = (time.time() - t0) * 1000.0

                    chosen_ref = decision["chosen_element"]
                    conf = decision["element_confidence"]
                    act_type = decision["action_type"]
                    chosen_desc = criteria.get(chosen_ref, "unknown")

                    msg = f"D1 Decision: {chosen_ref} ({chosen_desc}) | Action: {act_type} | Conf: {conf:.1%} | Latency: {lat:.1f}ms"
                    self.after(0, lambda m=msg: self.log(m))

                    # Model-driven Stop condition (only if allowed and not on search results/home)
                    if (decision["is_done"] or chosen_ref == "opt_done" or act_type == "stop") and not is_results_page and (is_video_page or not wants_video):
                        self.after(0, lambda s=step: self.log(
                            f"\n🎉 GOAL ACCOMPLISHED! D1 selected stop at step {s}!\n"
                            f"Target '{target_query}' reached. Tab left open in BrowserOS neo."
                        ))
                        try:
                            page.bring_to_front()
                        except Exception:
                            pass
                        break

                    elem_info = elements_map.get(chosen_ref, {})
                    is_input = elem_info.get("is_input", False) or "search_input" in chosen_desc.lower()

                    # 3. Perform action
                    if is_input or (step == 1 and not is_results_page and not is_channel_page and not is_video_page):
                        # Search action on step 1 or input selected
                        input_ref = chosen_ref if is_input else "e1"
                        self.after(0, lambda q=target_query, r=input_ref: self.log(f"Action: Typing target query '{q}' into {r} and pressing Enter..."))
                        try:
                            locator = page.locator(f'[data-d1-ref="{input_ref}"]')
                            if locator.count() == 0:
                                locator = page.locator('input[type="text"], input[type="search"], input:not([type]), [role="searchbox"]')
                            locator.click()
                            locator.fill(target_query)
                            locator.press("Enter")
                            history.append(f"Typed '{target_query}' into search box and submitted")
                        except Exception as ex:
                            self.after(0, lambda e=ex: self.log(f"[Type Warning] {e}"))
                        time.sleep(3.5)
                    else:
                        locator = page.locator(f'[data-d1-ref="{chosen_ref}"]')
                        if locator.count() == 0:
                            self.after(0, lambda: self.log(f"Element {chosen_ref} not found on page, refreshing candidates."))
                            step += 1
                            time.sleep(1.0)
                            continue
                        self.after(0, lambda r=chosen_ref, d=chosen_desc: self.log(f"Action: Clicking {r} ({d})..."))
                        try:
                            locator.click()
                            history.append(f"Clicked {chosen_desc}")
                        except Exception as ex:
                            self.after(0, lambda e=ex: self.log(f"[Click Warning] {e}"))
                        time.sleep(3.5)

                    step += 1

                self.after(0, lambda: self.log("\n✔ Agent execution loop completed! Tab left open in BrowserOS neo."))
                try:
                    page.bring_to_front()
                except Exception:
                    pass

        except Exception as e:
            self.after(0, lambda err=e: self.log(f"\n[Error] {err}"))
        finally:
            self.after(0, self._on_agent_done)

    def _on_agent_done(self):
        self.is_running = False
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")

    def inspect_active_tab(self):
        def run():
            try:
                with sync_playwright() as p:
                    browser = p.chromium.connect_over_cdp("http://127.0.0.1:9110")
                    ctx = browser.contexts[0]
                    pages = ctx.pages
                    if not pages:
                        self.after(0, lambda: messagebox.showinfo("No Pages", "No active pages found in BrowserOS."))
                        return

                    page = pages[-1]
                    raw_items = page.evaluate(EXTRACTOR_JS)
                    
                    lines = [f"=== ACTIVE TAB: {page.title()} ({page.url}) ===", f"Found {len(raw_items)} candidates:\n"]
                    for it in raw_items:
                        lines.append(f"[{it['ref']}] {it['description']}")

                    text = "\n".join(lines)
                    self.after(0, lambda: self._show_inspect_text(text))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", f"Failed to inspect: {e}"))

        threading.Thread(target=run, daemon=True).start()

    def _show_inspect_text(self, text: str):
        self.txt_inspector.delete("1.0", "end")
        self.txt_inspector.insert("1.0", text)

    def score_elements_inspector(self):
        text = self.txt_inspector.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Empty", "Please snapshot an active tab first.")
            return

        question = self.entry_insp_q.get().strip() or "Which element should be clicked?"
        
        def run():
            try:
                elements = {}
                for line in text.splitlines():
                    if line.startswith("[") and "]" in line:
                        parts = line.split("]", 1)
                        ref = parts[0].strip("[] ")
                        desc = parts[1].strip()
                        elements[ref] = desc

                if not elements:
                    return

                t0 = time.time()
                decision = self.engine.select_element_action(
                    goal=question,
                    current_url="browseros://inspector",
                    page_title="Inspector View",
                    elements=elements
                )
                lat = (time.time() - t0) * 1000.0

                result_text = f"\n\n=== D1 SCORING RESULT ({lat:.1f}ms) ===\n"
                result_text += f"Winning Element: {decision['chosen_element']} ({elements.get(decision['chosen_element'], 'unknown')})\n"
                result_text += f"Confidence: {decision['element_confidence']:.2%}\n"
                result_text += f"Action Type: {decision['action_type']}\n\nTop Probabilities:\n"
                
                probs = decision.get("element_probabilities", {})
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:10]
                for k, v in sorted_probs:
                    result_text += f"  • {k} ({elements.get(k, '')[:35]}): {v:.2%}\n"

                self.after(0, lambda: self.txt_inspector.insert("end", result_text))
                self.after(0, lambda: self.txt_inspector.see("end"))

            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Scoring Error", f"{e}"))

        threading.Thread(target=run, daemon=True).start()

if __name__ == "__main__":
    app = D1BrowserApp()
    app.mainloop()
