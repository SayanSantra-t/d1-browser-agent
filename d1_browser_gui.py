"""d1_browser_gui.py: Production System-1 Decision-Model Browser Automation Controller.
Features:
- Model-agnostic DecisionModelEngine supporting d1-2B (4-bit NF4), d1-omni-600M (FP16), or custom checkpoints
- Deterministic action binding + zero-regex Span-Read query extraction
- Scroll circuit breaker, loop prevention, and pre-click 'noul' safety confirmation
- Direct Chrome DevTools Protocol (CDP 9110) bridge to BrowserOS neo
"""

from __future__ import annotations
import os
import sys
import time
import threading
from tkinter import messagebox
import customtkinter as ctk

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from d1_engine import DecisionModelEngine, MODEL_PRESETS
from agent_orchestrator import DecisionAgentOrchestrator
from dom_extractor import EXTRACTOR_JS, build_criteria_menu

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class D1BrowserApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("System-1 Decision-Model Browser Agent (v1.0 Production Harness)")
        self.geometry("1120x800")
        self.minsize(960, 680)

        self.engine: DecisionModelEngine | None = None
        self.orchestrator: DecisionAgentOrchestrator | None = None
        self.is_running = False
        self.selected_model = "d1-2B"

        self._build_ui()
        self._async_load_model(self.selected_model)

    def _build_ui(self):
        header = ctk.CTkFrame(self, height=62, corner_radius=0)
        header.pack(fill="x", side="top")

        ctk.CTkLabel(
            header,
            text="⚡ System-1 Decision-Model Browser Agent",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(side="left", padx=20, pady=12)

        model_box = ctk.CTkFrame(header, fg_color="transparent")
        model_box.pack(side="right", padx=15)

        ctk.CTkLabel(model_box, text="Decision Model:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(0, 6))
        self.model_menu = ctk.CTkOptionMenu(
            model_box,
            values=list(MODEL_PRESETS.keys()),
            width=155,
            command=self._on_model_selected,
        )
        self.model_menu.set(self.selected_model)
        self.model_menu.pack(side="left", padx=(0, 12))

        self.lbl_status = ctk.CTkLabel(
            model_box,
            text="● Initializing GPU...",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#ffc107",
        )
        self.lbl_status.pack(side="left")

        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=15, pady=10)

        self.tab_auto = self.tabs.add("Autonomous Browser Agent")
        self.tab_inspector = self.tabs.add("Live Element Scorer")
        self.tab_info = self.tabs.add("Harness & Hardware Specs")

        self._build_auto_tab()
        self._build_inspector_tab()
        self._build_info_tab()

    def _build_auto_tab(self):
        frame = self.tab_auto
        card = ctk.CTkFrame(frame)
        card.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(card, text="Target URL:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=6, sticky="w")
        self.entry_url = ctk.CTkEntry(card, width=520, placeholder_text="https://www.youtube.com")
        self.entry_url.insert(0, "https://www.youtube.com")
        self.entry_url.grid(row=0, column=1, padx=10, pady=6, sticky="w")

        ctk.CTkLabel(card, text="User Goal:", font=ctk.CTkFont(weight="bold")).grid(row=1, column=0, padx=10, pady=6, sticky="w")
        self.entry_goal = ctk.CTkEntry(card, width=520, placeholder_text="open greencode and play its latest video")
        self.entry_goal.insert(0, "open greencode and play its latest video")
        self.entry_goal.grid(row=1, column=1, padx=10, pady=6, sticky="w")

        ctk.CTkLabel(card, text="Query Override:", font=ctk.CTkFont(weight="bold")).grid(row=2, column=0, padx=10, pady=6, sticky="w")
        self.entry_query = ctk.CTkEntry(card, width=520, placeholder_text="(Optional — leave empty for Span-Read / NLP extraction)")
        self.entry_query.grid(row=2, column=1, padx=10, pady=6, sticky="w")

        chips_frame = ctk.CTkFrame(card, fg_color="transparent")
        chips_frame.grid(row=3, column=1, sticky="w", padx=10, pady=(0, 8))
        ctk.CTkLabel(chips_frame, text="Presets:").pack(side="left", padx=(0, 5))
        ctk.CTkButton(
            chips_frame, text="GreenCode Latest Video", width=155, height=24,
            command=lambda: self._set_preset("https://www.youtube.com", "open greencode and play its latest video", "")
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            chips_frame, text="TechChip Channel", width=130, height=24,
            command=lambda: self._set_preset("https://www.youtube.com", "open techchip youtube channel", "")
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            chips_frame, text="Wikipedia FlashAttn", width=140, height=24,
            command=lambda: self._set_preset("https://www.wikipedia.org", "find out how flash attention works on wikipedia", "")
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            chips_frame, text="GitHub llama.cpp", width=125, height=24,
            command=lambda: self._set_preset("https://github.com/trending", "open the llama.cpp repository", "")
        ).pack(side="left", padx=4)

        btn_box = ctk.CTkFrame(card, fg_color="transparent")
        btn_box.grid(row=0, column=2, rowspan=3, padx=15, pady=8)

        self.btn_start = ctk.CTkButton(
            btn_box,
            text="▶ Start Agent",
            font=ctk.CTkFont(weight="bold"),
            fg_color="#28a745",
            hover_color="#218838",
            width=135,
            height=40,
            command=self.start_automation,
        )
        self.btn_start.pack(pady=4)

        self.btn_stop = ctk.CTkButton(
            btn_box,
            text="⏹ Stop",
            font=ctk.CTkFont(weight="bold"),
            fg_color="#dc3545",
            hover_color="#c82333",
            width=135,
            height=32,
            state="disabled",
            command=self.stop_automation,
        )
        self.btn_stop.pack(pady=4)

        ctk.CTkLabel(frame, text="Live System-1 Decision Execution Feed:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=15, pady=(5, 2))
        self.txt_log = ctk.CTkTextbox(frame, wrap="word", font=ctk.CTkFont(family="Consolas", size=12))
        self.txt_log.pack(fill="both", expand=True, padx=10, pady=5)
        self.txt_log.insert("end", "Production Decision-Model Harness ready. Connect BrowserOS neo on CDP 127.0.0.1:9110.\n\n")

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

        ctk.CTkButton(top, text="⚡ Score Elements", fg_color="#007bff", command=self.score_elements_inspector).pack(side="left", padx=10)

        self.txt_inspector = ctk.CTkTextbox(frame, wrap="none", font=ctk.CTkFont(family="Consolas", size=12))
        self.txt_inspector.pack(fill="both", expand=True, padx=10, pady=10)

    def _build_info_tab(self):
        frame = self.tab_info
        info_box = ctk.CTkTextbox(frame, wrap="word", font=ctk.CTkFont(family="Consolas", size=13))
        info_box.pack(fill="both", expand=True, padx=15, pady=15)

        info_text = (
            "PRODUCTION DECISION-MODEL BROWSER HARNESS (v1.0):\n"
            "==========================================================\n\n"
            "• Supported Models: LiquidAI d1-2B (4-bit NF4, ~1.34 GB VRAM) & d1-omni-600M (FP16, ~1.12 GB VRAM)\n"
            "• Hardware Target: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM, Ampere sm_86)\n"
            "• Output Tokens: 0 (Zero-token single forward pass via system_one)\n\n"
            "HARNESS ARCHITECTURE HIGHLIGHTS:\n"
            "----------------------------------------------------------\n"
            "1. Rich Semantic DOM Extractor (dom_extractor.py): Resolves SPA card titles (#video-title,\n"
            "   #channel-title, headings), filters ad/CTA noise, and tags regions ([header], [main], [modal]).\n"
            "2. Deterministic Action Binding (d1_engine.py): Eliminates element+action conflicts by deriving\n"
            "   click/type_and_search/scroll_down/stop directly from the chosen candidate's semantic role.\n"
            "3. Zero-Regex Span Extraction: Uses d1-2B's native type='text' Span-Read primitive with automatic\n"
            "   NLP fallback for models without span heads.\n"
            "4. Loop & Scroll Circuit Breakers (agent_orchestrator.py): Caps consecutive scrolls, blocks\n"
            "   re-clicking visited elements, runs 'noul' safety checks on risky clicks, and records JSON\n"
            "   trajectories in trajectories/ for offline replay.\n"
        )
        info_box.insert("1.0", info_text)
        info_box.configure(state="disabled")

    def _on_model_selected(self, choice: str):
        if self.is_running:
            messagebox.showwarning("Agent Running", "Stop the running task before switching models.")
            self.model_menu.set(self.selected_model)
            return
        self.selected_model = choice
        self._async_load_model(choice)

    def _async_load_model(self, model_name: str):
        self.lbl_status.configure(text=f"● Loading {model_name}...", text_color="#ffc107")
        self.model_menu.configure(state="disabled")

        def load():
            try:
                if self.engine is None:
                    self.engine = DecisionModelEngine(model_name, device="cuda")
                    self.engine.load()
                else:
                    self.engine.switch_model(model_name)
                self.orchestrator = DecisionAgentOrchestrator(engine=self.engine)
                status_txt = f"● {model_name} Online ({self.engine.peak_vram_mb:.0f} MB VRAM)"
                self.after(0, lambda: self.lbl_status.configure(text=status_txt, text_color="#28a745"))
            except Exception as e:
                self.after(0, lambda: self.lbl_status.configure(text=f"● Load Error: {e}", text_color="#dc3545"))
            finally:
                self.after(0, lambda: self.model_menu.configure(state="normal"))

        threading.Thread(target=load, daemon=True).start()

    def log(self, text: str):
        self.txt_log.insert("end", text + "\n")
        self.txt_log.see("end")

    def start_automation(self):
        if not self.engine or not self.engine._is_loaded:
            messagebox.showwarning("Model Loading", "Decision model is still loading into GPU. Please wait.")
            return

        self.is_running = True
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")

        url = self.entry_url.get().strip()
        goal = self.entry_goal.get().strip()
        manual_query = self.entry_query.get().strip() or None

        threading.Thread(target=self._run_agent_thread, args=(url, goal, manual_query), daemon=True).start()

    def stop_automation(self):
        self.is_running = False
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.log("\n[User requested Stop] Halting automation loop.")

    def _run_agent_thread(self, start_url: str, goal: str, manual_query: str | None):
        self.after(0, lambda: self.log(f"\n{'='*56}\nStarting Live Task on BrowserOS neo (CDP 9110)..."))
        try:
            with sync_playwright() as p:
                browser = p.chromium.connect_over_cdp("http://127.0.0.1:9110")
                ctx = browser.contexts[0]
                page = ctx.new_page()

                res = self.orchestrator.run_cdp_task(
                    page=page,
                    start_url=start_url,
                    goal=goal,
                    manual_query=manual_query,
                    max_steps=10,
                    log_fn=lambda m: self.after(0, lambda msg=m: self.log(msg)),
                    stop_flag=lambda: not self.is_running,
                )

                summary = (
                    f"\n>>> RESULT: success={res['success']} | steps={res['steps']} | "
                    f"avg_latency={res['avg_latency_ms']}ms | reason={res['reason']}\n"
                    f"    Final URL: {res['final_url']}"
                )
                self.after(0, lambda s=summary: self.log(s))
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
                    raw_items = page.evaluate(EXTRACTOR_JS, {"maxCandidates": 40})
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
        if not self.engine or not self.engine._is_loaded:
            messagebox.showwarning("Model Loading", "Model not loaded yet.")
            return
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

                decision = self.engine.select_element_action(
                    goal=question,
                    current_url="browseros://inspector",
                    page_title="Inspector View",
                    elements=elements,
                )

                result_text = f"\n\n=== {self.engine.model_label} SCORING RESULT ({decision['latency_ms']:.1f}ms) ===\n"
                result_text += f"Winning Element: {decision['chosen_element']} ({elements.get(decision['chosen_element'], 'unknown')})\n"
                result_text += f"Confidence: {decision['element_confidence']:.2%}\n"
                result_text += f"Derived Action: {decision['action_type']}\n\nTop Probabilities:\n"

                probs = decision.get("element_probabilities", {})
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:10]
                for k, v in sorted_probs:
                    result_text += f"  • {k} ({elements.get(k, '')[:42]}): {v:.2%}\n"

                self.after(0, lambda: self.txt_inspector.insert("end", result_text))
                self.after(0, lambda: self.txt_inspector.see("end"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Scoring Error", f"{e}"))

        threading.Thread(target=run, daemon=True).start()


if __name__ == "__main__":
    app = D1BrowserApp()
    app.mainloop()
