"""agent_orchestrator.py: Production System-1 Decision Agent Orchestrator.
Provides:
- Model-agnostic execution over CDP (BrowserOS neo) or Accessibility Snapshots
- Consecutive-scroll circuit breaker & tab-escalation (e.g., clicking 'Videos' tab on channels)
- Visited-element loop prevention
- Deterministic goal-completion verification
- Pre-click 'noul' safety guardrail
- JSON trajectory recording for offline replay and regression testing
"""

from __future__ import annotations
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from browser_adapter import BrowserSnapshotAdapter
from d1_engine import DecisionModelEngine
from dom_extractor import EXTRACTOR_JS, build_criteria_menu

TRAJECTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories")

RISKY_PATTERNS = re.compile(
    r"\b(?:delete|remove|pay|checkout|place\s+order|purchase|unsubscribe|deactivate|erase|transfer|"
    r"cancel|refund|withdraw|terminate|empty\s+trash|permanently)\b",
    flags=re.IGNORECASE,
)


class DecisionAgentOrchestrator:
    def __init__(
        self,
        engine: Optional[DecisionModelEngine] = None,
        max_consecutive_scrolls: int = 2,
        enable_safety_gate: bool = True,
        record_trajectories: bool = True,
    ):
        self.engine = engine or DecisionModelEngine()
        self.max_consecutive_scrolls = max_consecutive_scrolls
        self.enable_safety_gate = enable_safety_gate
        self.record_trajectories = record_trajectories
        self.action_history: List[str] = []
        self.clicked_history: set[Tuple[str, str]] = set()
        self.consecutive_scrolls = 0

    def reset(self):
        self.action_history.clear()
        self.clicked_history.clear()
        self.consecutive_scrolls = 0

    def check_goal_reached(
        self,
        step: int,
        goal: str,
        target_query: str,
        url: str,
        title: str,
    ) -> Tuple[bool, str]:
        """Deterministic post-navigation check so the agent halts immediately upon goal satisfaction."""
        if step <= 1:
            return False, ""

        url_l = url.lower()
        title_l = title.lower()
        is_results = (
            "/results" in url_l
            or "search?" in url_l
            or "?q=" in url_l
            or "special:search" in url_l
            or "special%3asearch" in url_l
            or "?search=" in url_l
        )
        if is_results:
            return False, ""

        is_video = "/watch" in url_l or "/shorts/" in url_l
        is_channel = "/@" in url_l or "/channel/" in url_l or "/c/" in url_l
        wants_video = bool(
            re.search(r"\b(?:video|vid|clip|watch|play|stream|first|latest|newest)\b", goal, flags=re.IGNORECASE)
        )

        clean_target = re.sub(r"[^a-z0-9]", "", target_query.lower())
        clean_url = re.sub(r"[^a-z0-9]", "", url_l)
        clean_title = re.sub(r"[^a-z0-9]", "", title_l)

        if wants_video:
            if is_video:
                # Action verbs are goal verbs, not topic tokens — counting them lets a
                # 'play' match in the click history stand in for a real topic match.
                _ACTION_VERBS = {"play", "watch", "see", "open", "click", "stream"}
                target_tokens = [
                    t
                    for t in re.findall(r"[a-z0-9]+", target_query.lower())
                    if len(t) >= 3 and t not in _ACTION_VERBS
                ]
                last_click = next((h for h in reversed(self.action_history) if h.startswith("Clicked ")), "")
                combined_context = f"{title_l} {last_click.lower()}"
                # Check if we navigated via a matching channel_link — any topic token in
                # the channel link counts ('techchip' in 'channel_link: TechChip'), not
                # just the full query string (live L4: topic+channel goals never matched).
                visited_matching_channel = any(
                    "channel_link:" in h.lower()
                    and any(t in re.sub(r"[^a-z0-9]", "", h.lower()) for t in target_tokens)
                    for h in self.action_history
                )
                # Check topic keyword overlap in video title or clicked video link
                token_hits = sum(1 for tok in target_tokens if tok in combined_context)
                required_hits = max(1, (len(target_tokens) + 1) // 2) if target_tokens else 0
                if (
                    not target_tokens
                    or token_hits >= required_hits
                    or (visited_matching_channel and token_hits >= 1)
                ):
                    return True, f"Video page opened: '{title}'"
            return False, ""

        if clean_target and (clean_target in clean_url or clean_target in clean_title):
            return True, f"Target page reached: '{title}'"

        # Token-overlap arrival: the extracted query's salient tokens appearing across
        # host/path/title is strong arrival evidence even when the URL wording differs
        # from the query (e.g. query 'official obs studio download' vs URL
        # obsproject.com/download). Requires a host-token hit so a mere keyword in the
        # page body alone cannot trigger completion.
        _GENERIC_ARRIVAL = {
            "official", "page", "site", "website", "web", "online", "free", "best",
            "find", "open", "the", "www", "http", "https", "com", "org", "net", "main",
        }
        arr_tokens = {
            t
            for t in re.findall(r"[a-z0-9]+", target_query.lower())
            if len(t) >= 3 and t not in _GENERIC_ARRIVAL
        }
        if arr_tokens:
            after_scheme = re.sub(r"^[a-z0-9]+://", "", url_l)
            clean_host = re.sub(r"[^a-z0-9]", "", after_scheme.split("/")[0])
            path_str = after_scheme.split("/", 1)[1] if "/" in after_scheme else ""
            clean_path = re.sub(r"[^a-z0-9]", "", path_str)
            host_hits = sum(1 for t in arr_tokens if t in clean_host)
            path_hits = sum(1 for t in arr_tokens if t in clean_path)
            # Multi-token queries must match BOTH the site identity (host) and the
            # destination (path) — host-only matches would stop at homepages when the
            # goal implies a deeper page (forum/docs/...). Single-token queries arrive
            # on a lone host hit ('open youtube' -> youtube.com).
            if (host_hits >= 1 and path_hits >= 1) or (len(arr_tokens) == 1 and host_hits >= 1):
                return True, f"Target page reached (host+path token match): '{title}'"

        return False, ""

    def decide_from_candidates(
        self,
        goal: str,
        target_query: str,
        url: str,
        title: str,
        raw_items: List[Dict[str, Any]],
        image: Any = None,
    ) -> Dict[str, Any]:
        """Runs one guarded System-1 decision step over extracted DOM candidates."""
        url_l = url.lower()
        is_video_page = "/watch" in url_l or "/shorts/" in url_l
        is_channel_page = "/@" in url_l or "/channel/" in url_l or "/c/" in url_l
        wants_video = bool(
            re.search(r"\b(?:video|vid|clip|watch|play|first|latest|newest)\b", goal, flags=re.IGNORECASE)
        )

        # Filter out previously clicked non-input elements on the exact same URL (keyed by role+label)
        filtered = [
            it
            for it in raw_items
            if it.get("is_input") or (url, f"{it.get('role','')}:{it.get('label','')}") not in self.clicked_history
        ]
        if not filtered and raw_items:
            # Every non-input candidate was already clicked on this exact URL and nothing
            # navigated: the visible page is exhausted. Scroll for fresh content (respecting
            # the circuit breaker) rather than re-clicking visited items; escalate if
            # scrolling is exhausted too.
            if self.consecutive_scrolls < self.max_consecutive_scrolls:
                self.consecutive_scrolls += 1
                return {
                    "is_done": False,
                    "chosen_element": "opt_scroll",
                    "element_confidence": 0.0,
                    "action_type": "scroll_down",
                    "description": "All visible candidates already visited; scrolling to reveal more",
                    "latency_ms": 0.0,
                    "escalated": "visited_fallback_scroll",
                }
            return {
                "is_done": False,
                "chosen_element": "opt_human",
                "element_confidence": 0.0,
                "action_type": "request_human",
                "description": "Loop detected: every candidate on this page was already clicked without reaching the goal",
                "latency_ms": 0.0,
                "escalated": "visited_loop_escalation",
            }
        if not filtered:
            filtered = raw_items

        # On search results pages, the URL itself (?q=, /results, Special:Search) proves a
        # search already happened — but only trust that when the results actually match
        # the topic. Empty/failed searches must go back through the input, not re-click
        # the "Search" button (live L5-2B loop).
        is_results_page = (
            "/results" in url_l
            or "search?" in url_l
            or "?q=" in url_l
            or "special:search" in url_l
            or "special%3asearch" in url_l
            or "?search=" in url_l
        )
        narrowed_to_search = False
        target_tokens = [t for t in re.findall(r"[a-z0-9.]+", target_query.lower()) if len(t) >= 3]
        # Tokens already carried by the host ('wikipedia' on wikipedia.org) match every
        # boilerplate link ("Italiano — Wikipedia — ...") and make direct-match checks
        # useless — exclude them from topic matching (live L5-600M language drift).
        after_scheme = re.sub(r"^[a-z0-9]+://", "", url_l)
        host_tokens = set(re.findall(r"[a-z0-9]+", after_scheme.split("/")[0]))
        match_tokens = [t for t in target_tokens if t not in host_tokens]

        def _matches_topic(it: Dict[str, Any]) -> bool:
            hay = f"{it.get('description','')} {it.get('label','')}".lower()
            return any(t in hay for t in match_tokens)

        has_direct_match = any(not it.get("is_input") and _matches_topic(it) for it in filtered)

        if is_results_page:
            non_inputs = [it for it in filtered if not it.get("is_input")]
            if non_inputs and has_direct_match:
                filtered = non_inputs
            elif non_inputs:
                # Results exist but none match the topic: empty or failed search —
                # re-search through the input instead of clicking Search buttons.
                inputs = [it for it in filtered if it.get("is_input")]
                if inputs:
                    filtered = inputs
                    narrowed_to_search = True
                else:
                    filtered = non_inputs
            else:
                inputs = [it for it in filtered if it.get("is_input")]
                if inputs:
                    filtered = inputs
                    narrowed_to_search = True

        # On watch pages, suppress engagement-mutating side actions (subscribe, bell,
        # join, ...). They mutate account state and are never the way to watch a video
        # (live L4: 600M clicked 'Subscribe to TechChip' at 81.7% on a watch page).
        if is_video_page:
            distracting = re.compile(
                r"\b(subscrib\w*|notifications?|bell|join|member(?:ship)?|sponsor|donate|merch)\b",
                re.IGNORECASE,
            )
            essential = [
                it
                for it in filtered
                if it.get("is_input")
                or not distracting.search(f"{it.get('label','')} {it.get('description','')}")
            ]
            if essential:
                filtered = essential

        # On channel pages when seeking a video, prefer video_links first; only show Videos tab if none exist.
        # Exception: explicit sort/filter intents (e.g. 'popular videos') need the sort buttons visible.
        sort_intent = bool(re.search(r"\b(?:popular|trending|most\s+viewed|top\s+rated)\b", goal, flags=re.IGNORECASE))
        has_channel_videos = False
        if is_channel_page and wants_video and not sort_intent:
            vid_only = [it for it in filtered if it.get("role") == "video_link"]
            if vid_only:
                # Topic-aware narrowing: prefer uploads actually about the topic
                # ('wifi security') over unrelated videos from the same channel.
                topic_toks = [
                    t
                    for t in re.findall(r"[a-z0-9]+", target_query.lower())
                    if len(t) >= 3
                    and t not in {"play", "watch", "see", "open", "click", "stream", "video", "channel"}
                ]
                on_topic = [
                    it
                    for it in vid_only
                    if any(
                        t in it.get("description", "").lower() or t in it.get("label", "").lower()
                        for t in topic_toks
                    )
                ]
                if on_topic:
                    vid_only = on_topic
                filtered = vid_only
                has_channel_videos = True
            else:
                vid_tabs = [
                    it
                    for it in filtered
                    if it.get("role") == "tab" and "video" in it.get("description", "").lower()
                ]
                if vid_tabs:
                    filtered = vid_tabs

        # On search-portal homepages or listing pages when target isn't directly visible yet,
        # prioritize search inputs or header search trigger buttons (e.g. GitHub "Search or jump to...")
        is_portal_home = bool(
            re.match(
                r"^https?://(?:www\.)?(?:youtube\.com|google\.com|wikipedia\.org|bing\.com|duckduckgo\.com)"
                r"/?(?:\?.*)?$",
                url_l,
            )
        )
        if not has_direct_match:
            search_boxes = [it for it in filtered if it.get("is_input")]
            if search_boxes and is_portal_home:
                filtered = search_boxes
                narrowed_to_search = True
            elif not search_boxes and ("github.com" in url_l):
                search_triggers = [
                    it for it in filtered
                    if it.get("region") == "header" and "search" in it.get("label", "").lower()
                ]
                if search_triggers:
                    filtered = search_triggers
                    narrowed_to_search = True

        # Circuit breaker: disable opt_scroll if we already scrolled max_consecutive_scrolls times
        # or if video links are already visible on the target channel page
        allow_scroll = (self.consecutive_scrolls < self.max_consecutive_scrolls) and not has_channel_videos
        allow_scroll = allow_scroll and not narrowed_to_search
        if not allow_scroll and is_channel_page and wants_video:
            # Escalate directly to 'Videos' tab or top video link if scrolling didn't reveal a video
            vid_links = [it for it in filtered if it.get("role") == "video_link"]
            if vid_links:
                chosen = vid_links[0]
                self.consecutive_scrolls = 0
                return {
                    "is_done": False,
                    "chosen_element": chosen["ref"],
                    "element_confidence": 1.0,
                    "action_type": "click",
                    "description": chosen["description"],
                    "latency_ms": 0.0,
                    "escalated": "scroll_circuit_breaker_top_video",
                }
            vid_tabs = [
                it for it in filtered if it.get("role") == "tab" and "video" in it.get("label", "").lower()
            ]
            if vid_tabs:
                chosen = vid_tabs[0]
                self.consecutive_scrolls = 0
                return {
                    "is_done": False,
                    "chosen_element": chosen["ref"],
                    "element_confidence": 1.0,
                    "action_type": "click",
                    "description": chosen["description"],
                    "latency_ms": 0.0,
                    "escalated": "scroll_circuit_breaker_videos_tab",
                }

        elements_meta = {it["ref"]: it for it in filtered}
        criteria = build_criteria_menu(
            filtered,
            include_scroll=allow_scroll,
            include_done=(is_video_page or (is_channel_page and not wants_video)),
            include_human=("login" in url_l or "signin" in url_l or "captcha" in title.lower()),
        )

        decision = self.engine.select_element_action(
            goal=goal,
            current_url=url,
            page_title=title,
            elements=criteria,
            elements_meta=elements_meta,
            target_query=target_query,
            history=self.action_history,
            image=image,
        )

        chosen_ref = decision["chosen_element"]
        chosen_desc = criteria.get(chosen_ref, "unknown")
        act_type = decision["action_type"]
        decision["description"] = chosen_desc

        if act_type == "scroll_down":
            self.consecutive_scrolls += 1
        else:
            self.consecutive_scrolls = 0

        # Safety guardrail check on risky clicks
        if self.enable_safety_gate and act_type == "click" and RISKY_PATTERNS.search(chosen_desc):
            safety = self.engine.check_action_safety(chosen_desc)
            decision["safety_check"] = safety
            if safety["is_risky"]:
                decision["action_type"] = "request_human"
                decision["blocked_by_safety"] = True

        return decision

    @staticmethod
    def _robust_click(loc: Any, log_fn=None, timeout_ms: int = 8000) -> bool:
        """Click with layered fallbacks: normal -> JS-dispatch -> scroll+force.

        Live L4 crash: YouTube overlays (popups, sticky headers, search combobox)
        intercept pointer events for 30s until Playwright times out and raises,
        killing the whole task. Layered fallbacks never raise.
        """
        try:
            loc.first.click(timeout=timeout_ms)
            return True
        except Exception as err:
            if log_fn:
                log_fn(f"    normal click failed ({type(err).__name__}), trying JS fallback")
        try:
            loc.first.evaluate("el => el.click()")
            return True
        except Exception as err:
            if log_fn:
                log_fn(f"    JS click failed ({type(err).__name__}), trying scroll+force")
        try:
            loc.first.scroll_into_view_if_needed(timeout=5000)
            loc.first.click(timeout=timeout_ms, force=True)
            return True
        except Exception as err:
            if log_fn:
                log_fn(f"    click failed permanently: {err}")
            return False

    def run_cdp_task(
        self,
        page: Any,
        start_url: str,
        goal: str,
        manual_query: Optional[str] = None,
        max_steps: int = 10,
        log_fn: Optional[Callable[[str], None]] = None,
        stop_flag: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Executes a complete live browser task on a Playwright CDP page."""
        self.reset()
        log = log_fn or (lambda msg: None)

        q_info = (
            {"query": manual_query, "method": "manual", "confidence": 1.0, "latency_ms": 0.0}
            if manual_query
            else self.engine.extract_query(goal)
        )
        target_query = q_info["query"]
        log(
            f"Model: {self.engine.model_label} | Goal: '{goal}' | "
            f"Query: '{target_query}' ({q_info['method']}, {q_info['latency_ms']:.1f}ms)"
        )

        page.bring_to_front()
        page.goto(start_url, wait_until="domcontentloaded")
        page.wait_for_timeout(2200)

        trajectory: List[Dict[str, Any]] = []
        latencies: List[float] = []
        success = False
        reason = "max_steps_reached"

        for step in range(1, max_steps + 1):
            if stop_flag and stop_flag():
                reason = "stopped_by_user"
                break

            page.bring_to_front()
            url = page.url
            title = page.title()

            done, done_msg = self.check_goal_reached(step, goal, target_query, url, title)
            if done:
                success = True
                reason = done_msg
                log(f"[Step {step}] DONE: {done_msg}")
                break

            raw_items = page.evaluate(EXTRACTOR_JS, {"maxCandidates": 40})
            decision = self.decide_from_candidates(goal, target_query, url, title, raw_items)
            lat_ms = decision.get("latency_ms", 0.0)
            if lat_ms > 0:
                latencies.append(lat_ms)

            chosen_ref = decision["chosen_element"]
            act_type = decision["action_type"]
            desc = decision["description"]
            conf = decision["element_confidence"]

            log(f"[Step {step}] {url} -> {chosen_ref} ({desc}) | act={act_type} | conf={conf:.1%} | {lat_ms:.1f}ms")

            trajectory.append(
                {
                    "step": step,
                    "url": url,
                    "title": title,
                    "candidate_count": len(raw_items),
                    "chosen_ref": chosen_ref,
                    "description": desc,
                    "action_type": act_type,
                    "confidence": conf,
                    "latency_ms": lat_ms,
                }
            )

            if act_type == "stop":
                success = True
                reason = "model_selected_stop"
                break
            if act_type == "request_human":
                reason = "human_help_required"
                log(f"[Step {step}] Paused for human confirmation/login: {desc}")
                break
            if act_type == "scroll_down":
                page.mouse.wheel(0, 700)
                self.action_history.append("Scrolled down")
                page.wait_for_timeout(1500)
                continue

            if act_type == "type_and_search":
                loc = page.locator(f'[data-d1-ref="{chosen_ref}"]')
                if loc.count() == 0:
                    loc = page.locator('input[name="search_query"], input[type="search"], input[type="text"]').first
                self._robust_click(loc, log)
                loc.first.fill(target_query)
                loc.first.press("Enter")
                self.action_history.append(f"Searched '{target_query}'")
                page.wait_for_timeout(2800)
                continue

            # Default: click
            meta_clicked = next((it for it in raw_items if it["ref"] == chosen_ref), None)
            click_key = f"{meta_clicked.get('role','')}:{meta_clicked.get('label','')}" if meta_clicked else desc
            self.clicked_history.add((url, click_key))
            loc = page.locator(f'[data-d1-ref="{chosen_ref}"]')
            if loc.count() > 0:
                if self._robust_click(loc, log):
                    self.action_history.append(f"Clicked {desc}")
            else:
                log(f"[Step {step}] Ref {chosen_ref} stale, refreshing DOM...")
            page.wait_for_timeout(2800)

        result = {
            "model": self.engine.model_label,
            "goal": goal,
            "target_query": target_query,
            "query_extraction": q_info,
            "success": success,
            "reason": reason,
            "steps": len(trajectory),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "final_url": page.url,
            "final_title": page.title(),
            "trajectory": trajectory,
        }

        if self.record_trajectories:
            os.makedirs(TRAJECTORY_DIR, exist_ok=True)
            ts = int(time.time() * 1000)
            fname = os.path.join(TRAJECTORY_DIR, f"traj_{self.engine.model_label}_{ts}.json")
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            result["trajectory_file"] = fname

        return result


# Backward-compatible alias
class D1BrowserOrchestrator:
    def __init__(self, d1_engine: Optional[DecisionModelEngine] = None, **kwargs):
        self.d1 = d1_engine or DecisionModelEngine()
        self.action_history: List[str] = []

    def run_step(self, goal: str, current_url: str, snapshot_text: str) -> Dict[str, Any]:
        candidates = BrowserSnapshotAdapter.parse_snapshot(snapshot_text, max_candidates=50)
        decision = self.d1.select_element_action(
            goal=goal,
            current_url=current_url,
            page_title="",
            elements=candidates,
            history=self.action_history,
        )
        chosen = decision.get("chosen_element")
        action = decision.get("action_type")
        self.action_history.append(f"{action} -> {chosen} ({candidates.get(chosen, 'unknown')})")
        return decision
