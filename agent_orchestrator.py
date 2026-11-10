"""agent_orchestrator.py: Executes the high-speed System-1 decision loop for browser automation.
"""

from __future__ import annotations
import time
from typing import Any, Callable, Dict, List, Optional
from .browser_adapter import BrowserSnapshotAdapter
from .d1_engine import D1DecisionEngine

class D1BrowserOrchestrator:
    def __init__(
        self,
        d1_engine: Optional[D1DecisionEngine] = None,
        snapshot_fn: Optional[Callable[[], str]] = None,
        act_fn: Optional[Callable[[str, str, Optional[str]], bool]] = None
    ):
        self.d1 = d1_engine or D1DecisionEngine()
        self.snapshot_fn = snapshot_fn
        self.act_fn = act_fn
        self.action_history: List[str] = []

    def run_step(self, goal: str, current_url: str, snapshot_text: str) -> Dict[str, Any]:
        """
        Executes one fast System-1 decision step:
        1. Parses snapshot into candidates
        2. Calls d1-omni Choice primitive
        3. Returns the decision payload
        """
        t0 = time.time()
        
        # 1. Parse into compact candidate options
        candidates = BrowserSnapshotAdapter.parse_snapshot(snapshot_text, max_candidates=50)
        
        # 2. Query d1-omni decision model
        decision = self.d1.select_element_action(
            goal=goal,
            current_url=current_url,
            elements=candidates,
            history=self.action_history
        )
        
        latency_ms = (time.time() - t0) * 1000.0
        decision["latency_ms"] = latency_ms
        decision["candidate_count"] = len(candidates)
        
        # 3. Record in history
        chosen = decision.get("chosen_element")
        action = decision.get("action_type")
        summary = f"{action} -> {chosen} ({candidates.get(chosen, 'unknown')})"
        self.action_history.append(summary)
        
        return decision
