"""d1_engine.py: System-1 Decision Engine powered by Liquid AI's d1-omni-600M.
Executes instant zero-token forward pass decisions on NVIDIA CUDA.
"""

from __future__ import annotations
import os
import re
import time
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoModel

DEFAULT_MODEL_PATH = r"C:\Users\sayan\models\d1-omni-600M"

def extract_search_text(goal: str) -> str:
    """Extracts target keyword/search string from user goal, handling compound and multi-action goals."""
    text = goal.strip()
    quotes = re.findall(r'["\']([^"\']+)["\']', text)
    if quotes:
        return quotes[0].strip()
        
    # Strip leading command verbs
    text = re.sub(r'^(?:please\s+)?(?:search(?:\s+for)?|find|open|look\s+up|go\s+to|navigate\s+to|play|show(?:\s+me)?)\s+', '', text, flags=re.IGNORECASE)
    
    # Strip compound secondary actions ("and open its first video", "then play latest video", "click first video", etc.)
    text = re.sub(r'\s+(?:and\s+)?(?:then\s+)?(?:open|play|click(?:\s+on)?|watch|view|run|start)\s+(?:its\s+|the\s+)?(?:first|latest|recent|newest|top|1st)?\s*(?:video|vid|clip|upload|short)?.*$', '', text, flags=re.IGNORECASE)
    
    # Strip platform words ("on youtube", "in google")
    text = re.sub(r'\s+(?:on|in|at)\s+(?:youtube|yt|google|wikipedia|amazon|reddit|twitter|github|bing)(?:\.com)?.*$', '', text, flags=re.IGNORECASE)
    
    # Strip channel / page / website qualifiers
    text = re.sub(r'\s+(?:youtube|yt)?\s*(?:official\s+)?(?:channel|page|website|account|video|stream)\s*$', '', text, flags=re.IGNORECASE)
    
    # Also handle "youtube channel <name>" prefix
    text = re.sub(r'^(?:youtube|yt)\s+(?:channel\s+)?', '', text, flags=re.IGNORECASE)
    
    return text.strip() or goal.strip()

class D1DecisionEngine:
    """Wrapper around d1-omni-600M for browser automation decisions."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str = "cuda"):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = None
        self._is_loaded = False

    def load(self):
        """Loads d1-omni-600M into GPU VRAM."""
        if self._is_loaded:
            return

        print(f"Loading d1-omni-600M on {self.device.upper()} (dtype={self.dtype})...", flush=True)
        self.model = AutoModel.from_pretrained(
            self.model_path,
            device_map=self.device,
            dtype=self.dtype,
            trust_remote_code=True
        )
        self.model.eval()
        self._is_loaded = True
        print(f"d1-omni-600M loaded successfully on {self.device}!", flush=True)

    def select_element_action(
        self,
        goal: str,
        current_url: str,
        page_title: str,
        elements: Dict[str, str],
        target_query: Optional[str] = None,
        history: Optional[List[str]] = None,
        image: Any = None
    ) -> Dict[str, Any]:
        """
        Evaluates current page state, goal status, and candidate elements in a single forward pass.
        """
        if not self._is_loaded:
            self.load()
            
        if not elements:
            return {"error": "No interactive elements provided", "is_done": False}

        target = target_query or extract_search_text(goal)
        is_results_page = "/results" in current_url.lower() or "search?" in current_url.lower()
        is_video_page = "/watch" in current_url.lower()
        is_channel_page = ("/@" in current_url or "/channel/" in current_url or "/c/" in current_url)
        wants_video = bool(re.search(r'\b(?:video|vid|clip|watch|play|first|latest)\b', goal, flags=re.IGNORECASE))

        # Format state
        history_text = " -> ".join(history[-4:]) if history else "Start of task"
        state = {
            "user_goal": goal,
            "target_query": target,
            "current_page_title": page_title,
            "current_url": current_url,
            "action_history": history_text
        }

        # Multi-question System-1 decision schema
        if is_video_page:
            instructions_elem = (
                f"We are watching a video on {current_url} ({page_title}). The goal is '{goal}'. "
                f"Which element should be interacted with, or is the goal complete?"
            )
        elif is_channel_page and wants_video:
            instructions_elem = (
                f"We are on the '{target}' channel page ({page_title}). "
                f"Which video link should be clicked to watch or open the first/latest video?"
            )
        elif is_results_page:
            instructions_elem = (
                f"We are browsing search results for '{target}' on {current_url} ({page_title}). "
                f"Which matching result, channel, or video should be clicked to reach '{target}'?"
            )
        else:
            instructions_elem = (
                f"Goal: '{goal}'. Target: '{target}'. Current page: {current_url} ({page_title}). "
                f"Which element should be used to search for, open, or reach '{target}'?"
            )

        act_criteria = {
            "type_and_search": "Type search keywords into input box and press Enter",
            "click": "Click this link, video, channel, or button",
            "scroll_down": "Scroll down to reveal more results"
        }
        if "opt_done" in elements or is_video_page or (is_channel_page and not wants_video):
            act_criteria["stop"] = "Stop immediately, target is reached"

        questions = {
            # Decision 1: Target element selection
            "target_element": {
                "type": "choice",
                "instructions": instructions_elem,
                "criteria": elements
            },
            # Decision 2: Operation type
            "action_type": {
                "type": "choice",
                "instructions": "What operation should be executed?",
                "criteria": act_criteria
            }
        }

        # Single forward pass (zero generated tokens)
        with torch.no_grad():
            output = self.model.system_one(
                state=state,
                questions=questions,
                images=[image] if image is not None else None
            )

        answers = output.get("answers", {})
        elem_ans = answers.get("target_element", {})
        act_ans = answers.get("action_type", {})

        chosen_elem = elem_ans.get("choice")
        chosen_act = act_ans.get("choice", "click")

        is_done = (
            chosen_elem == "opt_done" or
            chosen_act == "stop"
        )

        return {
            "is_done": is_done,
            "chosen_element": chosen_elem,
            "element_confidence": elem_ans.get("confidence", 0.0),
            "element_probabilities": elem_ans.get("probabilities", {}),
            "action_type": chosen_act,
            "action_confidence": act_ans.get("confidence", 0.0),
            "input_tokens": output.get("usage", {}).get("input_tokens", 0)
        }
