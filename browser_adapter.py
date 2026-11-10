"""browser_adapter.py: Parses and filters BrowserOS accessibility snapshots into clean Choice candidates.
"""

from __future__ import annotations
import re
from typing import Dict, List, Tuple

class BrowserSnapshotAdapter:
    """Parses BrowserOS accessibility tree snapshot text into structured candidate menus."""
    
    # Matches elements like: `- button "Search" [ref=e2]` or `- textbox "Message" [ref=e23]`
    ELEMENT_PATTERN = re.compile(
        r'-\s+([a-zA-Z0-9_\-]+)\s*(?:"([^"]*)")?\s*\[ref=([a-zA-Z0-9_\-]+)\](?:\s*\[cursor=([a-zA-Z0-9_\-]+)\])?(?:\s*\[url="([^"]*)"\])?'
    )
    
    # Actionable roles
    ACTIONABLE_ROLES = {
        "button", "link", "textbox", "searchbox", "input", "checkbox", 
        "radio", "combobox", "menuitem", "tab", "switch"
    }

    @classmethod
    def parse_snapshot(cls, snapshot_text: str, max_candidates: int = 50) -> Dict[str, str]:
        """
        Parses raw snapshot string into a clean dictionary:
        { "e1": "button: 'Search'", "e2": "textbox: 'Enter query'" }
        """
        candidates: Dict[str, str] = {}
        lines = snapshot_text.splitlines()

        for line in lines:
            line_str = line.strip()
            if not line_str.startswith("-"):
                continue

            match = cls.ELEMENT_PATTERN.search(line_str)
            if not match:
                continue

            role = match.group(1).lower()
            label = match.group(2) or ""
            ref = match.group(3)
            cursor = match.group(4) or ""
            url = match.group(5) or ""

            # Check if actionable
            is_actionable = (role in cls.ACTIONABLE_ROLES) or (cursor == "pointer")

            if not is_actionable:
                continue

            # Build human-readable description for D1 decision scoring
            desc_parts = []
            if label:
                clean_label = label.strip()
                if len(clean_label) > 60:
                    clean_label = clean_label[:57] + "..."
                desc_parts.append(f'"{clean_label}"')
            else:
                desc_parts.append("unlabeled")

            if url and role == "link":
                desc_parts.append(f"(links to {url[:40]})")

            desc = f"{role} { ' '.join(desc_parts) }".strip()
            candidates[ref] = desc

            if len(candidates) >= max_candidates:
                break

        # Always add standard navigation options as fallback choices
        candidates["opt_scroll"] = "action: Scroll down the page to reveal more content"
        candidates["opt_done"] = "action: Goal completed successfully, finish task"

        return candidates
