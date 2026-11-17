"""d1_engine.py: Model-Agnostic System-1 Decision Engine.
Supports LiquidAI d1-omni-600M, d1-2B, d1-3B, d1-Omni-2B, or any custom System-1 decision model.
Features:
- Automatic Windows/Ampere compatibility shims ('axial' RoPE + SDPA fallback)
- Deterministic action derivation from selected element semantics (eliminates action-type mismatch)
- Zero-regex query extraction via native type='text' Span-Read (with NLP fallback for 600M)
- Calibrated 'idk' abstention & 'noul' pre-click safety checks
- Multi-tab batched inference via system_one_batch()
"""

from __future__ import annotations
import gc
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from transformers import AutoModel, BitsAndBytesConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

# Register 'axial' RoPE initializer if running on transformers < 5.19
if "axial" not in ROPE_INIT_FUNCTIONS:
    def _axial_rope_init(config, device=None, seq_len=None, **kwargs):
        base = config.rope_parameters["rope_theta"]
        spatial_dim = config.hidden_size // config.num_attention_heads // 2
        inv_freq = 1.0 / (
            base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float, device=device) / spatial_dim)
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["axial"] = _axial_rope_init


def _patch_d1_attention():
    """Forces SDPA fallback on Windows PyTorch builds lacking compiled FlashAttention kernels."""
    for mod in list(sys.modules.values()):
        if getattr(mod, "__name__", "").endswith(".attention") and hasattr(mod, "_fused"):
            mod._fused = lambda q: False


MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "d1-omni-600M": {
        "path": r"C:\Users\sayan\models\d1-omni-600M",
        "quant_4bit": False,
        "supports_span_read": False,
        "media_kwarg": "images",
    },
    "d1-2B": {
        "path": r"C:\Users\sayan\models\d1-2B",
        "quant_4bit": True,
        "supports_span_read": True,
        "media_kwarg": "media",
    },
}

DEFAULT_MODEL_PATH = MODEL_PRESETS["d1-2B"]["path"]


def extract_search_text(goal: str) -> str:
    """Clean NLP/regex fallback query extractor for models without native span-read heads."""
    text = goal.strip()
    quotes = re.findall(r'["\']([^"\']+)["\']', text)
    if quotes:
        return quotes[0].strip()

    # Handle "open the <x> page for the <topic>" (e.g., "open the wikipedia page for the periodic table").
    # Keep a real site qualifier ('pypi', 'wikipedia') — dropping it loses the site identity and
    # sends the search to lookalike pages (live L2 failure: 'requests library' -> robotframework docs).
    page_for = re.match(
        r'^(?:can\s+you\s+)?(?:please\s+)?(?:open|go\s+to|visit|find|check)\s+(?:the\s+)?([a-z0-9][a-z0-9 .\-]*?)?\bpage\s+for\s+(?:the\s+)?(.+)$',
        text,
        flags=re.IGNORECASE,
    )
    if page_for:
        site = (page_for.group(1) or "").strip()
        topic = page_for.group(2).strip()
        fillers = {"the", "a", "an", "official", "main", "home", "web"}
        if site and site.lower() not in fillers:
            return f"{site} {topic}".strip()
        return topic

    # Handle "<site> article about <topic>" (e.g., "open the wikipedia article about the
    # adiabatic process"). Keep the site qualifier; live L5 showed searching the raw goal
    # surfaces '(page does not exist)' placeholder links.
    article = re.match(
        r'^(?:can\s+you\s+)?(?:please\s+)?(?:open|go\s+to|visit|find|read|check)\s+(?:the\s+)?'
        r'(wikipedia|wiki)\s+article\s+(?:about|on|for|of)\s+(?:the\s+)?(.+)$',
        text,
        flags=re.IGNORECASE,
    )
    if article:
        topic = article.group(2).strip()
        topic = re.sub(
            r'\s+(?:and\s+(?:then\s+)?|then\s+)(?:open|play|click(?:\s+on)?|watch|view|run|start)'
            r'(?:\s+(?:it|them|that))?\s*$',
            '', topic, flags=re.IGNORECASE,
        )
        return f"{article.group(1)} {topic}".strip()

    # Handle "search <platform> for <topic>" (e.g., "search wikipedia for slow scan television sstv")
    plat_for = re.match(
        r'^(?:can\s+you\s+)?(?:please\s+)?(?:search|look\s+up|check)\s+(?:youtube|yt|google|wikipedia|amazon|reddit|github|bing)\s+for\s+(?:the\s+)?(.+)$',
        text,
        flags=re.IGNORECASE,
    )
    if plat_for:
        return plat_for.group(1).strip()

    # Handle "find the official website of <entity>" (e.g., "find the official website of obs studio")
    site_of = re.match(
        r'^(?:can\s+you\s+)?(?:please\s+)?(?:find|open|go\s+to|visit|check|show\s+me)?\s*(?:the\s+)?official\s+(?:website|site|homepage|home\s+page|docs?|documentation)\s+of\s+(.+)$',
        text,
        flags=re.IGNORECASE,
    )
    if site_of:
        return site_of.group(1).strip()

    # Handle "open <platform> and (look for|search for|find) <topic>" (e.g., "open amazon and look for arduino uno r4 wifi board")
    plat_and = re.match(
        r'^(?:can\s+you\s+)?(?:please\s+)?(?:open|go\s+to|visit|check)\s+(?:youtube|yt|google|amazon|reddit|github|wikipedia|bing)\s+'
        r'and\s+(?:look\s+(?:for|up)|search\s+for|find|check|get)\s+(?:the\s+)?(.+?)\s*(?:on\s+(?:youtube|yt|google|wikipedia|amazon|reddit|github|bing))?$',
        text,
        flags=re.IGNORECASE,
    )
    if plat_and:
        return plat_and.group(1).strip()

    # Handle conversational clauses like "find that video where techchip talks about kali linux"
    talks_match = re.search(
        r'\bwhere\s+([a-zA-Z0-9_.\-]+)\s+(?:talks?|speaks?|explains?|shows?)\s+(?:about\s+)?(.+)$',
        text,
        flags=re.IGNORECASE,
    )
    if talks_match:
        entity = talks_match.group(1).strip()
        topic = talks_match.group(2).strip()
        topic = re.sub(r'\s+(?:on|in|at)\s+(?:youtube|yt|google|wikipedia|reddit|github).*$', '', topic, flags=re.IGNORECASE)
        # Strip trailing secondary actions ("... and play it", "... then watch it") —
        # live L4: query became 'techchip wifi security and play it' and polluted the search.
        topic = re.sub(
            r'\s+(?:and\s+(?:then\s+)?|then\s+)(?:open|play|click(?:\s+on)?|watch|view|run|start)'
            r'(?:\s+(?:it|them|that))?\s*$',
            '', topic, flags=re.IGNORECASE,
        )
        return f"{entity} {topic}".strip()

    # Strip leading polite/command verbs + optional articles ("the", "a", "an")
    text = re.sub(
        r'^(?:can\s+you\s+)?(?:please\s+)?(?:go\s+to\s+\w+\s+and\s+)?'
        r'(?:search(?:\s+for)?|find(?:\s+out\s+how)?|open|look\s+up|go\s+to|navigate\s+to|play|watch|check|show(?:\s+me)?)\s+(?:the\s+|a\s+|an\s+)?',
        '',
        text,
        flags=re.IGNORECASE,
    )

    # Strip leading conversational wrappers ("video about X", "clip where X")
    text = re.sub(r'^(?:video|vid|clip|item|result)s?\s+(?:about|where)\s+', '', text, flags=re.IGNORECASE)

    # Strip "the newest/latest <Channel> video" wrapper -> "<Channel>"
    newest_vid = re.match(
        r'^(?:the\s+)?(?:newest|latest|recent|first|top)\s+([a-zA-Z0-9_.\-]+)\s+(?:video|vid|clip|upload)(?:\s+on\s+.*)?$',
        text,
        flags=re.IGNORECASE,
    )
    if newest_vid:
        return newest_vid.group(1).strip()

    # Strip compound secondary actions ("and play its latest video", etc.)
    text = re.sub(
        r'\s+(?:and\s+)?(?:then\s+)?(?:open|play|click(?:\s+on)?|watch|view|run|start)\s+'
        r'(?:its\s+|the\s+)?(?:first|latest|recent|newest|top|1st)?\s*(?:video|vid|clip|upload|short)?.*$',
        '',
        text,
        flags=re.IGNORECASE,
    )

    # Strip trailing platform qualifiers ("on youtube", "on reddit", "on google", "on github")
    text = re.sub(
        r'\s+(?:on|in|at)\s+(?:youtube|yt|google|wikipedia|amazon|reddit|twitter|x|github|bing)(?:\.com)?.*$',
        '',
        text,
        flags=re.IGNORECASE,
    )

    # Strip trailing "<entity> channel" references ("... on techchip channel")
    text = re.sub(r'\s+on\s+(?:the\s+)?[a-z0-9.\-]+\s*channel\s*$', '', text, flags=re.IGNORECASE)

    # Strip trailing "works" in "how X works"
    text = re.sub(r'\s+works$', '', text, flags=re.IGNORECASE)

    # Strip channel / page / website / repository qualifiers
    text = re.sub(
        r'\s+(?:youtube|yt)?\s*(?:official\s+)?(?:channel|page|website|account|video|stream|repository|repo)\s*$',
        '',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r'^(?:youtube|yt)\s+(?:channel\s+)?', '', text, flags=re.IGNORECASE)

    # Strip a leading dangling article left by span/nlp handling ("the adiabatic process")
    text = re.sub(r'^the\s+', '', text, flags=re.IGNORECASE)

    return text.strip() or goal.strip()


# ---------------------------------------------------------------------------
# Deterministic decision helpers (Iterations 12-14): locks, policies, parsing
# ---------------------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "on", "in", "at", "to", "for", "with", "about", "from",
    "my", "me", "we", "our", "is", "are", "was", "were", "be", "its", "it", "this", "that",
    "these", "those", "there", "where", "how", "what", "when", "who", "whom", "which", "please",
    "can", "could", "would", "you", "your", "not", "skip", "exclude", "excluding", "except",
    "but", "then", "also", "page", "site", "website", "official", "new", "open", "find", "search",
    "look", "check", "show", "play", "watch", "click", "go", "navigate", "read", "browse", "get",
    "let", "want", "need", "should", "must", "step",
}

_WORD_NUM = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
             "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
_REL_UNITS = {"minute": 1, "min": 1, "hour": 60, "hr": 60, "day": 1440, "week": 10080,
              "month": 43200, "year": 525600}

_NEGATION_RE = re.compile(
    r'(?:,\s*|\s+but\s+|\s+)(?:not|skip|excluding|except(?:\s+for)?|rather\s+than)\s+'
    r'(?:the\s+|any\s+|all\s+)?([a-z][a-z0-9 \-]{2,60}?)(?=[,.;]|$)',
    re.IGNORECASE,
)
_CRED_FIELD_RE = re.compile(r"\b(passwords?|captchas?|otps?|one[- ]time|passcodes?|2fa|two[- ]factor)\b", re.IGNORECASE)
_ACCOUNT_GOAL_RE = re.compile(
    r"\b(my|account|booked|orders?|profile|inbox|dashboard|subscriptions?|sign\s?in(?:to)?|log\s?in(?:to)?|login|credentials?)\b",
    re.IGNORECASE,
)
_MODAL_DISMISS_RE = re.compile(
    r"\b(block|reject|no\s+thanks|no,?\s*(?:thanks|exit)|decline|maybe\s+later|not\s+now|dismiss|close|exit)\b",
    re.IGNORECASE,
)
_MODAL_AFFIRM_RE = re.compile(r"\b(yes|i\s+am|i'm|continue|enter|agree|proceed|accept)\b", re.IGNORECASE)
_MODAL_INFO_LINK_RE = re.compile(
    r"\b(terms?(?:\s+of\s+(?:service|use))?|privacy|polic(?:y|ies)|legal|copyright|"
    r"learn\s+more|read\s+more|manage\s+settings)\b",
    re.IGNORECASE,
)
_GATE_CONTEXT_RE = re.compile(r"\b(1[68]\+?|age\s+(?:verification|gate|restriction)|years?\s+old|are\s+you\s+over|enter(?:ing)?\s+(?:the\s+)?site)\b", re.IGNORECASE)
_OPTOUT_CONTEXT_RE = re.compile(
    r"\b(notifications?|newsletters?|cookies?|consent|subscrib\w+|promos?|deals?|offers?|allow)\b", re.IGNORECASE
)
_VERSION_RE = re.compile(r"\b\d+(?:\.\d+)+\b")

# Download-aggregator domains: on "download/install" goals, third-party re-hosts rank
# below the official site when both are visible.
AGGREGATOR_DOMAINS = (
    "cnet", "macupdate", "filehippo", "sourceforge", "softonic", "uptodown",
    "filehorse", "techspot", "softpedia", "majorgeeks", "chip.de", "soft112",
)


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _content_tokens(text: str, min_len: int = 3) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9.]+", (text or "").lower()) if len(t) >= min_len and t not in STOPWORDS]


def parse_negations(goal: str) -> Tuple[str, List[str]]:
    """Extracts trailing negation clauses ('... not the pull requests') -> (clean_goal, negated_phrases)."""
    phrases: List[str] = []

    def _capture(m):
        phrases.append(m.group(1).strip())
        return " "

    clean = _NEGATION_RE.sub(_capture, goal or "")
    clean = re.sub(r"\s{2,}", " ", clean).strip().rstrip(",").strip()
    return clean, phrases


def _desc_is_negated(desc: str, neg_phrases: Sequence[str]) -> bool:
    if not neg_phrases:
        return False
    d = _norm_ws(desc)
    for ph in neg_phrases:
        ptoks = _content_tokens(ph, min_len=3)
        if ptoks and all(t in d for t in ptoks):
            return True
    return False


def _temporal_direction(goal: str) -> Optional[str]:
    g = (goal or "").lower()
    if re.search(r"\b(oldest|earliest)\b", g):
        return "min"
    if re.search(r"\b(newest|latest|most\s+recent|brand\s+new)\b", g):
        return "max"
    return None


_SORT_INTENT_RE = re.compile(r"\b(popular|trending|most\s+viewed|top\s+rated|newest|latest|oldest|earliest)\b", re.IGNORECASE)


def _sort_intent_word(goal: str) -> Optional[str]:
    m = _SORT_INTENT_RE.search(goal or "")
    return m.group(1).lower() if m else None


def _ordinal_index(goal: str) -> Optional[int]:
    g = (goal or "").lower()
    m = re.search(
        r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|(\d+)(?:st|nd|rd|th))\s+"
        r"(?:video|vid|clip|item|result|entry|episode|part|upload|post)\b",
        g,
    )
    if not m:
        return None
    if m.group(1) in _WORD_NUM:
        return _WORD_NUM[m.group(1)]
    return int(m.group(2))


def _video_date_key(desc: str):
    """Parses a recency sort key from a video-link description.

    Returns (relative_minutes_ago, absolute_year); exactly one is non-None.
    """
    dl = (desc or "").lower()
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\s*ago\b", dl)
    if m:
        val = float(m.group(1))
        unit = m.group(2).rstrip("s")
        unit = {"min": "min", "hr": "hour"}.get(unit, unit)
        return val * _REL_UNITS.get(unit, 1e9), None
    m = re.search(r"\((?:[a-z]{3}\s+a\s*)?(?:[a-z]{3}\s+)?(?:\d{1,2},\s*)?(\d{4})\)", dl)
    if m:
        return None, int(m.group(1))
    m = re.search(r"\b(19\d{2}|20\d{2})\b", dl)
    if m:
        return None, int(m.group(1))
    return None, None


def _desc_has_version(desc: str, ver: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(ver)}(?![\d.])", desc or "") is not None


def _overlap_score(desc: str, toks: Sequence[str]) -> int:
    """Counts target tokens present in desc; boundary-aware so 'localllama' does not
    match 'localllamafans', but trailing-dot tokens ('llama.cpp') still match URLs."""
    d = _norm_ws(desc)
    hits = 0
    for t in toks:
        if t.endswith("."):
            if t in d:
                hits += 1
        elif re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", d):
            hits += 1
    return hits


def derive_action_from_element(chosen_ref: str, element_meta: Optional[Dict[str, Any]], description: str) -> str:
    """Deterministically binds the execution action from the chosen candidate's semantic role."""
    if chosen_ref == "opt_done":
        return "stop"
    if chosen_ref == "opt_scroll":
        return "scroll_down"
    if chosen_ref == "opt_human":
        return "request_human"
    if chosen_ref == "idk":
        return "abstain"

    meta = element_meta or {}
    role = str(meta.get("role", "")).lower()
    desc_l = description.lower()
    if meta.get("is_input") or role in ("search_input", "textbox", "searchbox", "combobox") or "search_input:" in desc_l:
        return "type_and_search"
    return "click"


class DecisionModelEngine:
    """Model-agnostic System-1 Decision Engine for browser and desktop automation."""

    def __init__(
        self,
        model_name_or_path: str = "d1-2B",
        device: str = "cuda",
        quant_4bit: Optional[bool] = None,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = None
        self._is_loaded = False
        self.load_time_s = 0.0
        self.peak_vram_mb = 0.0
        self.configure(model_name_or_path, quant_4bit=quant_4bit)

    def configure(self, model_name_or_path: str, quant_4bit: Optional[bool] = None):
        preset = MODEL_PRESETS.get(model_name_or_path)
        if preset:
            self.model_label = model_name_or_path
            self.model_path = preset["path"]
            self.quant_4bit = preset["quant_4bit"] if quant_4bit is None else quant_4bit
            self.supports_span_read = preset["supports_span_read"]
            self.media_kwarg = preset["media_kwarg"]
        else:
            self.model_label = os.path.basename(model_name_or_path.rstrip("\\/")) or model_name_or_path
            self.model_path = model_name_or_path
            is_large = any(k in self.model_label.lower() for k in ("2b", "3b", "4b", "7b"))
            self.quant_4bit = is_large if quant_4bit is None else quant_4bit
            self.supports_span_read = is_large
            self.media_kwarg = "media" if is_large else "images"

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        self._is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def switch_model(self, model_name_or_path: str, quant_4bit: Optional[bool] = None):
        self.unload()
        self.configure(model_name_or_path, quant_4bit=quant_4bit)
        self.load()

    def load(self):
        if self._is_loaded and self.model is not None:
            return

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        print(f"Loading {self.model_label} ({'4-bit NF4' if self.quant_4bit else 'FP16'})...", flush=True)
        if self.quant_4bit and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                dtype=self.dtype,
                device_map=self.device,
            )
        self.model.eval()
        _patch_d1_attention()
        self.load_time_s = time.time() - t0
        if torch.cuda.is_available():
            self.peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        self._is_loaded = True
        print(f"{self.model_label} ready in {self.load_time_s:.2f}s (VRAM: {self.peak_vram_mb:.1f} MB)", flush=True)

    def extract_query(self, goal: str) -> Dict[str, Any]:
        """Extracts the search string using native Span-Read (if supported) or NLP fallback."""
        if not self._is_loaded:
            self.load()

        clean_goal, _negs = parse_negations(goal)
        fallback = extract_search_text(clean_goal)
        if not self.supports_span_read:
            return {"query": fallback, "method": "nlp_fallback", "confidence": 1.0, "latency_ms": 0.0}

        t0 = time.time()
        with torch.no_grad():
            out = self.model.system_one(
                clean_goal,
                {
                    "search_query": {
                        "type": "text",
                        "instructions": (
                            "Extract ONLY the exact search query, channel name, repository name, or topic "
                            "to type into a search bar (exclude command verbs like open/find/watch and platform names)."
                        ),
                    }
                },
            )
        lat_ms = (time.time() - t0) * 1000.0
        ans = out.get("answers", {}).get("search_query", {})
        found = float(ans.get("found", 0.0))
        span_text = (ans.get("text") or "").strip()

        span_text = self._postclean_span(clean_goal, span_text)

        if found >= 0.55 and span_text:
            return {"query": span_text, "method": "span_read", "confidence": found, "latency_ms": lat_ms}
        return {"query": fallback, "method": "span_fallback", "confidence": found, "latency_ms": lat_ms}

    @staticmethod
    def _postclean_span(goal: str, span_text: str) -> str:
        """Cleans and completes a raw Span-Read extraction.

        1. Gap-fill: if the span is a literal substring of the goal, append the trailing
           content words the span dropped (e.g. 'flight AI 202' + ' status') up to the
           first platform qualifier / secondary clause.
        2. Strip conversational wrappers the span may have captured.
        """
        span_text = " ".join((span_text or "").split()).strip(' "\'')
        if not span_text:
            return span_text

        span_text = re.sub(r"^the\s+", "", span_text, flags=re.IGNORECASE).strip()

        goal_ws = " ".join((goal or "").split())
        idx = goal_ws.lower().find(span_text.lower())
        if idx >= 0 and not span_text.lower().startswith(("open ", "find ", "search ", "play ", "watch ", "check ", "look ")):
            suffix = goal_ws[idx + len(span_text):]
            suffix = re.split(
                r",|;"
                r"|\s+(?:on|in|at)\s+(?:youtube|yt|google|wikipedia|amazon|reddit|twitter|x|github|bing)\b"
                r"|\s+(?:and|then)\s+(?:play|open|click|watch|find|check|search|look|read|go|scroll)\b"
                r"|\s+not\s+|\bskip\s+"
                r"|\s+(?:youtube|yt)\s+(?:channel|video|page|website)\b"
                r"|\s+(?:channel|page|website|account|stream|repository|repo|video|vid|clip)\s*$",
                suffix,
                flags=re.IGNORECASE,
            )[0]
            suffix = re.sub(r"^\s*(?:and|then|the)\s+", " ", suffix).strip().strip(' "\'')
            if suffix:
                span_text = f"{span_text} {suffix}".strip()

        # Site-qualifier completion: for '<site> page for the <topic>' goals the model
        # spans only the topic; prepend the site qualifier so the search carries the
        # site identity (live L2 failure: 'requests library' landed on a lookalike page).
        m_pg = re.search(
            r'(?:open|go\s+to|visit|find|check)\s+(?:the\s+)?([a-z0-9][a-z0-9 .\-]*?)?\bpage\s+for\s+(?:the\s+)?(.+)$',
            goal_ws,
            flags=re.IGNORECASE,
        )
        if m_pg and span_text.lower() == (m_pg.group(2) or "").strip().lower():
            site = (m_pg.group(1) or "").strip()
            if site and site.lower() not in {"the", "a", "an", "official", "main", "home", "web"}:
                span_text = f"{site} {span_text}"

        # Strip trailing platform / wrapper qualifiers
        span_text = re.sub(
            r'\s+(?:on|in|at)\s+(?:youtube|yt|google|wikipedia|reddit|github)(?:\.com)?$',
            '', span_text, flags=re.IGNORECASE,
        ).strip()
        span_text = re.sub(
            r'\s+(?:channel|page|website|account|stream|repository|repo)\s*$',
            '', span_text, flags=re.IGNORECASE,
        ).strip()
        span_text = re.sub(
            r'^(?:the\s+)?official\s+(?:website|site|homepage|home\s+page|docs?|documentation)\s+of\s+',
            '', span_text, flags=re.IGNORECASE,
        ).strip()
        span_text = re.sub(r'\s+(?:on\s+)?(?:the\s+)?[a-z0-9.\-]+\s*channel$', '', span_text, flags=re.IGNORECASE).strip()
        span_text = re.sub(r'\s+works$', '', span_text, flags=re.IGNORECASE).strip()
        span_text = re.sub(r'\s+(?:talks?|speaks?|explains?|shows?)\s+(?:about\s+)?', ' ', span_text, flags=re.IGNORECASE).strip()
        # Goal-aware channel cut: if the goal ends "on <entity> channel" but the span dropped
        # the word "channel", strip the dangling "on <entity>" too
        m_chan = re.search(r'\s+on\s+([a-z0-9.\-]+)\s+channel\s*$', goal_ws, flags=re.IGNORECASE)
        if m_chan:
            span_text = re.sub(
                rf'\s+on\s+{re.escape(m_chan.group(1))}\s*(?:channel)?\s*$',
                '', span_text, flags=re.IGNORECASE,
            ).strip()
        m_new = re.match(r'^(?:the\s+)?(?:newest|latest|recent|first)\s+([a-zA-Z0-9_.\-]+)\s+(?:video|vid|clip)$', span_text, flags=re.IGNORECASE)
        if m_new:
            span_text = m_new.group(1).strip()
        return span_text

    def check_action_safety(self, action_description: str) -> Dict[str, Any]:
        """Runs a single-pass 'noul' safety check before executing potentially destructive/financial clicks."""
        if not self._is_loaded:
            self.load()

        t0 = time.time()
        with torch.no_grad():
            out = self.model.system_one(
                f"Proposed browser action: {action_description}",
                {
                    "requires_approval": {
                        "type": "noul",
                        "instructions": (
                            "Does this action delete data, spend money, place an order, or alter account security "
                            "so it requires human confirmation?"
                        ),
                    }
                },
            )
        lat_ms = (time.time() - t0) * 1000.0
        p_risky = float(out.get("answers", {}).get("requires_approval", {}).get("noul", 0.0))
        # Deterministic destructive-keyword floor (all models): compact decision models
        # under-predict some destructive verbs; safety outranks raw model confidence.
        if re.search(
            r"\b(?:delete|remove|pay|checkout|place\s+order|purchase|unsubscribe|deactivate|erase|transfer|"
            r"cancel|refund|withdraw|terminate|empty\s+trash|permanently)\b",
            action_description,
            flags=re.IGNORECASE,
        ):
            p_risky = max(p_risky, 0.85)
        return {"is_risky": p_risky >= 0.45, "risk_probability": p_risky, "latency_ms": lat_ms}

    def select_element_action(
        self,
        goal: str,
        current_url: str,
        page_title: str,
        elements: Dict[str, str],
        elements_meta: Optional[Dict[str, Dict[str, Any]]] = None,
        target_query: Optional[str] = None,
        history: Optional[List[str]] = None,
        image: Any = None,
        abstain_below: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Single forward pass decision:
        - Picks best target_element from criteria
        - Derives execution action deterministically from element semantics
        """
        if not self._is_loaded:
            self.load()

        if not elements:
            return {"error": "No interactive elements provided", "is_done": False}

        # Single-candidate fast path: d1 Choice requires >= 2 options; if only 1 candidate remains, select it directly
        if len(elements) == 1:
            only_ref, only_desc = next(iter(elements.items()))
            meta = (elements_meta or {}).get(only_ref)
            only_act = derive_action_from_element(only_ref, meta, only_desc)
            return {
                "is_done": only_act == "stop",
                "chosen_element": only_ref,
                "element_confidence": 1.0,
                "element_probabilities": {only_ref: 1.0},
                "action_type": only_act,
                "action_confidence": 1.0,
                "latency_ms": 0.0,
                "input_tokens": 0,
            }

        # Negation parsing: strip clauses like "..., not the pull requests" and remember them
        clean_goal, neg_phrases = parse_negations(goal)
        history = history or []
        effective = {
            ref: desc
            for ref, desc in elements.items()
            if not _desc_is_negated(desc, neg_phrases)
            # Placeholder links ("... (page does not exist)" / "... (la pagina non esiste)")
            # are never goal targets (live L5: full-string lock clicked a nonexistent page).
            and "does not exist" not in desc.lower()
            and "non esiste" not in desc.lower()
        }
        if not effective:
            effective = dict(elements)

        target = target_query or extract_search_text(clean_goal)
        url_l = current_url.lower()
        is_results_page = (
            "/results" in url_l
            or "search?" in url_l
            or "?q=" in url_l
            or "special:search" in url_l
            or "special%3asearch" in url_l
            or "?search=" in url_l
        )
        is_video_page = "/watch" in url_l or "/shorts/" in url_l
        is_channel_page = "/@" in url_l or "/channel/" in url_l or "/c/" in url_l
        wants_video = bool(re.search(r'\b(?:video|vid|clip|watch|play|first|latest|newest|oldest|earliest)\b', goal, flags=re.IGNORECASE))

        def _is_concrete(ref: str) -> bool:
            return not ref.startswith(("opt_", "idk"))

        def _is_input_ref(ref: str) -> bool:
            meta = (elements_meta or {}).get(ref) or {}
            desc_l = elements.get(ref, "").lower()
            return bool(meta.get("is_input")) or "search_input:" in desc_l or "textbox:" in desc_l

        concrete = [(ref, desc) for ref, desc in effective.items() if _is_concrete(ref)]

        def _locked_decision(ref: str, desc: str, prob: float = 1.0, note: str = "") -> Dict[str, Any]:
            meta = (elements_meta or {}).get(ref)
            act = derive_action_from_element(ref, meta, desc)
            return {
                "is_done": act == "stop",
                "chosen_element": ref,
                "element_confidence": prob,
                "element_probabilities": {ref: prob},
                "action_type": act,
                "action_confidence": prob,
                "latency_ms": 0.0,
                "input_tokens": 0,
                "locked": note or "deterministic",
            }

        # --- Policy 1: login / captcha / OTP wall -> human escalation (never auto-type credentials) ---
        if "opt_human" in effective and any(_CRED_FIELD_RE.search(d) for _, d in concrete) and _ACCOUNT_GOAL_RE.search(goal):
            return _locked_decision("opt_human", effective["opt_human"], note="login_captcha_human_escalation")

        # --- Policy 2: modal obstruction policy (opt-out dismissal vs gate affirmation) ---
        modal_items = [
            (ref, desc) for ref, desc in concrete
            if "[modal]" in desc.lower() or "blocked behind modal" in desc.lower()
            or _OPTOUT_CONTEXT_RE.search(desc.split(":", 1)[-1] if ":" in desc else desc)
        ]
        modal_buttons = [(ref, desc) for ref, desc in modal_items if "button" in desc.lower()]
        if modal_items:
            all_modal_txt = " ".join(d for _, d in modal_items)
            gate_ctx = bool(_GATE_CONTEXT_RE.search(all_modal_txt))
            optout_ctx = bool(_OPTOUT_CONTEXT_RE.search(all_modal_txt))
            affirm = [(r, d) for r, d in modal_buttons if _MODAL_AFFIRM_RE.search(d)]
            dismiss = [(r, d) for r, d in modal_buttons if _MODAL_DISMISS_RE.search(d)]
            if gate_ctx and affirm:
                # Age/region gate: only the affirmative button proceeds toward the goal
                return _locked_decision(affirm[0][0], affirm[0][1], note="modal_gate_affirmative")
            if dismiss and (optout_ctx or not gate_ctx):
                # Opt-out modals (notifications/newsletters/cookies): prefer the dismissive button
                return _locked_decision(dismiss[0][0], dismiss[0][1], note="modal_optout_dismiss")
            # Informational-only overlays (terms/privacy/copyright links, no action buttons)
            # are passive notices: clicking their links derails the task. Drop them from the
            # candidate pool entirely and act on the underlying page instead.
            modal_info_only = not modal_buttons and all(
                "link" in d.lower() and _MODAL_INFO_LINK_RE.search(d) for _, d in modal_items
            )
            if modal_info_only:
                modal_refs = {r for r, _ in modal_items}
                remaining = {r: d for r, d in effective.items() if r not in modal_refs}
                if remaining:
                    effective = remaining
                    concrete = [(ref, effective[ref]) for ref in effective if _is_concrete(ref)]
                    modal_items = []
            # No deterministic modal action: restrict candidates to modal elements so the
            # model cannot click content links blocked behind the modal
            if modal_buttons or len(modal_items) >= 2:
                pool = modal_buttons or modal_items
                non_info = [(r, d) for r, d in pool if not _MODAL_INFO_LINK_RE.search(d)]
                keep = dict(non_info or pool)
                if len(keep) >= 2:
                    effective = keep
                    concrete = [(ref, effective[ref]) for ref in effective if _is_concrete(ref)]

        # --- Policy 3: temporal / ordinal video lock (oldest / newest / Nth) ---
        if wants_video and len(concrete) >= 2:
            video_items = [
                (ref, desc) for ref, desc in concrete
                if ((elements_meta or {}).get(ref) or {}).get("role") == "video_link"
                or re.match(r"video_link:", desc, flags=re.IGNORECASE)
            ]
            direction = _temporal_direction(goal)
            if direction and len(video_items) >= 2:
                dated = [(ref, desc, _video_date_key(desc)) for ref, desc in video_items]
                rels = [k[0] for _, _, k in dated]
                years = [k[1] for _, _, k in dated]
                ordered = None
                if all(r is not None for r in rels):
                    # Relative ages: larger 'minutes ago' = older
                    ordered = sorted(dated, key=lambda x: -x[2][0])
                elif all(y is not None for y in years):
                    # Absolute years: smaller year = older
                    ordered = sorted(dated, key=lambda x: x[2][1])
                if ordered is not None:
                    pick = ordered[0] if direction == "min" else ordered[-1]
                    return _locked_decision(pick[0], pick[1], note=f"temporal_{direction}")
            ordinal = _ordinal_index(goal)
            if ordinal:
                numbered = [(ref, desc) for ref, desc in video_items if re.match(r"^\s*(\d+)[.)]", desc)]
                if numbered and len(numbered) >= ordinal:
                    m = re.match(r"^\s*(\d+)[.)]", numbered[ordinal - 1][1])
                    if m and int(m.group(1)) == ordinal:
                        return _locked_decision(numbered[ordinal - 1][0], numbered[ordinal - 1][1], note="ordinal_numbered")
                elif len(video_items) >= ordinal:
                    pick = video_items[ordinal - 1]
                    return _locked_decision(pick[0], pick[1], note="ordinal_position")

        # --- Policy 4: deterministic completion stop (mirrors orchestrator check_goal_reached) ---
        # Runs before target locks: if we are already on the target page/video, stop must win
        # over locks that would re-click visible target mentions (e.g. the channel link).
        if "opt_done" in effective and not is_results_page:
            ctx = _norm_ws(page_title + " " + current_url)
            if not is_video_page and not (is_channel_page and wants_video):
                toks = _content_tokens(target)
                if toks and all(t in ctx for t in toks):
                    return _locked_decision("opt_done", effective["opt_done"], note="completion_stop")
            if is_video_page and wants_video:
                vtoks = _content_tokens(target)
                last_click = next((h for h in reversed(history) if h.lower().startswith("clicked ")), "")
                combined = _norm_ws(page_title + " " + last_click)
                if vtoks:
                    hits = sum(1 for t in vtoks if t in combined)
                    if hits >= max(1, (len(vtoks) + 1) // 2):
                        return _locked_decision("opt_done", effective["opt_done"], note="completion_stop_video")

        # --- Policy 5: sort/filter-intent lock ('popular videos' -> the Popular sort button) ---
        sort_word = _sort_intent_word(goal)
        if sort_word and len(concrete) >= 2:
            sort_matches = [
                (ref, desc) for ref, desc in concrete
                if re.search(rf"(?<![a-z0-9]){re.escape(sort_word)}(?![a-z0-9])", _norm_ws(desc))
                and "selected" not in desc.lower()
            ]
            if len(sort_matches) == 1:
                return _locked_decision(sort_matches[0][0], sort_matches[0][1], note=f"sort_intent_{sort_word}")

        # --- Policy 6: version-pin lock (2.11.4 vs 2.11.40, changelog pagination, etc.) ---
        target_versions = _VERSION_RE.findall(target)
        if target_versions:
            for ver in target_versions:
                ver_matches = [(ref, desc) for ref, desc in concrete if _desc_has_version(desc, ver)]
                if len(ver_matches) == 1:
                    return _locked_decision(ver_matches[0][0], ver_matches[0][1], note=f"version_pin_{ver}")

        # --- Policy 5: unique full-string target lock (zero forward pass when target is literally visible) ---
        t_norm = _norm_ws(target)
        if len(t_norm) >= 5:
            full_matches = [(ref, desc) for ref, desc in concrete if t_norm in _norm_ws(desc)]
            if len(full_matches) == 1:
                return _locked_decision(full_matches[0][0], full_matches[0][1], note="unique_full_string_target")

        # --- Policy 7: multi-input disambiguation (site docs vs forums vs blog search) ---
        target_tokens_pre = _content_tokens(target) or _content_tokens(clean_goal)
        if len(target_tokens_pre) >= 1:
            input_items = [(ref, desc) for ref, desc in concrete if _is_input_ref(ref)]
            non_input_hits = any(_overlap_score(desc, target_tokens_pre) > 0 for ref, desc in concrete if not _is_input_ref(ref))
            if len(input_items) >= 2 and not non_input_hits:
                scored_inputs = [(_overlap_score(desc, target_tokens_pre), ref, desc) for ref, desc in input_items]
                scored_inputs.sort(reverse=True)
                if scored_inputs[0][0] >= 1 and scored_inputs[0][0] > scored_inputs[1][0]:
                    return _locked_decision(scored_inputs[0][1], scored_inputs[0][2], note="input_disambiguation")

        # --- Policy 8: suppress search inputs on results pages (no pointless re-search) ---
        # The URL itself (?q=, /results) is proof a search already happened.
        if is_results_page:
            non_inputs = [(ref, desc) for ref, desc in concrete if not _is_input_ref(ref)]
            if non_inputs:
                effective = {ref: desc for ref, desc in effective.items() if not _is_input_ref(ref)}
                concrete = non_inputs

        # --- Policy 9: zero-overlap abstention (nothing on the page matches the goal) ---
        if "idk" in effective and not any(_is_input_ref(ref) for ref, _ in concrete):
            toks = _content_tokens(target) or _content_tokens(clean_goal)
            if toks and all(_overlap_score(desc, toks) == 0 for _, desc in concrete):
                return _locked_decision("idk", effective["idk"], prob=1.0, note="zero_overlap_abstention")

        # --- Policy 9b: premature-stop veto on watch pages ---
        # A stop is only credible when the topic actually matches the video title or the
        # video link just clicked (live L4: 2B stopped on 'Agent Zero...' while the goal
        # was a wifi-security video). Without corroboration, remove opt_done from the menu.
        if is_video_page and wants_video:
            _ACTION_VERBS = {"play", "watch", "see", "open", "click", "stream"}
            topic_tokens = [t for t in re.findall(r"[a-z0-9]+", target.lower()) if len(t) >= 3 and t not in _ACTION_VERBS]
            title_l = (page_title or "").lower()
            clicked_descs = " ".join(h.lower() for h in history if h.startswith("Clicked "))
            corroborated = any(
                t in title_l or t in clicked_descs for t in topic_tokens
            )
            if topic_tokens and not corroborated:
                effective.pop("opt_done", None)

        # --- Policy 10: instructions & model call ---
        history_text = " -> ".join(history[-4:]) if history else "Start of task"
        state = {
            "user_goal": goal,
            "target_query": target,
            "current_page_title": page_title,
            "current_url": current_url,
            "action_history": history_text,
        }

        # Check if any visible link/button already directly matches target keywords
        target_tokens = [t for t in re.findall(r"[a-z0-9.]+", target.lower()) if len(t) >= 3]
        has_direct_match = any(
            not ("search_input" in desc.lower() or "textbox" in desc.lower())
            and any(tok in desc.lower() for tok in target_tokens)
            for _, desc in concrete
        )

        has_search_box = any(
            "search_input" in desc.lower() or "searchbox" in desc.lower()
            for _, desc in concrete
        )
        has_modal = any("[modal]" in desc.lower() for _, desc in concrete)

        if is_video_page:
            instructions_elem = (
                f"We are on video page {current_url} ({page_title}). Goal: '{goal}'. "
                f"Pick 'opt_done' if the target video is open, or pick the next element to click."
            )
        elif is_channel_page and wants_video:
            direction = _temporal_direction(goal)
            ordinal = _ordinal_index(goal)
            if direction == "min":
                instructions_elem = (
                    f"We are on the '{target}' channel page ({page_title}). Goal: '{goal}'. "
                    f"Pick the OLDEST video_link (earliest upload date), or 'tab: Videos' if no video links are listed."
                )
            elif direction == "max":
                instructions_elem = (
                    f"We are on the '{target}' channel page ({page_title}). Goal: '{goal}'. "
                    f"Pick the NEWEST video_link (most recent upload date), or 'tab: Videos' if no video links are listed."
                )
            elif ordinal:
                instructions_elem = (
                    f"We are on the '{target}' channel page ({page_title}). Goal: '{goal}'. "
                    f"Pick the {ordinal}th video_link in list order, or 'tab: Videos' if no video links are listed."
                )
            else:
                instructions_elem = (
                    f"We are on the '{target}' channel page ({page_title}). "
                    f"Pick the latest/top video_link to play, or 'tab: Videos' if no video links are listed."
                )
        elif is_results_page:
            instructions_elem = (
                f"We are on search results for '{target}' ({page_title}). "
                f"Pick the matching channel_link, video_link, or result link for '{target}' (do NOT re-pick search_input)."
            )
        elif has_modal:
            instructions_elem = (
                f"A modal dialog is blocking {page_title}. Dismiss or accept the modal first before proceeding with '{goal}'."
            )
        elif has_direct_match:
            instructions_elem = (
                f"Goal: '{goal}'. A matching link for '{target}' is already visible on this page ({page_title}). "
                f"Click that matching link directly (do NOT pick search_input)."
            )
        elif has_search_box:
            instructions_elem = (
                f"Goal: '{goal}'. No link matching '{target}' is visible on {page_title} yet. "
                f"Select the search_input box to type and search for '{target}'."
            )
        else:
            instructions_elem = (
                f"Goal: '{goal}'. Target: '{target}'. Page: {current_url} ({page_title}). "
                f"Pick the most relevant link, button, or scroll action to reach '{target}'."
            )

        questions = {
            "target_element": {
                "type": "choice",
                "instructions": instructions_elem,
                "criteria": effective,
            }
        }

        kwargs: Dict[str, Any] = {}
        if image is not None:
            kwargs[self.media_kwarg] = [image]

        t0 = time.time()
        with torch.no_grad():
            output = self.model.system_one(state, questions, **kwargs)
        lat_ms = (time.time() - t0) * 1000.0

        elem_ans = output.get("answers", {}).get("target_element", {})
        probs = elem_ans.get("probabilities", {})
        chosen_elem = elem_ans.get("choice", "")
        chosen_conf = float(elem_ans.get("confidence", 0.0))

        # Model-agnostic calibrated abstention over probabilities
        if abstain_below is not None and "idk" in probs:
            others = {k: v for k, v in probs.items() if k != "idk"}
            total_others = sum(others.values())
            if total_others <= 0 or max(others.values(), default=0.0) < abstain_below * total_others:
                chosen_elem = "idk"
                chosen_conf = float(probs["idk"])

        # Token-overlap floor: if the model picked a concrete element with zero keyword
        # overlap while another concrete element matches the target, override once.
        if (
            _is_concrete(chosen_elem)
            and not _is_input_ref(chosen_elem)
            and chosen_elem in effective
        ):
            toks = _content_tokens(target) or _content_tokens(clean_goal)
            if toks:
                # Drop non-discriminative tokens present in >60% of candidates (e.g. the
                # shared head word on a disambiguation page) so senses are compared fairly
                cand_count = len(concrete)
                discriminative = [
                    t for t in toks
                    if sum(1 for _, d in concrete if t in _norm_ws(d)) <= 0.6 * cand_count
                ]
                toks = discriminative or toks
                download_intent = bool(re.search(r"\b(?:download|install|get)\b", goal, flags=re.IGNORECASE))

                def _final_score(desc: str) -> int:
                    d = _norm_ws(desc)
                    score = _overlap_score(desc, toks)
                    if download_intent:
                        if any(a in d for a in AGGREGATOR_DOMAINS):
                            score -= 3  # third-party re-host on a download goal
                        if re.search(r"\b(?:youtube|tutorials?|video)\b", d):
                            score -= 1  # a tutorial video is not the download page
                        if re.search(r"\b[a-z0-9\-]+\.(?:com|org|net|io|dev)\b", d):
                            score += 1  # bare official-domain signal (e.g. obsproject.com)
                    return score

                chosen_score = _final_score(effective.get(chosen_elem, ""))
                if chosen_score <= 0:
                    scored = sorted(
                        ((_final_score(desc), ref) for ref, desc in concrete if not _is_input_ref(ref)),
                        reverse=True,
                    )
                    if scored and scored[0][0] >= 1:
                        best_score, best_ref = scored[0]
                        chosen_elem = best_ref
                        chosen_conf = float(probs.get(best_ref, 0.0))

        chosen_desc = elements.get(chosen_elem, "")
        meta = (elements_meta or {}).get(chosen_elem)
        chosen_act = derive_action_from_element(chosen_elem, meta, chosen_desc)

        return {
            "is_done": chosen_act == "stop",
            "chosen_element": chosen_elem,
            "element_confidence": chosen_conf,
            "element_probabilities": probs,
            "action_type": chosen_act,
            "action_confidence": chosen_conf,
            "latency_ms": lat_ms,
            "input_tokens": output.get("usage", {}).get("input_tokens", 0),
        }

    def batch_decide(self, requests: Sequence[tuple]) -> List[Dict[str, Any]]:
        """Evaluates multiple (state, questions) requests in one batched GPU forward pass."""
        if not self._is_loaded:
            self.load()
        with torch.no_grad():
            return self.model.system_one_batch(requests)


# Backward-compatible alias
D1DecisionEngine = DecisionModelEngine
