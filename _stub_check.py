"""_stub_check.py: No-GPU sanity pass — stubs the d1 model and runs the V2 suite
to verify deterministic locks fire and no exceptions occur."""

import sys, os, re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import d1_engine
from d1_engine import DecisionModelEngine
from run_iterations_v2 import run_engine_suite_v2, run_orch_suite_v2, infer_meta_v2
from test_suite_v2 import BROWSER_CASES_V2, ORCH_CASES_V2

# Stub: never load a real model; system_one returns a deliberately WRONG pick
# (first concrete element) so we can see which cases the deterministic layer saves.
class StubModel:
    def system_one(self, state, questions, **kw):
        answers = {}
        for qname, q in questions.items():
            if q is None or not isinstance(q, dict) or "criteria" not in q:
                answers[qname] = {"noul": 0.1}
                continue
            crit = q["criteria"]
            first = next(iter(crit.keys()))
            answers[qname] = {"choice": first, "confidence": 0.5, "probabilities": {k: 0.1 for k in crit}}
        return {"answers": answers, "usage": {"input_tokens": 0}}

eng = DecisionModelEngine.__new__(DecisionModelEngine)
eng.model_label = "stub"
eng.model = StubModel()
eng._is_loaded = True
eng.supports_span_read = True
eng.media_kwarg = "media"
eng.device = "cpu"
eng.dtype = None
eng.quant_4bit = False
d1_engine.DecisionModelEngine.load = lambda self: None

def log(msg):
    print(msg)

run_engine_suite_v2(eng, log)
print("\n--- ORCH (stub) ---")
try:
    run_orch_suite_v2(eng, log)
except Exception as e:
    import traceback; traceback.print_exc()
