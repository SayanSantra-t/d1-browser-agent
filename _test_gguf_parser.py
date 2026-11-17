import gguf_agent_live as g

a = g.parse_action('{"action":"click","ref":"e12"}', {"e12"})
b = g.parse_action('{"name":"act","arguments":{"kind":"click","ref":"e7"}}', {"e7"})
c = g.parse_action('Let me think... <think>hmm</think>```json\n{"tool":"act","args":{"kind":"type","ref":"e3","text":"obs download"}}\n``` done', {"e3"})
d = g.parse_action('{"action":"done"}', set())
e = g.parse_action('I would click e5 now', {"e5"})
f = g.parse_action('no json at all here', {"e5"})
print("click:", a)
print("neo-act:", b)
print("think+fence+type:", c)
print("done:", d)
print("prose-ref:", e)
print("fallback:", f)
assert a["kind"] == "click" and a["ref"] == "e12"
assert b["kind"] == "click" and b["ref"] == "e7"
n = g.parse_action('{\n "tool": "e2",\n "args": {\n  "page": 1,\n  "action": "link",\n  "value": "OBS Download Page - obsproject.com"\n }\n}', {"e2"})
print("duckets-native:", n)
assert n["kind"] == "click" and n["ref"] == "e2"
n2 = g.parse_action('{"tool": "e1", "args": {"action": "type", "value": "wifi security"}}', {"e1"})
print("duckets-native-type:", n2)
assert n2["kind"] == "type" and n2["text"] == "wifi security"
assert c["kind"] == "type" and c["text"] == "obs download"
assert d["kind"] == "done"
assert e["kind"] == "click" and e["ref"] == "e5"
assert f["kind"] == "unparsed"
print("ALL PARSER TESTS PASS")
