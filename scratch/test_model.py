import httpx
import time

t_preclosed = """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
<think>
</think>
"""

r = httpx.post("http://127.0.0.1:11434/api/create", json={
    "model": "qwen3-fast",
    "from": "qwen3:4b",
    "template": t_preclosed,
    "parameters": {
        "stop": ["<|im_start|>", "<|im_end|>"]
    },
    "stream": False
}, timeout=60)
print("Create fast response:", r.status_code)

print("Testing qwen3-fast...")
t0 = time.perf_counter()
r2 = httpx.post("http://127.0.0.1:11434/api/chat", json={
    "model": "qwen3-fast",
    "messages": [
        {"role": "system", "content": "You are the receptionist for Karthi Dental Clinic. Answer in one short sentence directly."},
        {"role": "user", "content": "hi"}
    ],
    "stream": False
}, timeout=30)
dur = round((time.perf_counter() - t0) * 1000)
d = r2.json()
m = d.get("message", {})
print(f"Duration: {dur}ms | Eval: {d.get('eval_count')} | Content: {repr(m.get('content', ''))}")
