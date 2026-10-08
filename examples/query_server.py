"""Minimal client for a model served with `js2 serve <model> --max-model-len N` (tunnel on localhost:8000).

    python examples/query_server.py "Explain KV cache in one paragraph."

Uses only the standard library so it runs anywhere. For real evals send many requests concurrently
(asyncio + the `openai` package); vLLM prints "Maximum concurrency for N tokens: Cx" at startup.
"""
import json, sys, urllib.request

base = "http://localhost:8000/v1"
model = json.load(urllib.request.urlopen(f"{base}/models"))["data"][0]["id"]
body = {"model": model, "messages": [{"role": "user", "content": sys.argv[1] if len(sys.argv) > 1 else "Hello!"}],
        "max_tokens": 256, "temperature": 0.7}
req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
out = json.load(urllib.request.urlopen(req))
print(f"[{model}] {out['choices'][0]['message']['content']}")
print("usage:", out["usage"])
