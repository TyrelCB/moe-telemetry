#!/usr/bin/env python3
"""HumanEval -> humaneval_local.jsonl (prompt/test/entry_point/task_id)."""
import gzip, io, json, urllib.request

raw = urllib.request.urlopen(
    "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
    timeout=120).read()
rows = [json.loads(l) for l in gzip.GzipFile(fileobj=io.BytesIO(raw)).read().decode().splitlines()]
with open("humaneval_local.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps({"task_id": r["task_id"], "prompt": r["prompt"],
                            "test": r["test"], "entry_point": r["entry_point"]}) + "\n")
print("wrote humaneval_local.jsonl:", len(rows))
