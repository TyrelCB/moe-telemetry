#!/usr/bin/env python3
"""Out-of-domain probe set: GSM8K math + HumanEval code, same JSONL schema."""
import gzip, io, json, urllib.request

out = []
# GSM8K math word problems
url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
rows = [json.loads(l) for l in urllib.request.urlopen(url, timeout=60).read().decode().splitlines()][:24]
for r in rows:
    out.append({"messages": [{"role": "user", "content": r["question"].strip() + "\n" + r["answer"].strip()}]})
# HumanEval code prompts + reference solution
url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
raw = urllib.request.urlopen(url, timeout=60).read()
rows = [json.loads(l) for l in gzip.GzipFile(fileobj=io.BytesIO(raw)).read().decode().splitlines()][:8]
for r in rows:
    text = (r["prompt"] + r["canonical_solution"]).strip()
    out.append({"messages": [{"role": "user", "content": text}]})

with open("ood.jsonl", "w") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
print(f"wrote ood.jsonl: {len(out)} samples "
      f"(mean {sum(len(o['messages'][0]['content']) for o in out)//len(out)} chars)")
