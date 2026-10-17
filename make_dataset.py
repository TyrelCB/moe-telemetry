#!/usr/bin/env python3
"""Extract user turns from Claude Code transcripts -> calibration JSONL.
Filter: 40-1200 chars, no near-duplicates, stratified by file for variety."""
import glob, json, random, hashlib, sys, re

random.seed(1337)
MAX_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
files = sorted(glob.glob('/home/tyrel/.claude/projects/*/*.jsonl'))
cands = []
seen = set()
for fp in files:
    per_file = []
    try:
        with open(fp, errors='replace') as f:
            for line in f:
                if '"type":"user"' not in line and '"role":"user"' not in line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get('type') != 'user' or d.get('isMeta'):
                    continue
                msg = d.get('message') or {}
                c = msg.get('content')
                texts = []
                if isinstance(c, str):
                    texts = [c]
                elif isinstance(c, list):
                    texts = [p.get('text', '') for p in c if isinstance(p, dict) and p.get('type') == 'text']
                for t in texts:
                    t = t.strip()
                    if not (40 <= len(t) <= 1200):
                        continue
                    if t.startswith('<') or 'tool_use_id' in t or 'system-reminder' in t:
                        continue
                    h = hashlib.md5(re.sub(r'\W+', ' ', t.lower()).encode()).hexdigest()
                    if h in seen:
                        continue
                    seen.add(h)
                    per_file.append(t)
    except Exception:
        continue
    random.shuffle(per_file)
    cands.append(per_file)

# round-robin across files for project diversity
out, round_i = [], 0
while len(out) < MAX_SAMPLES and any(cands):
    progressed = False
    for lst in cands:
        if round_i < len(lst):
            out.append(lst[round_i]); progressed = True
            if len(out) >= MAX_SAMPLES:
                break
    round_i += 1
    if not progressed:
        break

with open('/home/tyrel/projects/moe-telemetry/calib_transcripts.jsonl', 'w') as f:
    for t in out:
        f.write(json.dumps({'messages': [{'role': 'user', 'content': t}]}) + '\n')
print(f'wrote {len(out)} samples from {len(files)} transcripts')
