import random, json, collections
from datasets import load_dataset
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
by = collections.defaultdict(list)
for r in ds: by[r["difficulty"]].append(r["instance_id"])
print({k: len(v) for k, v in by.items()})
rng = random.Random(20260929)
# Mirror Verified: ~39% easy, ~52% medium, ~9% hard; 50 tasks.
want = {"<15 min fix": 20, "15 min - 1 hour": 25, "1-4 hours": 5}
pick = []
for k, n in want.items(): pick += rng.sample(sorted(by[k]), n)
pick.sort()
json.dump(pick, open("instances.json", "w"), indent=0)
print(len(pick), collections.Counter(i.split("__")[0] for i in pick))
