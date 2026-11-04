import json, random
from datasets import load_dataset
ds = {r["instance_id"]: r for r in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")}
pick = json.load(open("instances.json"))
bad = [i for i in pick if i.startswith("matplotlib__")]
rng = random.Random(20260930)
keep = [i for i in pick if i not in bad]
for b in bad:
    d = ds[b]["difficulty"]
    pool = sorted(i for i, r in ds.items() if r["difficulty"] == d and not i.startswith("matplotlib__") and i not in keep)
    keep.append(rng.choice(pool))
    print(b, d, "->", keep[-1])
keep.sort()
json.dump(keep, open("instances.json", "w"), indent=0)
open("images.txt","w").write("".join("docker.io/swebench/sweb.eval.x86_64."+i.replace("__","_1776_").lower()+":latest\n" for i in keep))
