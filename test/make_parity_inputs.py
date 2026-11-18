"""Inputs for the JS/Python parity test: test sets + random gazetteer keys, glossary forms, synthetic addresses."""
import json, os, random, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "runtime")); import normalizer as N
random.seed(7)
items = []
for f in ("cases.json", "blind.json"):
    items += [[c["op"], c["in"]] for c in json.load(open(os.path.join(ROOT, "test", f), encoding="utf-8"))]
G = list(N.gaz().keys()); random.shuffle(G)
items += [[random.choice(["translate", "transliterate"]), k] for k in G[:1500]]
GL = N.gloss()
forms = [(l, f) for l, d in GL.items() if not l.startswith("_") for f in d]
random.shuffle(forms)
for l, f in forms[:600]:
    items.append(["translate", f])
streets = {"ru": "ул.", "es": "Calle", "pt": "Rua", "fr": "rue", "it": "Via", "pl": "ul.", "tr": "Caddesi", "id": "Jl.", "vi": "Đường", "de": "Straße"}
for _ in range(500):
    l, st = random.choice(list(streets.items()))
    a, b = random.choice(G), random.choice(G)
    items.append([random.choice(["translate", "transliterate"]), f"{st} {a} {random.randint(1,300)}, {b}"])
for _ in range(300):
    items.append(["translate", " ".join(random.choice(forms)[1] for _ in range(random.randint(2, 4)))])
json.dump(items, open(os.path.join(ROOT, "test", "parity_inputs.json"), "w", encoding="utf-8"), ensure_ascii=False)
res = N.normalize([tuple(x) for x in items])
json.dump(res, open(os.path.join(ROOT, "test", "parity_python.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("inputs", len(items))
