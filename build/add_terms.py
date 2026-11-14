"""Add words or phrases to the Fast word list: instant, no training. Every model uses them at once.
  python build/add_terms.py terms.tsv          lines: <lang> TAB <foreign word or phrase> TAB <english>
  e.g.   de<TAB>Wasserkocher<TAB>kettle
The 70 tuned cases and the test labels are re-run; if any score drops the change is undone."""
import json, os, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, os.path.join(ROOT, "runtime"))
LANGS = ["zh", "es", "pt", "ar", "ru", "ja", "ko", "de", "fr", "hi", "tr", "id", "vi", "it", "pl", "th"]


def scores():
    code = ("import sys,json,os;sys.path[:0]=['runtime','build'];import normalizer as F\n"
            "from analyze_student import clean,chrf\n"
            "C=json.load(open('test/cases.json',encoding='utf-8'));G=json.load(open('test/gold_products.json',encoding='utf-8'))['items']\n"
            "c=sum(F.process(x['op'],x['in'])['output'].casefold()==x['want'].casefold() for x in C)\n"
            "g=sum(max(chrf(clean(F.process('translate',s)['output']),clean(e)) for e in it['en']) for it in G for s in it['src'].values())/sum(len(it['src']) for it in G)\n"
            "print(json.dumps([c,round(g,2)]))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    return json.loads(r.stdout.strip().split("\n")[-1])


def main(path):
    import normalizer as F
    gp = os.path.join(ROOT, "data", "glossary.json"); G = json.load(open(gp, encoding="utf-8")); before = scores(); n = 0
    for line in open(path, encoding="utf-8"):
        parts = [x.strip() for x in line.rstrip("\n").split("\t")]
        if len(parts) != 3 or parts[0] not in LANGS or not parts[1] or not parts[2]:
            if line.strip():
                print("skipped (need: lang TAB foreign TAB english, lang one of", " ".join(LANGS) + "):", line.strip())
            continue
        l, src, en = parts; k = F.norm_key(src)
        if G[l].get(k) != en.lower():
            G[l][k] = en.lower(); n += 1
    shutil.copy(gp, gp + ".before_add"); json.dump(G, open(gp, "w", encoding="utf-8"), ensure_ascii=False)
    after = scores(); print("terms added or changed:", n, "| tuned cases", before[0], "->", after[0], "| test labels (Fast, chrF)", before[1], "->", after[1])
    if after[0] < before[0] or after[1] < before[1] - 0.05:
        shutil.copy(gp + ".before_add", gp); print("a score dropped: change undone (data/glossary.json restored)"); sys.exit(1)
    print("kept. data/glossary.json updated; the browser build needs sh build/stage_web.sh")


if __name__ == "__main__":
    main(sys.argv[1])
