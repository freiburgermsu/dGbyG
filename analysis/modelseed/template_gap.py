"""How much of the ModelSEED template reaction gap (no free-energy estimate from
eQuilibrator, dGPredictor or group contribution) does dGbyG fill?

Reads ../ModelSEEDTemplates/templates/v7.0/*.json and the reaction JSON of the
ModelSEEDDatabase checkout carrying thermodynamics['dGbyG'] (PR #323), plus the
dGbyG staged table for per-reaction statuses. Coverage rule as in the MSDB
manuscript: dg < 1e7 and |sigma| < 2500.
Run: ~/Documents/py_venv/bin/python -B analysis/modelseed/template_gap.py
"""
import csv, glob, json, collections, sys
from pathlib import Path
HOME = Path.home() / "Documents"
TPL = HOME / "ModelSEEDTemplates" / "templates" / "v7.0"
DB = HOME / "ModelSEEDDatabase-dgbyg"
OUT = Path(__file__).resolve().parent / "results" / "templates"
SRC = ["eQuilibrator", "dGPredictor", "Group contribution"]

rx = {}
for f in sorted(glob.glob(str(DB / "Biochemistry" / "reaction_*.json"))):
    for r in json.load(open(f)):
        rx[r["id"]] = r
status = {}
with open(DB / "Biochemistry/Thermodynamics/dGbyG/ModelSEED_Reaction_Energies.tsv") as fh:
    rows = (l for l in fh if not l.startswith("#"))
    for row in csv.DictReader(rows, delimiter="\t"):
        status[row["reaction_id"]] = row["status"]

def cov(r, s):
    v = (r.get("thermodynamics") or {}).get(s)
    try:
        return v is not None and v[0] != "" and float(v[0]) < 1e7 and abs(float(v[1])) < 2500
    except (TypeError, ValueError):
        return False

lines = []
def say(x=""):
    print(x); lines.append(x)

tpls = {}
for f in sorted(TPL.glob("*.json")):
    t = json.load(open(f))
    ids = {x["reaction_ref"].split("/")[-1] for x in t["reactions"]}
    ids = {i for i in ids if i.startswith("rxn")}
    tpls[f.stem] = (ids, {x["reaction_ref"].split("/")[-1]: x["type"] for x in t["reactions"]})
tpls["union of all four"] = (set().union(*(v[0] for v in tpls.values())), {})

for name, (ids, types) in tpls.items():
    missing = [i for i in ids if i not in rx]
    have = [i for i in ids if i in rx]
    gap = [i for i in have if not any(cov(rx[i], s) for s in SRC)]
    fill = [i for i in gap if cov(rx[i], "dGbyG")]
    ops = collections.Counter(rx[i]["thermodynamics"]["dGbyG"][2] for i in fill)
    st = collections.Counter(status.get(i, "not in table") for i in gap if i not in fill)
    say(f"== {name}: {len(ids)} distinct reactions ({len(missing)} not in the database)")
    say(f"   no eQ/dGP/GC energy: {len(gap)} ({100*len(gap)/len(have):.1f}%)")
    say(f"   dGbyG fills {len(fill)} ({100*len(fill)/max(len(gap),1):.1f}% of the gap); calls {dict(ops)}; "
        f"committed {ops['>']+ops['<']}, resolved {ops['>']+ops['<']+ops['=']}")
    say(f"   still without an energy: {len(gap)-len(fill)}; dGbyG status: {dict(st.most_common())}")
    if types:
        tc = collections.Counter(types.get(i) for i in gap); tf = collections.Counter(types.get(i) for i in fill)
        say(f"   gap by template type: " + ", ".join(f"{k} {tf[k]}/{v}" for k, v in tc.most_common()))
    # transport and ModelSEED status among filled
    say(f"   filled: transport {sum(bool(rx[i].get('is_transport')) for i in fill)}, "
        f"ModelSEED status {dict(collections.Counter(rx[i].get('status','') for i in fill).most_common(5))}")
    if name == "union of all four":
        sig = sorted(float(rx[i]["thermodynamics"]["dGbyG"][1]) for i in fill)
        say(f"   filled sigma: median {sig[len(sig)//2]:.2f} kcal/mol, sigma>5: {sum(s>5 for s in sig)}")
        llm = collections.Counter(((rx[i]['thermodynamics'].get('LLMs') or ['','',''])[2]) for i in gap)
        say(f"   LLM calls on the gap: {dict(llm)}")
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / "template_gap_reactions.tsv", "w") as fh:
            fh.write("rxn\tdgbyg_status\tdgbyg_dg\tdgbyg_sd\tdgbyg_op\tis_transport\tms_status\n")
            for i in sorted(gap):
                v = rx[i]["thermodynamics"].get("dGbyG", ["", "", ""])
                fh.write(f"{i}\t{status.get(i,'')}\t{v[0]}\t{v[1]}\t{v[2]}\t{rx[i].get('is_transport')}\t{rx[i].get('status','')}\n")
(OUT / "template_gap.txt").write_text("\n".join(lines) + "\n")
