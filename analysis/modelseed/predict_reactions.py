#!/usr/bin/env python
"""Stage 2: dGbyG reaction energies and direction calls for every ModelSEED reaction.

Reads Stage 1's per-head compound matrix and sums it over each reaction's
stoichiometry. The reaction uncertainty is the standard deviation of the
SUMMED per-head vectors -- exactly dGbyG's Reaction.standard_dGr_prime -- not
compound sigmas added in quadrature (head errors are strongly correlated:
ATP's compound sigma is 5.7 kcal/mol, while ATP hydrolysis written with
neutral SMILES has sigma 0.08 and rxn00062 on ModelSEED structures 0.77).

Stoichiometry is netted by compound across compartments, which is what the
eQuilibrator and dGPredictor sublists already do (see the EQ:transport note in
ModelSEEDDatabase/Scripts/Thermodynamics/README.md), so all three sources score
the same compartment-collapsed reaction.

H+ (cpd00067) is dropped: its transformed formation energy is zero at fixed
pH (dGbyG returns zeros for [H+]), and dGbyG's normalisation moves every other
species to a different protonation state from ModelSEED's pH-7 form (neutral
where its uncharger succeeds, a neutral zwitterion where explicit H blocks it),
so ModelSEED's H+ count no longer balances anything dGbyG sees. The cost: with
ignore_H=True, dropping H+ also hides genuine CHARGE imbalance. 1,312 live
'ok' reactions would fail dGbyG's check with H+ kept, and 1,310 of them carry
ModelSEED status 'CI' (charge-imbalanced). eQuilibrator, dGPredictor and GC
score them too (901 / 1,307 / 1,137), so they stay in the like-for-like
population; compare.py counts them.

Balance is checked with dGbyG's own rule (Reaction.is_balanced: ignore_H=True,
ignore_H_ion=True, ignore_charge=False, ignore_H2O=False) on dGbyG's
normalised structures (residuals under 1e-9 count as balanced; dGbyG demands
exactly 0, which differs only for rxn10947, a wildcard reaction with fractional
coefficients). An unbalanced reaction gets no estimate, which is also what
dGbyG itself returns: its Reaction.balance() is meant to add water for an
oxygen deficit but never does (src/dGbyG/api.py:368 tests is_balanced on the
not-yet-updated reaction, so the original is always returned, and line 343
looks for '[H]O[H]' where .Smiles gives 'O'), so standard_dGr_prime is NaN.
Reactions that WOULD balance with water added are reported as
'unbalanced_water', with that counterfactual value in alt_dG_kcal -- our
construction, not dGbyG output.

Two further populations are scored but kept OUT of the primary estimate,
because neither eQuilibrator nor dGPredictor reports them (0 of each, checked
against the shipped sublists) and a like-for-like comparison needs the same
population:
  * 'wildcard' -- a participant's SMILES carries an R group ('*'). dGbyG
    featurises the dummy atom (atomic number 0) and returns a number, but it is
    an extrapolation outside anything the model was trained on. Checked before
    balance, so every R-group reaction lands here; only balanced ones carry an
    alt value.
  * 'null_net' -- every participant cancels across compartments (pure
    transport), so the compartment-collapsed reaction is the null reaction.
Their values sit in alt_dG_kcal / alt_sd_kcal with alt_kind naming why.

Direction uses the paper's rule set for eQuilibrator and dGPredictor
(reversibility_heuristics.make_ri_heuristics, z = 1: irreversible when
|ln Gamma| - sigma_lnGamma > ln 1000; ATP synthase forced '='), selected through
heuristics_for_source('dGPredictor'). A point-estimate variant (z = 0) is
reported alongside as a sensitivity column.

Outputs (results/, or results/repaired/ for --variant repaired, which reads
the repaired-structure compound predictions of predict_compounds.py):
  reaction_predictions.tsv   one row per reaction record (live and obsolete)

Run:
  ~/Documents/py_venv/bin/python analysis/modelseed/predict_reactions.py
"""
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MSDB = Path(os.environ.get("MSDB_ROOT", REPO.parent / "ModelSEEDDatabase"))
OUT = HERE / "results"
KCAL = 4.184
SENTINEL = 1e7
PROTON, WATER = "cpd00067", "cpd00001"

sys.path.insert(0, str(MSDB / "Scripts/Thermodynamics"))
from reversibility_heuristics import (  # noqa: E402
    explicit_energy, heuristics_for_source, make_ri_heuristics, run_reversibility,
    EQ_UNDECOMPOSABLE_SIGMA,
)

RULES = heuristics_for_source("dGPredictor")        # the paper's RI cascade, z = 1
RULES_Z0 = make_ri_heuristics(z=0.0, sigma_gate=EQ_UNDECOMPOSABLE_SIGMA)


def is_balanced_bags(net, bags):
    """dGbyG's reaction_utils.is_balanced on atom bags, with Reaction's flags.

    Returns (balanced, diff) where diff is the raw atom difference (before the
    ignore_* adjustments), keyed by element plus 'charge'."""
    diff = defaultdict(float)
    for cpd, coeff in net.items():
        for atom, num in bags[cpd].items():
            diff[atom] += num * coeff
    raw = dict(diff)
    # ignore_H_ion=True
    if diff.get("charge", 0) * diff.get("H", 0) <= 0:
        h_ion = 0
    elif diff["charge"] < 0:
        h_ion = -max(diff["charge"], diff["H"])
    else:
        h_ion = -min(diff["charge"], diff["H"])
    diff["charge"] = diff.get("charge", 0) + h_ion
    diff["H"] = diff.get("H", 0) + h_ion
    # ignore_H=True
    diff["H"] = 0
    return not any(abs(v) > 1e-9 for v in diff.values()), raw


def src_triple(thermo, label):
    v = (thermo or {}).get(label)
    if not v or v[0] in ("", None):
        return "", "", ""
    op = v[2] if len(v) > 2 else ""
    return v[0], v[1], op


def main():
    import argparse
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("native", "repaired"), default="native")
    args = ap.parse_args()
    global OUT
    if args.variant == "repaired":
        OUT = OUT / "repaired"
    cp = pd.read_csv(OUT / "compound_predictions.tsv", sep="\t", keep_default_na=False,
                     dtype={"status": str, "atom_bag_dgbyg": str, "smiles_dgbyg": str})
    heads_all = np.load(OUT / "compound_heads.npy")
    row = dict(zip(cp["id"], cp["row"]))
    status = dict(zip(cp["id"], cp["status"]))
    bags = {c: json.loads(b) for c, b in zip(cp["id"], cp["atom_bag_dgbyg"]) if b}
    wildcard = {c for c, s in zip(cp["id"], cp["smiles_modelseed"]) if "*" in s}
    water_heads = heads_all[row[WATER]].astype(np.float64)

    rows, tally = [], Counter()
    for shard in sorted(glob.glob(str(MSDB / "Biochemistry/reaction_[0-9][0-9].json"))):
        for rxn in json.load(open(shard)):
            stoich = rxn.get("stoichiometry") or []
            net = defaultdict(float)
            for s in stoich:
                net[s["compound"]] += float(s["coefficient"])
            n_proton = net.get(PROTON, 0.0)
            net = {c: v for c, v in net.items() if abs(v) > 1e-12 and c != PROTON}

            missing = sorted(c for c in net if status.get(c) != "ok")
            rec = dict(rxn=rxn["id"], name=rxn.get("name", ""), is_obsolete=rxn.get("is_obsolete", 0),
                       is_transport=rxn.get("is_transport", 0), rxn_status=rxn.get("status", ""),
                       n_net=len(net), n_proton=f"{n_proton:g}", n_missing=len(missing), missing_cpds=";".join(missing),
                       has_wildcard=int(any(c in wildcard for c in net)),
                       raw_diff="", status="", dG_kcal="", sd_kcal="", op="", op_rule="", op_z0="",
                       alt_kind="", alt_dG_kcal="", alt_sd_kcal="")

            if not stoich:
                rec["status"] = "no_stoichiometry"
            elif not net:
                # every participant cancels across compartments (pure transport):
                # the compartment-collapsed reaction is the null reaction
                rec["status"] = "null_net"
            elif missing:
                rec["status"] = "missing_structure" if any(status.get(c) == "no_smiles" or c not in status
                                                           for c in missing) else "compound_failed"
            else:
                ok, raw = is_balanced_bags(net, bags)
                rec["raw_diff"] = json.dumps({k: v for k, v in raw.items() if abs(v) > 1e-9}, sort_keys=True)
                vec = np.zeros(100, dtype=np.float64)
                for c, v in net.items():
                    vec += v * heads_all[row[c]].astype(np.float64)
                if rec["has_wildcard"]:
                    rec["status"] = "wildcard"
                    if ok:
                        rec["alt_kind"] = "wildcard"
                        rec["alt_dG_kcal"] = f"{vec.mean() / KCAL:.4f}"
                        rec["alt_sd_kcal"] = f"{vec.std() / KCAL:.4f}"
                elif ok:
                    rec["status"] = "ok"
                else:
                    # what dGbyG's Reaction.balance() is meant to do (but does not):
                    # add water for the O deficit
                    n_o = raw.get("O", 0)
                    fixed = dict(net)
                    if n_o:
                        fixed[WATER] = fixed.get(WATER, 0.0) - n_o
                        fixed = {c: v for c, v in fixed.items() if abs(v) > 1e-12}
                    ok2 = bool(n_o) and is_balanced_bags(fixed, bags)[0]
                    if ok2:
                        rec["status"] = "unbalanced_water"
                        vw = vec - n_o * water_heads
                        rec["alt_kind"] = "water_added"
                        rec["alt_dG_kcal"] = f"{vw.mean() / KCAL:.4f}"
                        rec["alt_sd_kcal"] = f"{vw.std() / KCAL:.4f}"
                    else:
                        rec["status"] = "unbalanced"
                if rec["status"] == "ok":
                    # rounded to 2 dp in kcal before the direction call, exactly as
                    # the database stores every per-source triple
                    # (Update_Reaction_dGPredictor_Energies.py)
                    dg, sd = round(vec.mean() / KCAL, 2), round(vec.std() / KCAL, 2)
                    rec["dG_kcal"], rec["sd_kcal"] = f"{dg:.2f}", f"{sd:.2f}"
                    st, op, _ = run_reversibility(rxn, explicit_energy(dg, sd), RULES)
                    rec["op"], rec["op_rule"] = op, st
                    rec["op_z0"] = run_reversibility(rxn, explicit_energy(dg, sd), RULES_Z0)[1]
            if rec["status"] == "null_net":
                rec["alt_kind"], rec["alt_dG_kcal"], rec["alt_sd_kcal"] = "null_net", "0.0000", "0.0000"

            t = rxn.get("thermodynamics") or {}
            for lab, key in (("eQuilibrator", "eq"), ("dGPredictor", "dgp"), ("Group contribution", "gc")):
                dg, err, op = src_triple(t, lab)
                rec[f"{key}_dG_kcal"], rec[f"{key}_sd_kcal"], rec[f"{key}_op"] = dg, err, op
            llm = t.get("LLMs") or []
            rec["llm_op"] = llm[2] if len(llm) > 2 else ""   # direction-only triple: ['', '', op]
            ev = rxn.get("thermo-evidence") or {}
            rec["grade"], rec["grade_source"], rec["grade_assessment"] = ev.get("grade", ""), ev.get("source", ""), ev.get("assessment", "")
            rec["reversibility"] = rxn.get("reversibility", "")
            rec["equation"] = rxn.get("equation", "")
            rows.append(rec)
            tally[(rec["status"], rec["is_obsolete"])] += 1

    cols = list(rows[0].keys())
    with open(OUT / "reaction_predictions.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows):,} reaction records")
    for (st, obs), n in sorted(tally.items()):
        print(f"  {st:20s} obsolete={obs}  {n:,}")


if __name__ == "__main__":
    main()
