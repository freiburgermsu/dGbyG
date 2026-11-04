#!/usr/bin/env python
"""Compact, database-shaped dGbyG tables for ModelSEED.

The ModelSEED `thermodynamics` dict stores per-source triples
[dG kcal/mol, sigma kcal/mol, operator]. These files carry exactly that for
dGbyG -- primary estimate and the structure-repaired variant -- so the values
can be ingested later without re-running the pipeline. Nothing is written into
ModelSEEDDatabase; rows exist only where Stage 2 produced an estimate (no
sentinels, following Update_Reaction_dGPredictor_Energies.py: a declined
prediction is simply absent).

Outputs (results/):
  modelseed_dgbyg_reactions.tsv   rxn, is_obsolete, dG/sd/op (primary), dG/sd/op (repaired)
  modelseed_dgbyg_compounds.tsv   cpd, dGf/sd (kcal/mol; dGbyG's transformed formation energy)

Operators come from the paper's eQuilibrator/dGPredictor rule set. If these are
ever loaded as thermodynamics['dGbyG'], register the label in
reversibility_heuristics.SOURCE_HEURISTIC_SET as 'DGP' first: an unknown label
falls back to the group-contribution rules, and Add_Reaction_Thermodynamics_
Operators.py would silently rewrite every dGbyG operator with them.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def main():
    cols = ["rxn", "is_obsolete", "status", "dG_kcal", "sd_kcal", "op"]
    a = pd.read_csv(RES / "reaction_predictions.tsv", sep="\t", keep_default_na=False, dtype=str, usecols=cols)
    b = pd.read_csv(RES / "repaired" / "reaction_predictions.tsv", sep="\t", keep_default_na=False, dtype=str,
                    usecols=["rxn", "status", "dG_kcal", "sd_kcal", "op"])
    a = a[a.status == "ok"].drop(columns="status")
    b = b[b.status == "ok"].drop(columns="status").rename(
        columns={"dG_kcal": "repaired_dG_kcal", "sd_kcal": "repaired_sd_kcal", "op": "repaired_op"})
    m = a.merge(b, on="rxn", how="outer")
    meta = pd.read_csv(RES / "reaction_predictions.tsv", sep="\t", keep_default_na=False, dtype=str,
                       usecols=["rxn", "is_obsolete"]).set_index("rxn")["is_obsolete"]
    m["is_obsolete"] = m["rxn"].map(meta)
    m = m.sort_values("rxn")
    m.to_csv(RES / "modelseed_dgbyg_reactions.tsv", sep="\t", index=False)
    c = pd.read_csv(RES / "compound_predictions.tsv", sep="\t", keep_default_na=False, dtype=str,
                    usecols=["id", "is_obsolete", "status", "dgbyg_dGf_kcal", "dgbyg_sd_kcal"])
    c = c[c.status == "ok"].drop(columns="status").rename(columns={"id": "cpd"})
    c.to_csv(RES / "modelseed_dgbyg_compounds.tsv", sep="\t", index=False)
    print(f"{len(m):,} reactions ({(m.dG_kcal.fillna('') != '').sum():,} primary, "
          f"{(m.repaired_dG_kcal.fillna('') != '').sum():,} repaired); {len(c):,} compounds")


if __name__ == "__main__":
    main()
