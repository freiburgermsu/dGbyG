#!/usr/bin/env python
"""Stage 3b: held-out dGbyG predictions from the grouped K-fold ensembles.

Two products:

  results/cv/training_heldout.tsv
      every dGbyG training row predicted by the ensemble that withheld it --
      dGbyG's own cross-validated error, the analogue of the paper's 10-fold
      eQuilibrator/dGPredictor calibration over each method's training set
      (calibrate_sigma.py).

  results/cv/anchor_heldout.tsv
      every one of the 363 live openTECR anchors predicted by the ensemble of
      the fold that WITHHELD its matching dGbyG training reaction.

Matching an anchor to its training reaction. All 363 anchors are dGbyG
training reactions, but only when the reaction is built from dGbyG's own KEGG
structures (structure_sensitivity.py: 363/363 exact on the KEGG key, only
100/363 on the ModelSEED key). ModelSEED structures differ from the training
forms (charged pH-7 species that survive dGbyG's uncharger, amides drawn as
imidic acids), so a ModelSEED-SMILES key misses most anchors -- and a missed
anchor would be predicted by a fold that still contains it. The match
therefore uses the anchor's openTECR KEGG equation (structure_sensitivity's
anchors_kegg.tsv: the equation used and its orientation against the ModelSEED
reaction, settled by species pairing) and cv_train.reaction_key over
dGbyG-normalised KEGG SMILES, coefficients scaled by their smallest magnitude.
An anchor with no match is reported with in_training='no' and predicted by a
fold fixed by a hash of its id.

Three structure sets are scored with that one held-out ensemble:
  ms        the ModelSEED reaction on ModelSEED structures -- what the database
            application actually computes (predict_reactions.py's reaction:
            netted across compartments, H+ dropped)
  kegg      the openTECR KEGG equation on dGbyG's native KEGG structures,
            oriented to the ModelSEED reaction
  ms_fixed  ModelSEED structures repaired as structure_sensitivity.py's
            'fix_amide' variant: uncharged without explicit H, then non-aromatic
            imidic acids restored to amides
Sigma is the std of the summed per-head vectors, as everywhere else.

Run:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=src ~/Documents/py_venv/bin/python -B \
      analysis/modelseed/cv_predict.py
"""
import argparse
import csv
import os
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cv_train as cvt  # noqa: E402

REPO = HERE.parents[1]
MSDB = Path(os.environ.get("MSDB_ROOT", REPO.parent / "ModelSEEDDatabase"))
RES = HERE / "results"
CV = RES / "cv"
KCAL = 4.184
PROTON = "cpd00067"


def scaled_key(stoich, strip_stereo=False):
    st = {}
    for s, c in stoich.items():
        if s in cvt.IGNORE:
            continue
        if strip_stereo:
            s = s.replace("@", "").replace("/", "").replace("\\", "")
        st[s] = st.get(s, 0.0) + c
    st = {s: c for s, c in st.items() if abs(c) > 1e-12}
    if not st:
        return None
    m = min(abs(c) for c in st.values())
    return cvt.reaction_key({s: c / m for s, c in st.items()})


def load_fold_models(k):
    from dGbyG.model.inference import Inference_Model
    return {f: Inference_Model(str(CV / f"mpnn_A139_B23_E{cvt.EMB}_L{cvt.LAYERS}_fold{f}"), device="cpu")
            for f in range(k)}


class Heads:
    """Per-head dGf'° of a dGbyG Compound under a given ensemble, cached by the
    normalised SMILES (the model input)."""

    def __init__(self):
        self.cache = {}

    def __call__(self, model, comp):
        key = (id(model), comp.Smiles)
        if key not in self.cache:
            if comp.Smiles == "[H+]":
                self.cache[key] = np.zeros(model.num_head)
            else:
                self.cache[key] = model(comp.graph_data).squeeze().numpy().astype(np.float64)
        return self.cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    cvt._setup(threads=args.threads)
    import pandas as pd
    from rdkit import Chem
    import dGbyG.api as api
    from dGbyG.utils.reaction_utils import parse_equation
    from structure_repair import imidic_to_amide
    from structure_sensitivity import parse_kegg

    folds_tbl = list(csv.DictReader(open(CV / "folds.tsv"), delimiter="\t"))
    k = max(int(r["fold"]) for r in folds_tbl) + 1
    models = load_fold_models(k)
    heads = Heads()
    comp_cache = {}

    def compound(kind, ident):
        if (kind, ident) not in comp_cache:
            if kind == "smiles":
                comp_cache[(kind, ident)] = api.Compound(ident, "smiles")
            elif kind == "kegg":
                comp_cache[(kind, ident)] = api.Compound(ident, "kegg")
            elif kind == "fixed":
                raw = Chem.MolFromSmiles(ident)
                fix = api.Compound(Chem.RemoveHs(raw))          # uncharging sees implicit H
                comp_cache[(kind, ident)] = api.Compound(imidic_to_amide(fix.mol))
        return comp_cache[(kind, ident)]

    # ---- (a) dGbyG's own training rows, each by the fold that withheld it
    df, eq, dg, scale, weight = cvt.load_training()
    with open(CV / "training_heldout.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["row", "type", "fold", "description", "measured_kJ", "scale_kJ",
                    "pred_kJ", "pred_sd_kJ", "measured_kcal", "pred_kcal", "pred_sd_kcal"])
        for i, r in enumerate(folds_tbl):
            f = int(r["fold"])
            vec = np.zeros(models[f].num_head)
            for s, c in parse_equation(eq[i]).items():
                vec += float(c) * heads(models[f], compound("smiles", s))
            w.writerow([i, r["type"], f, df["description"][i], f"{dg[i]:.4f}", f"{scale[i]:.4f}",
                        f"{vec.mean():.4f}", f"{vec.std():.4f}", f"{dg[i] / KCAL:.4f}",
                        f"{vec.mean() / KCAL:.4f}", f"{vec.std() / KCAL:.4f}"])
    print("training rows predicted held-out", flush=True)

    # ---- (b) anchors: fold from the KEGG-structure key of the matching training reaction
    exact, skel = {}, {}
    for r in folds_tbl:
        if r["type"] == "formation dg":
            continue
        st = defaultdict(float)
        for s, c in parse_equation(eq[int(r["row"])]).items():
            st[cvt.norm_smiles(s)] += float(c)
        exact.setdefault(scaled_key(st), set()).add(int(r["fold"]))
        skel.setdefault(scaled_key(st, strip_stereo=True), set()).add(int(r["fold"]))
    # grouped folds keep every copy of a reaction together, so each key has one fold
    assert all(len(v) == 1 for v in exact.values()), "a training key spans folds"

    ak = pd.read_csv(RES / "structure_sensitivity" / "anchors_kegg.tsv", sep="\t", keep_default_na=False)
    cp = pd.read_csv(RES / "compound_predictions.tsv", sep="\t", keep_default_na=False)
    smi_in = dict(zip(cp["id"], cp["smiles_modelseed"]))
    import json, glob
    stoich = {}
    want = set(ak["rxn"])
    for shard in sorted(glob.glob(str(MSDB / "Biochemistry/reaction_[0-9][0-9].json"))):
        for rxn in json.load(open(shard)):
            if rxn["id"] in want:
                net = defaultdict(float)
                for s in rxn.get("stoichiometry") or []:
                    net[s["compound"]] += float(s["coefficient"])
                stoich[rxn["id"]] = {c: v for c, v in net.items() if abs(v) > 1e-12 and c != PROTON}

    with open(CV / "anchor_heldout.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["rxn", "in_training", "fold",
                    "heldout_dG_kcal", "heldout_sd_kcal",              # ModelSEED structures
                    "heldout_kegg_dG_kcal", "heldout_kegg_sd_kcal",    # native KEGG structures
                    "heldout_fixed_dG_kcal", "heldout_fixed_sd_kcal"])  # repaired ModelSEED structures
        for _, a in ak.iterrows():
            kd, _ = parse_kegg(a["kegg_eq_used"])
            st = defaultdict(float)
            for cid, v in kd.items():
                st[compound("kegg", cid).Smiles] += v
            key, skey = scaled_key(st), scaled_key(st, strip_stereo=True)
            if key in exact:
                tag, f = "exact", next(iter(exact[key]))
            elif skey in skel:
                tag, f = "skeleton", min(skel[skey])
            else:
                tag, f = "no", zlib.crc32(a["rxn"].encode()) % k
            m = models[f]
            sign = -1.0 if a["orientation"] == "reversed" else 1.0
            v_kegg = sign * sum(v * heads(m, compound("kegg", cid)) for cid, v in kd.items())
            net = stoich[a["rxn"]]
            v_ms = sum(v * heads(m, compound("smiles", smi_in[c])) for c, v in net.items())
            v_fix = sum(v * heads(m, compound("fixed", smi_in[c])) for c, v in net.items())
            w.writerow([a["rxn"], tag, f,
                        f"{v_ms.mean() / KCAL:.4f}", f"{v_ms.std() / KCAL:.4f}",
                        f"{v_kegg.mean() / KCAL:.4f}", f"{v_kegg.std() / KCAL:.4f}",
                        f"{v_fix.mean() / KCAL:.4f}", f"{v_fix.std() / KCAL:.4f}"])
    print("anchors predicted held-out", flush=True)


if __name__ == "__main__":
    main()
