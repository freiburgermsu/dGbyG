#!/usr/bin/env python
"""Stage 1: dGbyG standard transformed formation energies for every ModelSEED compound.

Every compound record in ModelSEEDDatabase/Biochemistry/compound_*.json that
carries a SMILES is run through the shipped dGbyG ensemble
(models/mpnn_A139_B23_E300_L2_v2, 100 heads). Obsolete compounds are included
because live reactions can still reference them.

Condition: dGbyG's native training condition -- pH 7.0, I 0.25 M, pMg 14,
298.15 K. dGbyG's Legendre transform only moves Mg through explicit Mg atoms
(it has no Mg-binding constants), so re-transforming to the database's pMg 3.0
would not model ATP-Mg binding anyway; the paper's own Supplementary S2 puts
the pMg 14 -> 3 shift at a median 0.044 kcal/mol per reaction.

With --variant repaired, every SMILES first goes through structure_repair.py
(uncharged without explicit H, imidic acids restored to amides -- the forms
dGbyG was trained on) and outputs go to results/repaired/. That variant is a
labelled sensitivity analysis, never the primary estimate.

Outputs (results/, or results/repaired/ for --variant repaired):
  compound_heads.npy         float32 [n_compounds, 100], kJ/mol, NaN where no prediction
  compound_predictions.tsv   one row per compound record, row i <-> heads[i]

Run:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=src ~/Documents/py_venv/bin/python \
      analysis/modelseed/predict_compounds.py --workers 30
"""
import argparse
import csv
import glob
import json
import multiprocessing as mp
import os
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MSDB = Path(os.environ.get("MSDB_ROOT", REPO.parent / "ModelSEEDDatabase"))
OUT = HERE / "results"
KCAL = 4.184
SENTINEL = 1e7

_worker = {}


def _init_worker(threads, variant):
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(threads)
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    sys.path.insert(0, str(REPO / "src"))
    import dGbyG.api as api
    from dGbyG.model.inference import Inference_Model
    api.model_cache[api.infer_model_path] = Inference_Model(api.infer_model_path, device="cpu")
    _worker["api"] = api
    _worker["variant"] = variant
    if variant == "repaired":
        sys.path.insert(0, str(HERE))
        from structure_repair import repaired_compound
        _worker["repair"] = repaired_compound


def _predict(item):
    """(index, smiles) -> (index, status, dgbyg_smiles, atom_bag_json, heads or None)."""
    idx, smiles = item
    api = _worker["api"]
    try:
        if _worker["variant"] == "repaired":
            comp = _worker["repair"](api, smiles)
            if comp is None:
                return idx, "parse_fail", "", "", None
        else:
            comp = api.Compound(smiles, "smiles")
    except Exception as e:  # rdkit/dGbyG refused the input outright
        return idx, f"error:{type(e).__name__}", "", "", None
    if comp.mol is None:
        return idx, "parse_fail", "", "", None
    try:
        norm = comp.Smiles
        bag = json.dumps(comp.atom_bag, sort_keys=True)
        heads = np.asarray(comp.standard_dGf_prime_list, dtype=np.float32)
    except Exception as e:  # featurisation or inference failed (e.g. unsupported atom)
        return idx, f"error:{type(e).__name__}", "", "", None
    if heads.shape != (100,) or not np.isfinite(heads).all():
        return idx, "nonfinite", norm, bag, None
    return idx, "ok", norm, bag, heads


def _src(thermo, label):
    v = (thermo or {}).get(label)
    if not v or v[0] in ("", None):
        return "", ""
    dg, err = float(v[0]), float(v[1])
    if abs(dg) >= SENTINEL or abs(err) >= SENTINEL:
        return "", ""
    return dg, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N compounds only")
    ap.add_argument("--variant", choices=("native", "repaired"), default="native")
    args = ap.parse_args()
    global OUT
    if args.variant == "repaired":
        OUT = OUT / "repaired"
    OUT.mkdir(parents=True, exist_ok=True)

    records = []
    for shard in sorted(glob.glob(str(MSDB / "Biochemistry/compound_[0-9][0-9].json"))):
        records.extend(json.load(open(shard)))
    if args.limit:
        records = records[: args.limit]
    print(f"{len(records):,} compound records from {MSDB}", flush=True)

    work = [(i, r["smiles"]) for i, r in enumerate(records) if r.get("smiles")]
    print(f"{len(work):,} carry a SMILES", flush=True)

    heads = np.full((len(records), 100), np.nan, dtype=np.float32)
    status = ["no_smiles"] * len(records)
    norm = [""] * len(records)
    bags = [""] * len(records)
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.threads, args.variant)) as pool:
        for n, (i, st, ns, bag, h) in enumerate(pool.imap_unordered(_predict, work, chunksize=32), 1):
            status[i], norm[i], bags[i] = st, ns, bag
            if h is not None:
                heads[i] = h
            if n % 2000 == 0:
                print(f"  {n:,}/{len(work):,}", flush=True)

    np.save(OUT / "compound_heads.npy", heads)
    cols = ["row", "id", "name", "formula", "charge", "is_obsolete", "status",
            "smiles_modelseed", "smiles_dgbyg", "atom_bag_dgbyg",
            "dgbyg_dGf_kJ", "dgbyg_sd_kJ", "dgbyg_dGf_kcal", "dgbyg_sd_kcal",
            "eq_dGf_kcal", "eq_sd_kcal", "dgp_dGf_kcal", "dgp_sd_kcal",
            "gc_dGf_kcal", "gc_sd_kcal"]
    with open(OUT / "compound_predictions.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(cols)
        for i, r in enumerate(records):
            t = r.get("thermodynamics")
            eq, dgp, gc = _src(t, "eQuilibrator"), _src(t, "dGPredictor"), _src(t, "Group contribution")
            if status[i] == "ok":
                m, s = float(np.mean(heads[i])), float(np.std(heads[i]))
                vals = [f"{m:.4f}", f"{s:.4f}", f"{m / KCAL:.4f}", f"{s / KCAL:.4f}"]
            else:
                vals = ["", "", "", ""]
            w.writerow([i, r["id"], r.get("name", ""), r.get("formula", ""), r.get("charge", ""),
                        r.get("is_obsolete", 0), status[i], r.get("smiles") or "", norm[i], bags[i],
                        *vals, *eq, *dgp, *gc])

    from collections import Counter
    print(Counter(s if not s.startswith("error") else s for s in status).most_common(), flush=True)


if __name__ == "__main__":
    main()
