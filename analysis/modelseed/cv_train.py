#!/usr/bin/env python
"""Stage 3a: grouped K-fold retraining of dGbyG, for held-out predictions.

The shipped dGbyG ensemble is fitted on data/TrainingData.csv (449 TECRDB
reactions + 222 formation energies, all at pH 7, I 0.25, pMg 14). The ModelSEED
openTECR anchors overlap that set, so scoring the shipped model against them is
in-sample -- the same leak the paper avoids for eQuilibrator and dGPredictor by
collecting "one held-out prediction per anchored reaction".

This retrains the ensemble K times, each time withholding one fold of training
rows. Folds are GROUPED by a canonical reaction key (dGbyG-normalised SMILES
with stereochemistry removed, orientation-invariant, coefficients scaled by
their smallest magnitude, water and H+ ignored), so a reaction and its
duplicate, reversed, rescaled or STEREOISOMERIC copy are always withheld
together. Stereo must be stripped because dGbyG's graph features carry no
chirality (model/datasets.py mol_to_graph_data featurises atomic number,
hybridisation, aromaticity and charge only): D-glucose and D-mannose, or
fumarate and maleate, are the same input to it, so a stereo twin left in the
training folds would leak its target into the held-out prediction.

The recipe is examples/train_the_model.ipynb's, with the shipped v2
architecture (emb 300, 2 layers -- the folder name mpnn_A139_B23_E300_L2_v2):
targets perturbed by N(0, scale) per head, weight = 1/scale/median(scale),
9000 full-batch epochs, Adam lr 1e-4, weight decay 1e-6, keep the best epoch.

Outputs (results/cv/):
  folds.tsv                              training row -> group key, fold
  mpnn_A139_B23_E300_L2_fold{k}/{h}.pt   head h of the fold-k ensemble

Run (GPU, ~2 min per head; several trainings share the card):
  PYTHONPATH=src ~/Documents/py_venv/bin/python -B \
      analysis/modelseed/cv_train.py --folds 10 --heads 20 --workers 6 --device cuda
(--device cpu trains on CPU at ~0.9 s per epoch per thread, ~2.3 h per head.)
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "results" / "cv"
SEED = 20261009
EMB, LAYERS = 300, 2
EPOCHS, LR, WD = 9000, 1e-4, 1e-6
IGNORE = {"O", "[H+]"}          # water and proton, after dGbyG normalisation


def _setup(threads=1, device="cpu"):
    warnings.filterwarnings("ignore")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""   # dGbyG's Trainer picks CUDA whenever it is visible
    import torch
    torch.set_num_threads(threads)
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    sys.path.insert(0, str(REPO / "src"))


def load_training():
    import pandas as pd
    df = pd.read_csv(REPO / "data/TrainingData.csv")
    eq = df["reaction"].to_numpy()
    dg = df["standard_dg_prime"].to_numpy(dtype=float)
    mean_std = df["std"].mean()
    scale = []
    for n, sem in zip(df["n"], df["SEM"]):
        scale.append(mean_std if np.isnan(sem) else (sem ** 2 + mean_std ** 2 / n) ** 0.5)
    scale = np.asarray(scale)
    weight = 1 / scale / np.median(scale)
    return df, eq, dg, scale, weight


def norm_smiles(smi, cache={}):
    if smi not in cache:
        from rdkit import Chem
        from dGbyG.utils import to_mol
        from dGbyG.utils.mol_utils import normalize_mol
        m = to_mol(smi, cid_type="smiles")
        cache[smi] = Chem.MolToSmiles(Chem.RemoveHs(normalize_mol(m)), canonical=True) if m is not None else smi
    return cache[smi]


def flat_smiles(smi, cache={}):
    """norm_smiles with stereochemistry removed and re-canonicalised: the
    molecule as dGbyG's stereo-blind featurisation sees it."""
    if smi not in cache:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smi)
        if m is None:
            cache[smi] = smi.replace("@", "").replace("/", "").replace("\\", "")
        else:
            Chem.RemoveStereochemistry(m)
            cache[smi] = Chem.MolToSmiles(m, canonical=True)
    return cache[smi]


def group_key(stoich):
    """Orientation-, scale- and stereo-invariant key, water/H+ ignored."""
    st = {}
    for s, c in stoich.items():
        if s in IGNORE:
            continue
        f = flat_smiles(s)
        st[f] = st.get(f, 0.0) + c
    st = {s: c for s, c in st.items() if abs(c) > 1e-12}
    if not st:
        return "empty"
    m = min(abs(c) for c in st.values())
    return reaction_key({s: c / m for s, c in st.items()})


def reaction_key(stoich):
    """Orientation-invariant key for {normalised smiles: coeff}, water/H+ ignored."""
    items = tuple(sorted((s, round(float(c), 6)) for s, c in stoich.items()
                         if s not in IGNORE and abs(c) > 1e-12))
    neg = tuple(sorted((s, -c) for s, c in items))
    return min(repr(items), repr(neg))


def training_keys(df, eq):
    from dGbyG.utils.reaction_utils import parse_equation
    keys = []
    for kind, e in zip(df["type"], eq):
        st = {}
        for s, c in parse_equation(e).items():
            n = norm_smiles(s)
            st[n] = st.get(n, 0.0) + float(c)
        k = group_key(st)
        if k == "empty":
            # pure stereo interconversion: graph-identical sides, so dGbyG predicts
            # exactly 0 and the row carries no gradient -- no twin to leak; give it
            # its own group so these rows do not pile into one fold
            k = f"empty:{len(keys)}"
        keys.append(("F:" if kind == "formation dg" else "R:") + k)
    return keys


def assign_folds(keys, k):
    rng = np.random.default_rng(SEED)
    groups = sorted(set(keys))
    order = rng.permutation(len(groups))
    fold_of_group = {groups[g]: i % k for i, g in enumerate(order)}
    return np.asarray([fold_of_group[x] for x in keys])


def _train_one(task):
    fold, head, threads, device = task
    _setup(threads, device)
    import torch
    from dGbyG.model.datasets import TrainDataset
    from dGbyG.model import MPNN_model, Trainer
    dest = OUT / f"mpnn_A139_B23_E{EMB}_L{LAYERS}_fold{fold}" / f"{head}.pt"
    if dest.exists():
        return fold, head, "cached"
    df, eq, dg, scale, weight = load_training()
    folds = np.asarray([int(r["fold"]) for r in csv.DictReader(open(OUT / "folds.tsv"), delimiter="\t")])
    train = np.nonzero(folds != fold)[0]
    rng = np.random.default_rng(SEED + 1000 * fold + head)
    torch.manual_seed(SEED + 1000 * fold + head)
    noisy = dg[train] + rng.standard_normal(len(train)) * scale[train]
    ds = TrainDataset(equations=eq[train], dGs=noisy, weights=weight[train])
    net = MPNN_model(atom_dim=ds[0].x.size(1), bond_dim=ds[0].edge_attr.size(1), emb_dim=EMB, num_layer=LAYERS)
    tr = Trainer()
    tr.network = net
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        loss, _, best = tr.train(ds, epochs=EPOCHS, lr=LR, weight_decay=WD)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    torch.save(tr.network.state_dict(), tmp)
    tmp.rename(dest)
    return fold, head, f"best epoch {best}, loss {float(loss[best]):.3f}"


def main():
    global EPOCHS
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--heads", type=int, default=20)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="debug override")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = ap.parse_args()
    EPOCHS = args.epochs
    _setup(device=args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    df, eq, dg, scale, weight = load_training()
    keys = training_keys(df, eq)
    folds = assign_folds(keys, args.folds)
    with open(OUT / "folds.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["row", "key", "fold", "type"])
        for i, (k, f, t) in enumerate(zip(keys, folds, df["type"])):
            w.writerow([i, k, f, t])
    print(f"{len(keys)} training rows, {len(set(keys))} groups, folds sized "
          f"{np.bincount(folds).tolist()}", flush=True)

    tasks = [(f, h, args.threads, args.device) for h in range(args.heads) for f in range(args.folds)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_set_epochs, initargs=(EPOCHS,)) as pool:
        for n, (f, h, msg) in enumerate(pool.imap_unordered(_train_one, tasks), 1):
            print(f"[{n}/{len(tasks)}] fold {f} head {h}: {msg}", flush=True)


def _set_epochs(e):
    global EPOCHS
    EPOCHS = e


if __name__ == "__main__":
    main()
