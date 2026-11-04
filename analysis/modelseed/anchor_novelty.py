#!/usr/bin/env python
"""How novel is each held-out anchor to the fold model that predicts it?

cv_predict.py withholds an anchor's own training rows (and its stereo twins),
not its reaction CLASS or its compounds. dGbyG is a 2-layer message-passing
network with sum pooling, so its reaction value depends only on how the
reaction changes the multiset of 2-bond atom environments, as its features see
them (atomic number, hybridisation, aromaticity, formal charge; bond type).
Two different reactions that make the same change -- e.g. AMP + H2O ->
adenosine + Pi and fructose-1,6-bisphosphate + H2O -> fructose 1-phosphate +
Pi -- get identical per-head predictions. So, per anchor:

  wl2_zero           the change is empty: dGbyG predicts exactly 0 by
                     construction (pure stereo interconversions, and exchanges
                     such as ATP + GDP = ADP + GTP)
  wl2_twin_in_train  training reaction rows, still in the anchor's training
                     folds, with the same 2-bond change (orientation- and
                     scale-free): the model has effectively seen the anchor
  all_form_in_train  every participant (water and H+ aside) has its formation
                     energy among the training folds' formation rows
  target_kcal        dGbyG's own training target for the same reaction --
                     TECRDB transformed to pH 7, I 0.25, pMg 14 -- oriented to
                     the ModelSEED reaction (the anchors themselves are medians
                     of raw measurements at their own pH and T)
  ph_all_near_7 / t_all_near_298   every measurement within pH 6.5-7.5 /
                     297-299.5 K (openTECR ranges)

Adapted from the independent leak audit run during verification. Labels are
Weisfeiler-Lehman refinements of dGbyG's own atom features (2 iterations =
the network's 2 message-passing layers).

Output: results/cv/anchor_novelty.tsv
Run:    PYTHONPATH=src ~/Documents/py_venv/bin/python -B analysis/modelseed/anchor_novelty.py
"""
import csv
import hashlib
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cv_train as cvt  # noqa: E402

REPO = HERE.parents[1]
MSDB = Path(os.environ.get("MSDB_ROOT", REPO.parent / "ModelSEEDDatabase"))
RES = HERE / "results"
CV = RES / "cv"
KCAL = 4.184


def feats(a):
    return (a.GetAtomicNum(), int(a.GetHybridization()), int(a.GetIsAromatic()), a.GetFormalCharge())


def wl_labels(mol, iters):
    lab = {a.GetIdx(): repr(feats(a)) for a in mol.GetAtoms()}
    for _ in range(iters):
        new = {}
        for a in mol.GetAtoms():
            nb = sorted((str(b.GetBondType()), lab[b.GetOtherAtomIdx(a.GetIdx())]) for b in a.GetBonds())
            new[a.GetIdx()] = hashlib.md5((lab[a.GetIdx()] + repr(nb)).encode()).hexdigest()
        lab = new
    return Counter(lab.values())


def change_key(st_mols, iters=2):
    """Orientation- and scale-free signature of the reaction's 2-bond change."""
    tot = Counter()
    for m, c in st_mols:
        for lab, n in wl_labels(m, iters).items():
            tot[lab] += c * n
    delta = tuple(sorted((lab, round(v, 6)) for lab, v in tot.items() if abs(v) > 1e-9))
    if not delta:
        return "ZERO"
    m = min(abs(v) for _, v in delta)
    a = tuple((lab, round(v / m, 6)) for lab, v in delta)
    b = tuple((lab, -v) for lab, v in a)
    return min(repr(a), repr(b))


def mol_hash(mol):
    return hashlib.md5(repr(sorted(wl_labels(mol, 8).items())).encode()).hexdigest()


def main():
    cvt._setup(threads=2)
    from rdkit import Chem
    import dGbyG.api as api
    from dGbyG.utils import to_mol
    from dGbyG.utils.mol_utils import normalize_mol
    from dGbyG.utils.reaction_utils import parse_equation
    from cv_predict import scaled_key
    from structure_sensitivity import parse_kegg

    df, eq, dg, scale, weight = cvt.load_training()
    folds = [int(r["fold"]) for r in csv.DictReader(open(CV / "folds.tsv"), delimiter="\t")]
    nm = {}

    def nmol(s):
        if s not in nm:
            m = to_mol(s, "smiles")
            nm[s] = normalize_mol(m) if m is not None else None
        return nm[s]

    rows = []
    for i, (t, e) in enumerate(zip(df["type"], eq)):
        st = [(nmol(s), float(c)) for s, c in parse_equation(e).items()]
        st_s = defaultdict(float)
        for s, c in parse_equation(e).items():
            st_s[cvt.norm_smiles(s)] += float(c)
        rows.append(dict(i=i, type=t, fold=folds[i], change=change_key(st), key=scaled_key(st_s),
                         form_hash=mol_hash(st[0][0]) if t == "formation dg" else None,
                         scale_m=min(abs(c) for s, c in st_s.items() if s not in cvt.IGNORE and abs(c) > 1e-12)
                         if t != "formation dg" else None))
    R = pd.DataFrame(rows)
    rxn_rows = R[R.type != "formation dg"]

    ak = pd.read_csv(RES / "structure_sensitivity" / "anchors_kegg.tsv", sep="\t", keep_default_na=False)
    ho = pd.read_csv(CV / "anchor_heldout.tsv", sep="\t").set_index("rxn")
    ot = pd.read_csv(MSDB / "Biochemistry/Thermodynamics/SourceGrading/opentecr_comparison.csv").set_index("modelseed_rxn")
    kc = {}

    def kcomp(cid):
        if cid not in kc:
            kc[cid] = api.Compound(cid, "kegg")
        return kc[cid]

    out = []
    for _, a in ak.iterrows():
        kd, _ = parse_kegg(a["kegg_eq_used"])
        f = int(ho.loc[a["rxn"], "fold"])
        st_m = [(kcomp(cid).mol, v) for cid, v in kd.items()]
        ck = change_key(st_m)
        st = defaultdict(float)
        for cid, v in kd.items():
            st[kcomp(cid).Smiles] += v
        key = scaled_key(st)
        train = rxn_rows[rxn_rows.fold != f]
        parts = [mol_hash(kcomp(cid).mol) for cid in kd if kcomp(cid).Smiles not in cvt.IGNORE]
        train_form = set(R[(R.type == "formation dg") & (R.fold != f)].form_hash)
        # the anchor's own training rows: their targets, put on the KEGG equation as written
        own = rxn_rows[rxn_rows.key == key]
        tg = []
        for r in own.itertuples():
            st_r = defaultdict(float)
            for s, c in parse_equation(eq[r.i]).items():
                st_r[cvt.norm_smiles(s)] += float(c)
            # orientation and scale of the training row relative to the KEGG equation
            common = [s for s in st if s in st_r and s not in cvt.IGNORE and abs(st[s]) > 1e-12]
            ratio = st[common[0]] / st_r[common[0]] if common else np.nan
            tg.append(dg[r.i] * ratio)
        target = np.mean(tg) / KCAL if tg else np.nan
        if a["orientation"] == "reversed":
            target = -target
        o = ot.loc[a["rxn"]]
        out.append(dict(rxn=a["rxn"], fold=f, wl2_zero=int(ck == "ZERO"),
                        wl2_twin_in_train=int((train.change == ck).sum()) if ck != "ZERO" else 0,
                        all_form_in_train=int(bool(parts) and all(h in train_form for h in parts)),
                        n_training_rows=len(own), target_kcal=target,
                        ph_all_near_7=int(o.pH_min >= 6.5 and o.pH_max <= 7.5),
                        t_all_near_298=int(o.T_min >= 297 and o.T_max <= 299.5)))
    O = pd.DataFrame(out)
    O.to_csv(CV / "anchor_novelty.tsv", sep="\t", index=False, float_format="%.4f")
    print(f"{len(O)} anchors: exactly-zero {O.wl2_zero.sum()}, twin in training {int((O.wl2_twin_in_train > 0).sum())}, "
          f"all formation energies in training {O.all_form_in_train.sum()}, "
          f"neither {int(((O.wl2_zero == 0) & (O.wl2_twin_in_train == 0) & (O.all_form_in_train == 0)).sum())}; "
          f"target matched {O.target_kcal.notna().sum()}; pH all 6.5-7.5 {O.ph_all_near_7.sum()}, "
          f"T all 297-299.5 {O.t_all_near_298.sum()}")


if __name__ == "__main__":
    main()
