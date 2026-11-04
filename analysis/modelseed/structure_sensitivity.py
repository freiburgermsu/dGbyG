#!/usr/bin/env python
"""Structure-representation sensitivity of dGbyG on the 363 live openTECR anchors.

Question
  dGbyG scored on ModelSEED structures (Stage 2; comparison/anchors.tsv column
  dgbyg_dg) has an anchor MAE of ~1.6 kcal/mol, against ~0.8 for eQuilibrator
  and ~0.6 for dGPredictor, and its error is concentrated on a few compounds
  (dihydroorotate, ammonia, several hydratases). How much of that error comes
  from the MODEL, and how much from the STRUCTURES it was given?

  The shipped dGbyG ensemble was trained on TECRDB reactions written with KEGG
  structures (data/TrainingData.csv, data/KEGG/mol). Scoring each anchor from
  its openTECR KEGG equation with dGbyG's own KEGG structures is therefore the
  path the authors intended. It is also IN-SAMPLE wherever the anchor is a
  dGbyG training reaction, so its accuracy is an upper bound on what fixing the
  structures could gain. Each anchor carries an in_training flag and is scored
  in and out of that set.

Stages
  0  Baseline, run before any new dGbyG inference:
       - recompute the eQuilibrator, dGPredictor and dGbyG(ModelSEED) anchor
         accuracy from comparison/anchors.tsv with compare.py's energy_stats,
         and check the result against comparison/anchor_accuracy.tsv;
       - rebuild every anchor's dGbyG(ModelSEED) value from compound_heads.npy
         and the ModelSEED stoichiometry, exactly as predict_reactions.py does;
       - re-predict every anchor compound from its ModelSEED SMILES with the
         CPU model, to confirm the stored heads and normalised SMILES.
  1  KEGG reaction path. For each anchor:
       - take the first alternative of the openTECR KEGG equation and drop
         H+ (C00080);
       - build every species with dGbyG.api.Compound(cid, 'kegg');
       - check balance with dGbyG's Reaction. Its silent water auto-balancing
         is detected and flagged, never used;
       - dG and sigma are the mean and SD of the summed 100-head vector
         (= Reaction.standard_dGr_prime).
     Orientation: inside each reaction, KEGG species are paired with ModelSEED
     species by the 'KEGG:' aliases of the ModelSEED compound records, then by
     neutral InChIKey (the key build_opentecr_comparison.py matched anchors
     on), then by elimination. The paired coefficients vote on the sign.
     Agreement with the sign of dG_exp is used only as a last resort, and is
     flagged.
  2  Per compound: dGf'(ModelSEED SMILES) against dGf'(KEGG mol) for every
     pair. The cause of each structural difference is classified on the two
     dGbyG-normalised molecules. A census over the whole database (RDKit only,
     no inference) counts the compounds that carry each cause.
  3  Remedies, re-predicted on the anchors for both representations:
       taut      RDKit TautomerEnumerator().Canonicalize (default parameters)
                 on the dGbyG-normalised molecule, then dGbyG's own
                 normalisation, with explicit H kept as in the native path
       tautks    the same with stereo kept (tautomerRemoveSp3Stereo = False,
                 tautomerRemoveBondStereo = False); kept for completeness --
                 dGbyG's graph features carry no chirality, so stereo never
                 moves its output
       fix       dGbyG's normalize_mol applied to the molecule WITHOUT
                 explicit H (see "Mechanism")
       fix_taut, fix_tautks   both
       fix_amide  'fix', then only non-aromatic imidic acid / lactim groups
                  (N=C-OH) rewritten as amide / lactam (NH-C=O): a targeted
                  repair that keeps the training-set representation
  4  Sigma: per-compound sigma in the ammonia anchors and the dihydroorotate
     anchors, the reaction sigma with each species frozen at its mean head or
     swapped for its KEGG structure, and where NH4+ sits in the database's
     compound-sigma distribution.

Mechanism, documented here because the remedies depend on it
  For a SMILES or a KEGG id, Compound() builds the molecule through
  utils.to_mol, which calls Chem.AddHs. normalize_mol then runs RDKit's
  Uncharger on that molecule. With the H atoms explicit in the graph, the
  Uncharger cannot remove an H from a cationic centre ([NH4+], R-[NH3+],
  [nH+]). Because it preserves net charge, it then keeps a matching anion as
  well. As a result, ModelSEED's pH-7 microspecies keep their charges:
       NH4+                              stays [NH4+]
       an amino acid                     stays a zwitterion
       ATP and NAD                       keep [nH+] on adenine
  The training structures carry none of these charges. Given the same
  molecule without explicit H, normalize_mol neutralises them as intended,
  which is what the 'fix' variant does. ModelSEED also stores some species
  as tautomers that the KEGG structures do not use (imidic acid / lactim
  forms, e.g. S-dihydroorotate and NAD's carboxamide drawn as an imidate).
  normalize_mol does not canonicalise tautomers.

Conditions
  Both paths use dGbyG's native pH 7, I 0.25, pMg 14, 298.15 K, so every
  difference between them is structural.
Units
  kcal/mol throughout, in ModelSEED's written orientation.
Rounding
  dGbyG reaction values are rounded to 2 dp before scoring, as in Stage 2.

Outputs (results/structure_sensitivity/)
  baseline_check.tsv       Stage 0 checks (reproduced / expected / got)
  anchors_kegg.tsv         one row per anchor: both dGbyG paths, orientation,
                           balance, training overlap, remedy variants, top
                           contributing compound
  accuracy.tsv             anchor accuracy (own / common / in- and
                           out-of-training)
  compound_diffs.tsv       one row per (ModelSEED cpd, KEGG cid) anchor pair
  anchor_pairs.tsv         one row per paired species per anchor, with its signed
                           contribution coeff x (dGf_KEGG - dGf_MS) and cause
  census.tsv               database-wide counts of the detected causes
  census_kegg_classes.tsv  ModelSEED vs KEGG structure classes for every
                           KEGG-aliased compound
  remedy.tsv               anchor accuracy and representation gap per remedy
  calibration.tsv          z = (dG - dG_exp) / sigma per dGbyG path (compare.py's
                           calibration statistic)
  sigma.tsv                per-species sigma, freeze and swap attribution
  numbers.json             headline numbers
  run.log

Run (about 40 min on a loaded machine; one process, 1 torch thread by default):
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=src ~/Documents/py_venv/bin/python -B \
      analysis/modelseed/structure_sensitivity.py
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import time
import warnings
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # before torch is imported anywhere
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MSDB = Path(os.environ.get("MSDB_ROOT", REPO.parent / "ModelSEEDDatabase"))
RES = HERE / "results"
OUTD = Path(os.environ.get("STRUCT_SENS_OUT", RES / "structure_sensitivity"))
KEGG_MOL = REPO / "data" / "KEGG" / "mol"
KCAL = 4.184
PROTON_MS, PROTON_KEGG, WATER_MS = "cpd00067", "C00080", "cpd00001"
NAMED = ("rxn00187", "rxn00189", "rxn01361", "rxn01465")   # examples quoted in the task
VARIANTS = ("native", "fix", "taut", "tautks", "fix_taut", "fix_tautks", "fix_amide")

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
from compare import energy_stats  # noqa: E402  (the comparison's own metric definitions)
import cv_train as cvt  # noqa: E402            (training-set reaction keys)
from cv_predict import scaled_key  # noqa: E402

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem.MolStandardize import rdMolStandardize  # noqa: E402

RDLogger.DisableLog("rdApp.*")
from structure_repair import IMIDIC, imidic_to_amide  # noqa: E402,F811  (shared with cv_predict / predict_compounds)


# --------------------------------------------------------------------------- utils
class Tee:
    def __init__(self, path):
        self.fh = open(path, "w")

    def write(self, s):
        sys.__stdout__.write(s)
        self.fh.write(s)
        self.flush()        # keep run.log live during the long inference stages

    def flush(self):
        sys.__stdout__.flush()
        self.fh.flush()


def smi(mol):
    return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)


def charged_atoms(mol):
    return [a for a in mol.GetAtoms() if a.GetFormalCharge() != 0]


def heavy_formula(mol):
    return Counter(a.GetSymbol() for a in Chem.RemoveHs(mol).GetAtoms())


def inchi_key(mol):
    m = Chem.RemoveHs(mol)
    inchi = Chem.MolToInchi(m)
    return inchi, Chem.InchiToInchiKey(inchi) if inchi else ""


_UN = rdMolStandardize.Uncharger()


def neutral_key(mol):
    """build_opentecr_comparison.keys_from_smiles' 'full' key: InChIKey of the
    charge-neutralised largest fragment."""
    if mol is None:
        return ""
    try:
        return Chem.MolToInchiKey(_UN.uncharge(rdMolStandardize.FragmentParent(mol)))
    except Exception:
        return Chem.MolToInchiKey(mol)


def kegg_raw(cid):
    """The KEGG mol exactly as dGbyG's kegg_compound_to_mol reads it (before AddHs)."""
    return Chem.MolFromMolFile(str(KEGG_MOL / f"{cid}.mol"), removeHs=False)


def stats_row(x, y):
    r = energy_stats(x, y)
    return OrderedDict((k, (round(v, 4) if isinstance(v, float) else v)) for k, v in r.items())


def r2(x):
    return round(float(x), 2)


# ------------------------------------------------------------------ inputs
def load_msdb():
    cpd = {}
    for shard in sorted(glob.glob(str(MSDB / "Biochemistry/compound_[0-9][0-9].json"))):
        for c in json.load(open(shard)):
            kegg = set()
            for a in c.get("aliases") or []:
                if a.startswith("KEGG:"):
                    kegg.update(k.strip() for k in a[5:].split(";") if re.fullmatch(r"C\d{5}", k.strip()))
            cpd[c["id"]] = dict(name=c.get("name", ""), smiles=c.get("smiles") or "", kegg=kegg)
    rxn = {}
    for shard in sorted(glob.glob(str(MSDB / "Biochemistry/reaction_[0-9][0-9].json"))):
        for r in json.load(open(shard)):
            net = defaultdict(float)
            for s in r.get("stoichiometry") or []:
                net[s["compound"]] += float(s["coefficient"])
            # netted across compartments, H+ dropped: predict_reactions.py's reaction
            net = {c: v for c, v in net.items() if abs(v) > 1e-12 and c != PROTON_MS}
            rxn[r["id"]] = dict(net=net, obsolete=int(r.get("is_obsolete") or 0) == 1,
                                name=r.get("name", ""), definition=r.get("definition", ""))
    return cpd, rxn


def parse_kegg(eq):
    """'kegg:C1 + 2 kegg:C2 = kegg:C3' -> ({cid: signed coeff} without H+, had_proton)."""
    d = defaultdict(float)
    left, right = eq.split(" = ")
    for side, sign in ((left, -1.0), (right, 1.0)):
        for tok in side.split(" + "):
            tok = tok.strip()
            m = re.match(r"^(\d+(?:\.\d+)?)\s+(kegg:C\d{5})$", tok)
            coeff, cid = (float(m.group(1)), m.group(2)) if m else (1.0, tok)
            if not re.fullmatch(r"kegg:C\d{5}", cid):
                raise ValueError(f"unexpected token {tok!r} in {eq!r}")
            d[cid[5:]] += sign * coeff
    had_proton = PROTON_KEGG in d
    return {c: v for c, v in d.items() if abs(v) > 1e-12 and c != PROTON_KEGG}, had_proton


# ------------------------------------------------------------- dGbyG wrapper
class DGbyG:
    """Native dGbyG with a CPU model, plus the structure variants. Heads are cached
    by the final dGbyG-normalised SMILES (the model input)."""

    def __init__(self, threads):
        import torch
        torch.set_num_threads(threads)
        import dGbyG.api as api
        from dGbyG.model.inference import Inference_Model
        from dGbyG.utils.reaction_utils import is_balanced
        api.model_cache[api.infer_model_path] = Inference_Model(api.infer_model_path, device="cpu")
        self.api, self.is_balanced = api, is_balanced
        self.te = {"taut": rdMolStandardize.TautomerEnumerator()}
        p = rdMolStandardize.CleanupParameters()
        p.tautomerRemoveSp3Stereo = False
        p.tautomerRemoveBondStereo = False
        self.te["tautks"] = rdMolStandardize.TautomerEnumerator(p)
        self.cache = {}
        self.n_inferred = 0
        self.taut_fail = Counter()

    def heads(self, comp):
        s = comp.Smiles
        if s not in self.cache:
            self.cache[s] = np.asarray(comp.standard_dGf_prime_list, dtype=np.float64)
            self.n_inferred += 1
        return self.cache[s]

    def _canon(self, tag, mol):
        try:
            return self.te[tag].Canonicalize(Chem.RemoveHs(mol))
        except Exception:
            self.taut_fail[tag] += 1
            return Chem.RemoveHs(mol)

    def variants(self, native, raw):
        """native: Compound built the dGbyG way; raw: the parsed molecule before AddHs.
        Returns {variant: Compound}."""
        C = self.api.Compound
        out = {"native": native}
        out["fix"] = C(Chem.RemoveHs(raw))                       # uncharging sees implicit H
        for tag in ("taut", "tautks"):
            # tautomer only: explicit H restored so dGbyG's normalisation behaves as natively
            out[tag] = C(Chem.AddHs(self._canon(tag, native.mol)))
            # both: canonical tautomer of the fixed molecule, passed without explicit H
            out[f"fix_{tag}"] = C(self._canon(tag, out["fix"].mol))
        # targeted: fixed charges, then imidic acid / lactim -> amide / lactam only
        out["fix_amide"] = C(imidic_to_amide(out["fix"].mol))
        return out


# ------------------------------------------------------------------ pairing
def pair_species(ms_net, kd, cpd, ms_keys, kegg_keys):
    """Pair ModelSEED and KEGG species of one reaction.
    Returns (pairs [(cpd, cid, ms_coeff, kegg_coeff, method)], unpaired_ms, unpaired_kegg)."""
    pairs, ms_left, kg_left = [], dict(ms_net), dict(kd)
    for cid in list(kg_left):                                   # 1) KEGG aliases
        hits = [c for c in ms_left if cid in cpd.get(c, {}).get("kegg", ())]
        if len(hits) == 1:
            pairs.append((hits[0], cid, ms_left.pop(hits[0]), kg_left.pop(cid), "alias"))
    for cid in list(kg_left):                                   # 2) neutral InChIKey
        hits = [c for c in ms_left if ms_keys.get(c) and ms_keys[c] == kegg_keys.get(cid)]
        if len(hits) == 1:
            pairs.append((hits[0], cid, ms_left.pop(hits[0]), kg_left.pop(cid), "inchikey"))
    if len(ms_left) == 1 and len(kg_left) == 1:                 # 3) elimination
        c, cid = next(iter(ms_left)), next(iter(kg_left))
        pairs.append((c, cid, ms_left.pop(c), kg_left.pop(cid), "elimination"))
    return pairs, ms_left, kg_left


# ------------------------------------------------------------ classification
def classify(a_nat, b_nat, a_fix, b_fix, a_taut=None, b_taut=None):
    """Why do two dGbyG-normalised molecules (a = ModelSEED, b = KEGG) differ?
    Returns (cause, flags). Order: residual charge (removed by the 'fix'
    normalisation), then on the fixed molecules: tautomer (RDKit canonical
    tautomer or identical standard InChI, whose mobile-H layer unifies
    amide/imidic acid and lactam/lactim), protonation (InChIKey blocks 1+2 equal,
    protonation flag differs), stereo, same heavy-atom formula (C-H tautomer or
    isomer), different constitution."""
    sa, sb = smi(a_nat), smi(b_nat)
    flags = dict(ms_charged=len(charged_atoms(a_nat)), kegg_charged=len(charged_atoms(b_nat)))
    if sa == sb:
        return "identical", flags
    parts = []
    fa, fb = smi(a_fix), smi(b_fix)
    if (fa != sa or fb != sb) and (flags["ms_charged"] or flags["kegg_charged"]):
        parts.append("residual charge")
    if fa != fb:
        ia, ka = inchi_key(a_fix)
        ib, kb = inchi_key(b_fix)
        if (a_taut is not None and smi(a_taut) == smi(b_taut)) or (ia and ia == ib):
            parts.append("tautomer")
        elif ka[:25] == kb[:25]:
            parts.append("protonation")
        elif ka[:14] == kb[:14]:
            parts.append("stereo")
        elif Chem.MolToSmiles(Chem.RemoveHs(a_fix), isomericSmiles=False) == \
                Chem.MolToSmiles(Chem.RemoveHs(b_fix), isomericSmiles=False):
            parts.append("stereo")
        elif heavy_formula(a_fix) == heavy_formula(b_fix):
            parts.append("isomer/C-H tautomer")
        else:
            parts.append("different constitution")
    return "+".join(parts) if parts else "other normalisation", flags


# ------------------------------------------------------------------- stage 0
def stage0_baseline(an, cp, heads_all, rxn, row_of, dg):
    print("\n== Stage 0: baseline reproduction (before any new dGbyG inference)")
    rows = []
    ref = pd.read_csv(RES / "comparison" / "anchor_accuracy.tsv", sep="\t")
    for lab, k in (("eQuilibrator", "eq"), ("dGPredictor", "dgp"), ("dGbyG", "dgbyg")):
        sub = an[an[f"{k}_dg"].notna()]
        got = energy_stats(sub[f"{k}_dg"], sub["dG_exp"])
        exp = ref[(ref.source == lab) & (ref.subset == "own coverage")].iloc[0]
        for m in ("n", "mae", "rmse", "median_ae", "within_tau_pct", "pearson"):
            ok = abs(float(got[m]) - float(exp[m])) < 1e-3
            rows.append(dict(check=f"anchor_accuracy {lab} {m}", expected=round(float(exp[m]), 4),
                             got=round(float(got[m]), 4), reproduced=ok))
    # dGbyG(ModelSEED) per anchor rebuilt from the stored heads
    dif_dg, dif_sd, vecs = [], [], {}
    for r in an.rxn:
        vec = np.zeros(100)
        for c, v in rxn[r]["net"].items():
            vec += v * heads_all[row_of[c]].astype(np.float64)
        vecs[r] = vec
        a = an.set_index("rxn").loc[r]
        dif_dg.append(abs(r2(vec.mean() / KCAL) - a.dgbyg_dg))
        dif_sd.append(abs(r2(vec.std() / KCAL) - a.dgbyg_sd))
    rows.append(dict(check="dgbyg_dg rebuilt from compound_heads.npy (max |diff|, kcal)", expected=0.0,
                     got=round(max(dif_dg), 4), reproduced=max(dif_dg) < 0.006))
    rows.append(dict(check="dgbyg_sd rebuilt from compound_heads.npy (max |diff|, kcal)", expected=0.0,
                     got=round(max(dif_sd), 4), reproduced=max(dif_sd) < 0.006))
    for r, (e_dg, e_sd) in {"rxn01361": (-6.23, 7.0), "rxn01465": (-9.13, 2.5),
                            "rxn00187": (7.88, 13.6), "rxn00189": (-13.8, 13.6)}.items():
        a = an.set_index("rxn").loc[r]
        rows.append(dict(check=f"{r} dgbyg_dg / dgbyg_sd", expected=f"{e_dg} / {e_sd}",
                         got=f"{a.dgbyg_dg} / {a.dgbyg_sd}",
                         reproduced=abs(a.dgbyg_dg - e_dg) < 0.01 and abs(a.dgbyg_sd - e_sd) < 0.01))
    # the stored compound heads, re-predicted from the ModelSEED SMILES with this CPU model
    cps = cp.set_index("id")
    cids = sorted({c for r in an.rxn for c in rxn[r]["net"]})
    mx, smi_mis = 0.0, []
    nat = {}
    for c in cids:
        comp = dg.api.Compound(cps.at[c, "smiles_modelseed"], "smiles")
        nat[c] = comp
        mx = max(mx, float(np.abs(dg.heads(comp) - heads_all[row_of[c]]).max()))
        if comp.Smiles != cps.at[c, "smiles_dgbyg"]:
            smi_mis.append(c)
    rows.append(dict(check=f"{len(cids)} anchor compounds re-predicted from ModelSEED SMILES (max |head diff|, kJ)",
                     expected=0.0, got=float(f"{mx:.2e}"), reproduced=mx < 1e-3))
    rows.append(dict(check="re-normalised SMILES == compound_predictions smiles_dgbyg (mismatches)",
                     expected=0, got=len(smi_mis), reproduced=not smi_mis))
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    print(f"BASELINE REPRODUCED: {bool(tab.reproduced.all())}")
    return tab, vecs, nat


# ------------------------------------------------------------------- stage 1
def stage1_kegg(an, cpd, rxn, cp, heads_all, row_of, dg, ms_native):
    print("\n== Stage 1: KEGG-structure path")
    cps = cp.set_index("id")
    kegg_native, kegg_rawmol = {}, {}
    rec, pairs_all = [], []
    ms_keys = {c: neutral_key(Chem.MolFromSmiles(cpd[c]["smiles"])) for r in an.rxn for c in rxn[r]["net"]}
    t0 = time.time()
    for _, a in an.iterrows():
        alts = a.opentecr_reaction.split(" | ")
        kd, had_proton = parse_kegg(alts[0])
        alt_sets = {frozenset(parse_kegg(x)[0]) for x in alts}
        for cid in kd:
            if cid not in kegg_native:
                kegg_native[cid] = dg.api.Compound(cid, "kegg")
                kegg_rawmol[cid] = kegg_raw(cid)
        kkeys = {cid: neutral_key(Chem.RemoveHs(kegg_rawmol[cid])) for cid in kd}
        ms_net = rxn[a.rxn]["net"]
        pairs, ms_left, kg_left = pair_species(ms_net, kd, cpd, ms_keys, kkeys)
        votes = [np.sign(mc) * np.sign(kc) for _, _, mc, kc, _ in pairs]
        coeff_ok = all(abs(abs(mc) - abs(kc)) < 1e-9 for _, _, mc, kc, _ in pairs)
        # dGbyG's own balance test on the KEGG reaction as written, and what Reaction() does to it
        comps = {kegg_native[cid]: v for cid, v in kd.items()}
        bal = dg.is_balanced({c.mol: v for c, v in comps.items()}, ignore_H2O=False, ignore_H_ion=True,
                             ignore_charge=False, ignore_H=True)
        robj = dg.api.Reaction(dict(comps))
        water_added = len(robj.reaction) != len(comps) or any(robj.reaction[c] != v for c, v in comps.items())
        vec_k = np.zeros(100)
        for cid, v in kd.items():
            vec_k += v * dg.heads(kegg_native[cid])
        if votes and len(set(votes)) == 1:
            orient, how = int(votes[0]), "species pairing"
        else:
            orient = int(np.sign(a.dG_exp) * np.sign(vec_k.mean())) or 1
            how = "SIGN OF dG_exp (fallback)"
        vec_k = orient * vec_k
        vec_ms = np.zeros(100)
        for c, v in ms_net.items():
            vec_ms += v * heads_all[row_of[c]].astype(np.float64)
        # decomposition: dG_KEGG - dG_MS = sum over pairs of ms_coeff * (dGf_KEGG - dGf_MS)
        complete = not ms_left and not kg_left and coeff_ok
        contrib = {}
        for c, cid, mc, kc, meth in pairs:
            d = (dg.heads(kegg_native[cid]).mean() - heads_all[row_of[c]].astype(np.float64).mean()) / KCAL
            contrib[(c, cid)] = mc * d
            pairs_all.append(dict(rxn=a.rxn, cpd=c, cid=cid, ms_coeff=mc, kegg_coeff=kc, method=meth,
                                  key_match=ms_keys.get(c) == kkeys.get(cid), contrib_kcal=mc * d))
        resid = (vec_k.mean() - vec_ms.mean()) / KCAL - sum(contrib.values()) if complete else math.nan
        top = max(contrib.items(), key=lambda kv: abs(kv[1])) if contrib else ((None, None), math.nan)
        rec.append(OrderedDict(
            rxn=a.rxn, name=rxn[a.rxn]["name"], dG_exp=a.dG_exp, eq_dg=a.eq_dg, dgp_dg=a.dgp_dg,
            dgbyg_ms_dg=a.dgbyg_dg, dgbyg_ms_sd=a.dgbyg_sd,
            dgbyg_kegg_dg=r2(vec_k.mean() / KCAL), dgbyg_kegg_sd=r2(vec_k.std() / KCAL),
            gap_kegg_minus_ms=r2(vec_k.mean() / KCAL) - a.dgbyg_dg,
            orientation=("same" if orient == 1 else "reversed"), orientation_method=how,
            pairing=",".join(f"{k}:{v}" for k, v in sorted(Counter(p[4] for p in pairs).items())),
            unpaired_ms=";".join(ms_left), unpaired_kegg=";".join(kg_left), coeff_match=coeff_ok,
            neutral_key_mismatch=";".join(f"{c}/{cid}" for c, cid, *_ in pairs if ms_keys.get(c) != kkeys.get(cid)),
            decomposition_residual=resid, kegg_balanced_as_written=bool(bal), reaction_api_added_water=water_added,
            kegg_eq_used=alts[0], n_alternatives=len(alts), alternatives_same_species=len(alt_sets) == 1,
            had_proton=had_proton, is_transport=a.is_transport,
            top_cpd=top[0][0], top_cid=top[0][1], top_contrib_kcal=round(top[1], 3)))
    print(f"  {len(kegg_native)} KEGG compounds, {dg.n_inferred} inferences, {time.time() - t0:.0f} s")
    return pd.DataFrame(rec), pd.DataFrame(pairs_all), kegg_native, kegg_rawmol


def training_overlap(ak, kegg_native, ms_native, rxn, an):
    """Is the anchor one of dGbyG's training reactions? cv_predict.py's key
    (dGbyG-normalised SMILES, water/H+ ignored, orientation-invariant, scaled),
    computed from the KEGG structures, and also from the ModelSEED structures
    (the representation cv_predict.py matches on)."""
    from dGbyG.utils.reaction_utils import parse_equation
    df = pd.read_csv(REPO / "data/TrainingData.csv")
    exact, skel = set(), set()
    for kind, e in zip(df["type"], df["reaction"]):
        if kind == "formation dg":
            continue
        st = defaultdict(float)
        for s, c in parse_equation(e).items():
            st[cvt.norm_smiles(s)] += float(c)
        exact.add(scaled_key(st))
        skel.add(scaled_key(st, strip_stereo=True))
    out_k, out_m = [], []
    for _, a in ak.iterrows():
        kd, _ = parse_kegg(a.kegg_eq_used)
        st = defaultdict(float)
        for cid, v in kd.items():
            st[kegg_native[cid].Smiles] += v
        out_k.append("exact" if scaled_key(st) in exact else "skeleton" if scaled_key(st, True) in skel else "no")
        st = defaultdict(float)
        for c, v in rxn[a.rxn]["net"].items():
            st[ms_native[c].Smiles] += v
        out_m.append("exact" if scaled_key(st) in exact else "skeleton" if scaled_key(st, True) in skel else "no")
    return out_k, out_m


# ------------------------------------------------------------------- stage 2
def stage2_compounds(pairs, cpd, cp, heads_all, row_of, dg, ms_native, kegg_native, msv, kgv):
    print("\n== Stage 2: per-compound comparison")
    cps = cp.set_index("id")
    rows = []
    for (c, cid), g in pairs.groupby(["cpd", "cid"]):
        hm = heads_all[row_of[c]].astype(np.float64)
        hk = dg.heads(kegg_native[cid])
        cause, flags = classify(ms_native[c].mol, kegg_native[cid].mol, msv[c]["fix"].mol, kgv[cid]["fix"].mol,
                                msv[c]["fix_tautks"].mol, kgv[cid]["fix_tautks"].mol)
        rows.append(OrderedDict(
            cpd=c, cid=cid, name=cpd[c]["name"],
            dGf_ms=round(hm.mean() / KCAL, 3), sd_ms=round(hm.std() / KCAL, 3),
            dGf_kegg=round(hk.mean() / KCAL, 3), sd_kegg=round(hk.std() / KCAL, 3),
            diff_kegg_minus_ms=round((hk.mean() - hm.mean()) / KCAL, 3),
            cause=cause, **flags,
            same_neutral_inchikey=bool(g.key_match.iloc[0]),
            n_anchors=len(g), anchors=";".join(sorted(g.rxn)),
            sum_abs_contrib=round(float(g.contrib_kcal.abs().sum()), 3),
            max_abs_contrib=round(float(g.contrib_kcal.abs().max()), 3),
            smiles_modelseed=cps.at[c, "smiles_modelseed"],
            smiles_ms_dgbyg=ms_native[c].Smiles, smiles_kegg_dgbyg=kegg_native[cid].Smiles,
            smiles_ms_fix=msv[c]["fix"].Smiles, smiles_kegg_fix=kgv[cid]["fix"].Smiles))
    t = pd.DataFrame(rows).sort_values("diff_kegg_minus_ms", key=lambda s: -s.abs()).reset_index(drop=True)
    return t


def census(cp, rxn):
    """RDKit-only counts over every compound Stage 1 predicted (status ok)."""
    print("\n== Stage 2b: database census (RDKit only)")
    from dGbyG.utils.mol_utils import normalize_mol
    ok = cp[cp.status == "ok"]
    pat = {k: Chem.MolFromSmarts(v) for k, v in {
        "imidic acid / lactim (N=C-OH)": "[#7X2]=[#6X3]-[OX2H1]",
        "hydroxy-azine lactim (OH on aromatic C next to n)": "[OX2H1]-c:n",
        "enol (C=C-OH)": "[OX2H1]-[#6X3]=[#6X3]",
    }.items()}
    recs = {}
    t0 = time.time()
    for cid_, s_ms, s_nd in zip(ok["id"], ok["smiles_modelseed"], ok["smiles_dgbyg"]):
        m = Chem.MolFromSmiles(s_nd)
        f = Chem.MolFromSmiles(s_ms)
        fixs = smi(normalize_mol(f)) if f is not None else s_nd
        fm = Chem.MolFromSmiles(fixs)
        pos = [a for a in m.GetAtoms() if a.GetFormalCharge() > 0]
        neg = [a for a in m.GetAtoms() if a.GetFormalCharge() < 0]
        recs[cid_] = dict(
            wildcard="*" in s_ms,
            any_charge=bool(pos or neg),
            cation_with_H=any(a.GetTotalNumHs() > 0 for a in pos),
            cation_without_H=any(a.GetTotalNumHs() == 0 for a in pos),
            zwitterion=bool(pos and neg),
            net_charge=sum(a.GetFormalCharge() for a in m.GetAtoms()) != 0,
            fix_changes=fixs != s_nd,
            fix_any_charge=bool(charged_atoms(fm)) if fm is not None else False,
            **{k: bool(m.HasSubstructMatch(p)) for k, p in pat.items()})
    print(f"  {len(recs)} compounds in {time.time() - t0:.0f} s")
    C = pd.DataFrame.from_dict(recs, orient="index")
    live_ok = pd.read_csv(RES / "reaction_predictions.tsv", sep="\t", usecols=["rxn", "is_obsolete", "status"],
                          keep_default_na=False, dtype=str)
    live_ok = live_ok[(live_ok.is_obsolete.astype(int) != 1) & (live_ok.status == "ok")].rxn.tolist()
    rows = []
    for col in C.columns:
        hit = set(C.index[C[col]])
        nr = sum(1 for r in live_ok if any(c in hit for c in rxn[r]["net"]))
        rows.append(OrderedDict(flag=col, n_compounds=len(hit), pct_of_ok=round(100 * len(hit) / len(C), 2),
                                live_ok_reactions_touching=nr,
                                pct_of_live_ok=round(100 * nr / len(live_ok), 2)))
    tab = pd.DataFrame(rows)
    # training-set prevalence of the same flags (normalised the dGbyG way)
    from dGbyG.utils.reaction_utils import parse_equation
    tr = set()
    for e in pd.read_csv(REPO / "data/TrainingData.csv")["reaction"]:
        tr.update(cvt.norm_smiles(s) for s in parse_equation(e))
    trm = [Chem.MolFromSmiles(s) for s in tr]
    trm = [m for m in trm if m is not None]
    tab["training_structures_with_flag"] = [
        sum(1 for m in trm if any(a.GetFormalCharge() > 0 and a.GetTotalNumHs() > 0 for a in m.GetAtoms()))
        if f == "cation_with_H" else
        sum(1 for m in trm if charged_atoms(m)) if f == "any_charge" else
        sum(1 for m in trm if m.HasSubstructMatch(pat[f])) if f in pat else ""
        for f in tab.flag]
    tab.attrs["n_training_structures"] = len(trm)
    print(tab.to_string(index=False))
    print(f"  ({len(trm)} distinct dGbyG-normalised training structures)")
    return tab, C, live_ok


def census_kegg(cp, cpd):
    """Every ok compound with a KEGG alias that dGbyG has a mol for: native dGbyG
    normalisation of both structures, classified (InChI-based, no tautomer
    enumeration)."""
    print("\n== Stage 2c: ModelSEED vs KEGG structures for every KEGG-aliased compound")
    from dGbyG.utils.mol_utils import normalize_mol
    have = {f[:-4] for f in os.listdir(KEGG_MOL) if f.endswith(".mol")}
    ok = cp[cp.status == "ok"]
    kn_cache = {}

    def kegg_forms(cid):
        if cid not in kn_cache:
            try:
                raw = kegg_raw(cid)
                kn_cache[cid] = (normalize_mol(Chem.AddHs(raw)), normalize_mol(Chem.RemoveHs(raw)),
                                 neutral_key(Chem.RemoveHs(raw))) if raw is not None else None
            except Exception:
                kn_cache[cid] = None
        return kn_cache[cid]

    rows = []
    t0 = time.time()
    for c, s_ms in zip(ok["id"], ok["smiles_modelseed"]):
        ks = sorted(k for k in cpd[c]["kegg"] if k in have)
        if not ks or "*" in s_ms:
            continue
        msraw = Chem.MolFromSmiles(s_ms)
        mkey = neutral_key(msraw)
        forms = [(k, kegg_forms(k)) for k in ks]
        forms = [(k, f) for k, f in forms if f is not None]
        if not forms:
            continue
        # prefer the alias with the same neutral InChIKey, then the same skeleton
        forms.sort(key=lambda kf: (kf[1][2] != mkey, kf[1][2][:14] != mkey[:14], kf[0]))
        k, (knat, kfix, kkey) = forms[0]
        try:
            mnat = normalize_mol(Chem.AddHs(msraw))
            mfix = normalize_mol(Chem.RemoveHs(msraw))
            cause, _ = classify(mnat, knat, mfix, kfix)
        except Exception as e:
            cause = f"error:{type(e).__name__}"
        rows.append((c, k, cause))
    print(f"  {len(rows)} compounds in {time.time() - t0:.0f} s")
    t = pd.DataFrame(rows, columns=["cpd", "cid", "cause"])
    tab = t.cause.value_counts().rename_axis("cause").reset_index(name="n_compounds")
    tab["pct"] = (100 * tab.n_compounds / len(t)).round(2)
    print(tab.to_string(index=False))
    return tab, t


# ------------------------------------------------------------------- stage 3
def remedy_table(an, ak, rxn, msv, kgv, dg):
    print("\n== Stage 3: remedies on the anchors")
    akx = ak.set_index("rxn")
    vals, sds = {}, {}
    for v in VARIANTS:
        ms_v, kg_v, ms_s, kg_s = [], [], [], []
        for r in an.rxn:
            vec = np.zeros(100)
            for c, co in rxn[r]["net"].items():
                vec += co * dg.heads(msv[c][v])
            ms_v.append(r2(vec.mean() / KCAL))
            ms_s.append(r2(vec.std() / KCAL))
            kd, _ = parse_kegg(akx.at[r, "kegg_eq_used"])
            o = 1 if akx.at[r, "orientation"] == "same" else -1
            vec = np.zeros(100)
            for cid, co in kd.items():
                vec += co * dg.heads(kgv[cid][v])
            kg_v.append(r2(o * vec.mean() / KCAL))
            kg_s.append(r2(vec.std() / KCAL))
        vals[f"ms_{v}"], vals[f"kegg_{v}"] = np.array(ms_v), np.array(kg_v)
        sds[f"ms_{v}"], sds[f"kegg_{v}"] = np.array(ms_s), np.array(kg_s)
    y = an.dG_exp.to_numpy()
    kref = vals["kegg_native"]
    rows = []
    for v in VARIANTS:
        for side in ("ms", "kegg"):
            x = vals[f"{side}_{v}"]
            st = stats_row(x, y)
            gap_k = np.abs(x - kref)
            gap_pair = np.abs(vals[f"ms_{v}"] - vals[f"kegg_{v}"])
            rows.append(OrderedDict(
                structures=side, variant=v, **{k: st[k] for k in ("n", "mae", "rmse", "median_ae",
                                                                   "within_tau_pct", "pearson", "bias")},
                mean_abs_gap_to_kegg_native=round(float(gap_k.mean()), 4),
                n_gap_to_kegg_native_gt1=int((gap_k > 1).sum()),
                mean_abs_gap_ms_vs_kegg_same_variant=round(float(gap_pair.mean()), 4),
                n_ms_vs_kegg_same_variant_gt1=int((gap_pair > 1).sum())))
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    return tab, vals, sds


def calibration_table(an, vals, sds):
    """compare.calibration's statistic for each dGbyG path: z = (dG - dG_exp) / sigma,
    sigma == 0 excluded and counted; strict |z| < 1 and < 2."""
    y = an.dG_exp.to_numpy()
    rows = []
    for key in ("ms_native", "kegg_native", "ms_fix_amide", "ms_fix_tautks", "kegg_fix_tautks"):
        x, s = vals[key], sds[key]
        ok = s > 0
        z = (x[ok] - y[ok]) / s[ok]
        rows.append(OrderedDict(path=key, n=int(ok.sum()), n_sigma_zero=int((~ok).sum()),
                                rms_z=round(float(np.sqrt(np.mean(z ** 2))), 3),
                                median_abs_z=round(float(np.median(np.abs(z))), 3),
                                within_1sd_pct=round(100 * float(np.mean(np.abs(z) < 1)), 2),
                                within_2sd_pct=round(100 * float(np.mean(np.abs(z) < 2)), 2),
                                median_sigma=round(float(np.median(s)), 3)))
    tab = pd.DataFrame(rows)
    print("\n== Sigma calibration against dG_exp (anchors)")
    print(tab.to_string(index=False))
    return tab


# ------------------------------------------------------------------- stage 4
def sigma_table(rxns, rxn, pairs, heads_all, row_of, dg, kegg_native, cpd):
    rows = []
    pmap = {(p.rxn, p.cpd): p.cid for p in pairs.itertuples()}
    for r in rxns:
        net = rxn[r]["net"]
        H = {c: heads_all[row_of[c]].astype(np.float64) for c in net}
        vec = sum(v * H[c] for c, v in net.items())
        for c, v in net.items():
            frozen = vec - v * (H[c] - H[c].mean())
            cid = pmap.get((r, c))
            row = OrderedDict(rxn=r, cpd=c, name=cpd[c]["name"], coeff=v,
                              dGf_ms=round(H[c].mean() / KCAL, 3), sd_ms=round(H[c].std() / KCAL, 3),
                              cid=cid, rxn_dG_ms=round(vec.mean() / KCAL, 3), rxn_sd_ms=round(vec.std() / KCAL, 3),
                              rxn_sd_if_frozen=round(frozen.std() / KCAL, 3))
            if cid:
                hk = dg.heads(kegg_native[cid])
                sw = vec + v * (hk - H[c])
                row.update(dGf_kegg=round(hk.mean() / KCAL, 3), sd_kegg=round(hk.std() / KCAL, 3),
                           rxn_dG_if_swapped=round(sw.mean() / KCAL, 3), rxn_sd_if_swapped=round(sw.std() / KCAL, 3))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--threads", type=int, default=1,
                    help="torch threads (max 4 on this shared machine). On a loaded machine 1 is much "
                         "faster: 4 OpenMP threads measured 7.6 s/compound vs 0.18 s for 1 at load ~107")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: first N anchors (+ the named ones) and N*50 census compounds; "
                         "baseline checks then fail by construction")
    args = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(OUTD / "run.log")
    t_start = time.time()

    an = pd.read_csv(RES / "comparison" / "anchors.tsv", sep="\t")
    assert len(an) == 363 and an.rxn.is_unique
    if args.limit:
        an = an[an.rxn.isin(list(an.rxn[:args.limit]) + list(NAMED))].reset_index(drop=True)
    cp = pd.read_csv(RES / "compound_predictions.tsv", sep="\t", keep_default_na=False, dtype=str)
    heads_all = np.load(RES / "compound_heads.npy")
    row_of = dict(zip(cp["id"], cp["row"].astype(int)))
    cpd, rxn = load_msdb()
    print(f"{len(an)} anchors; {len(cpd):,} compound and {len(rxn):,} reaction records from {MSDB}")

    dg = DGbyG(args.threads)
    base, ms_vecs, ms_native = stage0_baseline(an, cp, heads_all, rxn, row_of, dg)
    base.to_csv(OUTD / "baseline_check.tsv", sep="\t", index=False)

    ak, pairs, kegg_native, kegg_rawmol = stage1_kegg(an, cpd, rxn, cp, heads_all, row_of, dg, ms_native)
    ak["in_training_kegg_key"], ak["in_training_ms_key"] = training_overlap(ak, kegg_native, ms_native, rxn, an)
    print("  orientation:", dict(Counter(ak.orientation)), dict(Counter(ak.orientation_method)))
    print("  pairing methods:", dict(Counter(pairs.method)), " pairs whose neutral InChIKeys differ:",
          int((~pairs.key_match).sum()))
    print("  KEGG reaction balanced as written:", int(ak.kegg_balanced_as_written.sum()),
          "| Reaction() added water:", int(ak.reaction_api_added_water.sum()),
          "| incomplete pairing:", int(ak.decomposition_residual.isna().sum()),
          "| max |decomposition residual| (kcal):", float(np.nanmax(np.abs(ak.decomposition_residual))))
    print("  in training (KEGG key):", dict(Counter(ak.in_training_kegg_key)),
          " (ModelSEED key, as cv_predict.py matches):", dict(Counter(ak.in_training_ms_key)))

    # ---- structure variants for every anchor species
    t0 = time.time()
    msv = {c: dg.variants(ms_native[c], Chem.MolFromSmiles(cpd[c]["smiles"])) for c in ms_native}
    kgv = {cid: dg.variants(kegg_native[cid], kegg_rawmol[cid]) for cid in kegg_native}
    for d in (msv, kgv):
        for vv in d.values():
            for comp in vv.values():
                dg.heads(comp)
    print(f"\n  variants built: {dg.n_inferred} inferences so far, {time.time() - t0:.0f} s; "
          f"tautomer canonicalisation failures: {dict(dg.taut_fail)}")
    lost = sum(1 for d in (msv, kgv) for vv in d.values()
               if vv["taut"].Smiles.count("@") < vv["native"].Smiles.count("@"))
    print(f"  structures that lost stereo under default TautomerEnumerator: {lost}")

    # ---- accuracy
    ak = ak.merge(an[["rxn", "gc_dg"]], on="rxn")
    akx = ak.set_index("rxn")
    rem, vals, sds = remedy_table(an, ak, rxn, msv, kgv, dg)
    for k, v in vals.items():
        if k not in ("ms_native", "kegg_native"):
            akx.loc[an.rxn, f"dgbyg_{k}_dg"] = v
            akx.loc[an.rxn, f"dgbyg_{k}_sd"] = sds[k]
    cal = calibration_table(an, vals, sds)
    ak = akx.reset_index()
    print("\n== Accuracy against dG_exp")
    src = OrderedDict([("eQuilibrator", "eq_dg"), ("dGPredictor", "dgp_dg"), ("Group contribution", "gc_dg"),
                       ("dGbyG_ModelSEED", "dgbyg_ms_dg"), ("dGbyG_KEGG", "dgbyg_kegg_dg"),
                       ("dGbyG_ModelSEED_fix", "dgbyg_ms_fix_dg"),
                       ("dGbyG_ModelSEED_fix_amide", "dgbyg_ms_fix_amide_dg"),
                       ("dGbyG_ModelSEED_fix_tautks", "dgbyg_ms_fix_tautks_dg")])
    core = ["eq_dg", "dgp_dg", "dgbyg_ms_dg", "dgbyg_kegg_dg"]
    common = ak[core].notna().all(axis=1)
    subsets = OrderedDict([
        ("own coverage", None), ("common (eQ, dGP, dGbyG x2)", common),
        ("common, in dGbyG training (KEGG key exact/skeleton)", common & (ak.in_training_kegg_key != "no")),
        ("common, NOT in dGbyG training", common & (ak.in_training_kegg_key == "no")),
        ("common, non-transport", common & (ak.is_transport.astype(str) != "1")),
    ])
    rows = []
    for sname, m in subsets.items():
        for lab, col in src.items():
            mm = ak[col].notna() if m is None else (m & ak[col].notna())
            if mm.sum() < 3:
                if lab == "eQuilibrator":
                    print(f"  (subset '{sname}': {int(mm.sum())} anchors -- not scored)")
                continue
            st = stats_row(ak.loc[mm, col], ak.loc[mm, "dG_exp"])
            rows.append(OrderedDict(subset=sname, source=lab, **st))
    acc = pd.DataFrame(rows)
    print(acc[["subset", "source", "n", "mae", "rmse", "median_ae", "within_tau_pct", "pearson", "bias"]]
          .to_string(index=False))

    gap = (ak.dgbyg_kegg_dg - ak.dgbyg_ms_dg).abs()
    err_ms = (ak.dgbyg_ms_dg - ak.dG_exp).abs()
    err_k = (ak.dgbyg_kegg_dg - ak.dG_exp).abs()
    share = float((err_ms.sum() - err_k.sum()) / err_ms.sum())
    print(f"\n  |dG_KEGG - dG_MS|: >0.01 in {(gap > 0.01).sum()}, >1 in {(gap > 1).sum()}, "
          f">2 in {(gap > 2).sum()}, >5 in {(gap > 5).sum()} anchors; "
          f"total-absolute-error reduction MS->KEGG = {100 * share:.1f}%")
    worst = ak.assign(gap=gap, err_ms=err_ms, err_k=err_k).sort_values("gap", ascending=False)
    print(worst[["rxn", "name", "dG_exp", "eq_dg", "dgp_dg", "dgbyg_ms_dg", "dgbyg_ms_sd", "dgbyg_kegg_dg",
                 "dgbyg_kegg_sd", "top_cpd", "top_cid", "top_contrib_kcal", "in_training_kegg_key"]]
          .head(25).to_string(index=False))

    # ---- per compound
    ct = stage2_compounds(pairs, cpd, cp, heads_all, row_of, dg, ms_native, kegg_native, msv, kgv)
    ct.to_csv(OUTD / "compound_diffs.tsv", sep="\t", index=False)
    big = ct[ct.diff_kegg_minus_ms.abs() > 1]
    print(f"  {len(ct)} distinct (ModelSEED, KEGG) pairs; identical dGbyG structure: "
          f"{(ct.cause == 'identical').sum()}; |dGf diff| > 1 kcal: {len(big)}")
    print("  causes, all pairs:", dict(Counter(ct.cause)))
    print("  causes, |diff| > 1:", dict(Counter(big.cause)))
    with pd.option_context("display.max_colwidth", 70, "display.width", 250):
        print(big[["cpd", "cid", "name", "dGf_ms", "sd_ms", "dGf_kegg", "sd_kegg", "diff_kegg_minus_ms", "cause",
                   "n_anchors", "sum_abs_contrib", "smiles_ms_dgbyg", "smiles_kegg_dgbyg"]].to_string(index=False))
    # how much of the anchor gap each cause carries
    pc = pairs.merge(ct[["cpd", "cid", "cause", "name"]], on=["cpd", "cid"])
    pc.to_csv(OUTD / "anchor_pairs.tsv", sep="\t", index=False)
    # NET attribution: within each anchor, signed contributions are summed per cause first, so a
    # cofactor pair on opposite sides (CoA / acyl-CoA, NAD / NADH ...) cancels as it does in dG
    net = pc.groupby(["rxn", "cause"]).contrib_kcal.sum().reset_index()
    by_cause = net.groupby("cause").contrib_kcal.agg(
        n_anchors_present="size", sum_abs_net_contrib=lambda s: s.abs().sum(),
        n_anchors_abs_net_gt1=lambda s: (s.abs() > 1).sum())
    gross = pc.groupby("cause").contrib_kcal.agg(sum_abs_gross_contrib=lambda s: s.abs().sum())
    by_cause = by_cause.join(gross)
    by_cause["share_of_net_pct"] = 100 * by_cause.sum_abs_net_contrib / by_cause.sum_abs_net_contrib.sum()
    print("\n  anchor gap (dG_KEGG - dG_MS) attributed to causes; net = signed contributions summed per "
          "cause within each anchor, then |.| summed over anchors; gross = no cancellation:")
    print(by_cause.round(3).to_string())
    dom = net.loc[net.contrib_kcal.abs().groupby(net.rxn).idxmax()].set_index("rxn")
    ak["dominant_cause"] = ak.rxn.map(dom.cause)
    ak["dominant_cause_net_kcal"] = ak.rxn.map(dom.contrib_kcal).round(3)
    print("\n  compounds with the largest gross per-compound contribution (cofactor pairs cancel in the net):")
    with pd.option_context("display.max_colwidth", 60, "display.width", 250):
        print(ct.sort_values("sum_abs_contrib", ascending=False)
              [["cpd", "cid", "name", "diff_kegg_minus_ms", "sd_ms", "sd_kegg", "cause", "n_anchors",
                "sum_abs_contrib", "smiles_ms_dgbyg"]].head(20).to_string(index=False))
    tc = ak.assign(gap=(ak.dgbyg_kegg_dg - ak.dgbyg_ms_dg).abs())
    tc = tc[tc.gap > 1]
    has_water = tc.rxn.map(lambda r: WATER_MS in rxn[r]["net"])
    print("\n  anchors with |dG_KEGG - dG_MS| > 1, by their dominant (net) cause:")
    print(tc.assign(with_water=has_water).groupby("dominant_cause").agg(
        n=("rxn", "size"), mean_gap=("gap", "mean"), with_water=("with_water", "sum"),
        mean_signed_gap_kegg_minus_ms=("gap_kegg_minus_ms", "mean")).round(3).to_string())

    cpc = cp.head(args.limit * 50) if args.limit else cp
    cen, cflags, live_ok = census(cpc, rxn)
    ck, ckt = census_kegg(cpc, cpd)
    cen.to_csv(OUTD / "census.tsv", sep="\t", index=False)
    ck.to_csv(OUTD / "census_kegg_classes.tsv", sep="\t", index=False)

    # ---- sigma
    print("\n== Stage 4: sigma")
    top_gap = [r for r in worst.rxn.head(6) if r not in NAMED]
    sg = sigma_table(list(NAMED) + top_gap, rxn, pairs, heads_all, row_of, dg, kegg_native, cpd)
    sg.to_csv(OUTD / "sigma.tsv", sep="\t", index=False)
    print(sg.to_string(index=False))
    sd_all = cp.loc[cp.status == "ok", ["id", "dgbyg_sd_kcal"]].assign(sd=lambda d: d.dgbyg_sd_kcal.astype(float))
    sd_all = sd_all.set_index("id").sd
    nh4 = sd_all["cpd00013"]
    print(f"\n  NH4+ (cpd00013) compound sigma {nh4:.2f} kcal: percentile "
          f"{100 * (sd_all < nh4).mean():.2f} of {len(sd_all)} compounds; KEGG NH3 (C00014) sigma "
          f"{dg.heads(kegg_native['C00014']).std() / KCAL:.2f}")
    fl = cflags.reindex(sd_all.index).fillna(False).astype(bool)
    sig_by = OrderedDict()
    for name, m in (("cation_with_H", fl.cation_with_H), ("no charge left", ~fl.any_charge),
                    ("any charge left", fl.any_charge)):
        v = sd_all[m]
        sig_by[name] = dict(n=int(len(v)), median_sd=round(float(v.median()), 3),
                            pct_sd_gt5=round(100 * float((v > 5).mean()), 2))
    print("  compound sigma by residual-charge flag:", sig_by)
    rp = pd.read_csv(RES / "reaction_predictions.tsv", sep="\t", keep_default_na=False, dtype=str)
    rp = rp[(rp.is_obsolete.astype(int) != 1) & (rp.status == "ok")].set_index("rxn")
    with_nh4 = [r for r in rp.index if "cpd00013" in rxn[r]["net"]]
    sd_r = rp.sd_kcal.astype(float)
    nh4_rxn = dict(n=len(with_nh4), median_sd=float(sd_r[with_nh4].median()),
                   median_sd_other=float(sd_r.drop(with_nh4).median()),
                   pct_sd_ge_13=round(100 * float((sd_r[with_nh4] >= 13).mean()), 1))
    print("  live ok reactions containing cpd00013:", nh4_rxn)

    # ---- outputs
    ak.to_csv(OUTD / "anchors_kegg.tsv", sep="\t", index=False)
    acc.to_csv(OUTD / "accuracy.tsv", sep="\t", index=False)
    rem.to_csv(OUTD / "remedy.tsv", sep="\t", index=False)
    cal.to_csv(OUTD / "calibration.tsv", sep="\t", index=False)
    g = lambda s, src_: acc[(acc.subset == s) & (acc.source == src_)].iloc[0].to_dict()  # noqa: E731
    numbers = OrderedDict(
        baseline_reproduced=bool(base.reproduced.all()),
        anchors=len(ak), orientation=dict(Counter(ak.orientation)),
        orientation_fallback=int((ak.orientation_method != "species pairing").sum()),
        kegg_reaction_api_added_water=int(ak.reaction_api_added_water.sum()),
        in_training_kegg_key=dict(Counter(ak.in_training_kegg_key)),
        in_training_ms_key=dict(Counter(ak.in_training_ms_key)),
        common=dict((s, g("common (eQ, dGP, dGbyG x2)", s)) for s in
                    ("eQuilibrator", "dGPredictor", "dGbyG_ModelSEED", "dGbyG_KEGG")),
        gap_gt1=int((gap > 1).sum()), gap_gt2=int((gap > 2).sum()),
        abs_error_reduction_pct=round(100 * share, 2),
        compound_pairs=len(ct), compound_pairs_diff_gt1=len(big),
        causes_diff_gt1=dict(Counter(big.cause)),
        gap_share_by_cause_net_pct=by_cause.share_of_net_pct.round(2).to_dict(),
        census=cen.set_index("flag")[["n_compounds", "live_ok_reactions_touching"]].to_dict("index"),
        census_kegg=ck.set_index("cause").n_compounds.to_dict(),
        nh4_sigma_kcal=round(float(nh4), 3), nh4_reactions=nh4_rxn, sigma_by_charge_flag=sig_by,
        calibration=cal.set_index("path").to_dict("index"),
        remedy={f"{r.structures}_{r.variant}": dict(mae=r.mae, rmse=r.rmse,
                                                    gap_to_kegg_native=r.mean_abs_gap_to_kegg_native)
                for r in rem.itertuples()},
        inferences=dg.n_inferred, runtime_s=round(time.time() - t_start))
    json.dump(numbers, open(OUTD / "numbers.json", "w"), indent=1, default=str)
    print(f"\nwrote {OUTD}  ({time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
