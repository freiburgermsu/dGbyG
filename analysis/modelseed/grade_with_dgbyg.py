#!/usr/bin/env python
"""Stage 5: grade dGbyG inside the MSDB paper's own gold / silver / bronze scheme.

The paper grades every ModelSEED reaction with
ModelSEEDDatabase/Scripts/Thermodynamics/SourceGrading/grade_thermo_sources.py
(+ optimize_thermo_source_assignment.py for the error models). This script
runs that code UNCHANGED, adding dGbyG as a fourth predictor (short code DGB)
purely by in-process monkeypatching. Nothing under ModelSEEDDatabase or
MSDB_thermo_paper is written: THERMO_GRADING_OUT / GRADES_OUT / EQDGP_OUT
point at results/grading/ before the SourceGrading modules are imported, and
sys.dont_write_bytecode is set before that import so no __pycache__ lands in
the SourceGrading directory (run with ``python -B`` anyway).

WHAT IS PATCHED, AND WHY EACH PATCH IS NEEDED
  load_db            wrapped: the original reads reaction JSON (obsolete records
                     included); the wrapper left-merges dg_DGB / sig_DGB / op_DGB
                     from results/reaction_predictions.tsv for rows with
                     status == 'ok' (dG_kcal, |sd_kcal|, op). Other statuses
                     (wildcard, null_net, missing_structure, unbalanced,
                     unbalanced_water, compound_failed) stay NaN: dGbyG declined.
  fit_error_models   re-defined from its own source with one edit,
  fit_p_ok           proxy["DGB"] = "EQ": dGbyG's proxy tier is |dGbyG - eQ| on
                     reactions where eQuilibrator sigma <= 0.70 (EQ_TRUSTED_SIGMA),
                     exactly as for GC and dGPredictor. DGB is NOT made a proxy
                     for any other source and the trusted-sigma table is
                     unchanged, so the other sources' curves are untouched.
  build              re-defined from its own source with two edits: PRECEDENCE
                     (the best_source tie-break) comes from GRADE_PRECEDENCE,
                     and the pandas-3 fix ranks.fillna(inf).idxmin(...) --
                     the shipped idxmin raises on all-NaN rows under pandas 3.
  K, LABEL, SOURCES  extended in place with DGB / 'dGbyG' for the 4-source runs.
Each re-definition asserts that the text it edits occurs exactly once, so a
change to the shipped code fails loudly instead of being silently overridden.

RUNS (each writes results/grading/<run>/ with the shipped file set)
  baseline         3 sources, precedence openTECR > EQ > DGP > GC. Must
                   reproduce the shipped grades (checked: see BASELINE CHECK).
  dgb_last         4 sources, precedence openTECR > EQ > DGP > GC > DGB
  dgb_after_eq     4 sources, precedence openTECR > EQ > DGB > DGP > GC
  heldout_*        as the two above, but dGbyG's values AT THE ANCHORS come
                   from results/cv/anchor_heldout.tsv (the fold ensemble that
                   withheld the matching training reaction). Run automatically
                   when that file exists, or with --heldout.
Precedence only breaks ties between equally graded sources, so the two
precedence variants of a data set have identical per-source grades, p_ok
curves, pooling, CV tables and best_grade; they differ in best_source and in
which source's value the M12 predictors-alone scoring uses.

BASELINE CHECK (before dGbyG is added)
  * the patched copies reproduce the ORIGINAL functions exactly (3 sources);
  * live best_grade 4,073 / 14,475 / 8,790 = 27,338 (M12);
  * predictors-alone re-grading of the 363 live anchors 172 / 186 / 5, within
    2 kcal/mol 93.0% / 89.2% (M12; recipe of no committed script: drop the
    openTECR rows, best grade per reaction, ties by EQ > DGP > GC);
  * validation_cv identical to the committed grade_calibration.json;
  * reaction_grades.tsv identical to the committed one (best_grade,
    best_source, n_src, birge).

VALIDATION
  shipped CV       grade_thermo_sources.validate_cv as shipped: 5 folds x 4
                   reps, seed 11, over the 801 stereo-exact anchor RECORDS.
                   438 of these are obsolete duplicates of the 363 live
                   anchors, so copies of one reaction land in train and test.
  grouped CV       same folds/reps/seed but the 363 distinct live anchors are
                   permuted; each obsolete anchor joins the fold of the live
                   anchor named in its linked_reaction. Test reactions are
                   removed from the proxy tier as shipped. Scored on the live
                   anchors only (n = 4 x 363 per source at most) and, for
                   comparison, on all anchor records in the test folds.

CAVEATS
  * dGbyG's shipped model is trained on TECRDB, the same measurements the
    openTECR anchors come from, so for the shipped values the anchor tier of
    dGbyG's calibration and every anchor-based accuracy number is IN-SAMPLE
    for dGbyG (as it is for eQuilibrator and dGPredictor, whose training sets
    also overlap openTECR -- the paper's own caveat). Neither CV removes that:
    they refit only the grading curves. The heldout_* runs remove it for the
    anchor values; non-anchor reactions (proxy tier, grading) keep the shipped
    model's values. The held-out anchor sigmas come from a 90%-data fold
    ensemble, so dGbyG's p_ok curve is then fitted on a mixture of sigma
    scales (anchor weight 801 x 3 against 11,631 x 1 proxy rows) and applied
    to the shipped sigmas.
  * results/cv/anchor_heldout.tsv is made by cv_predict.py (anchors matched
    to their training reaction on dGbyG's KEGG structures, folds grouped on
    stereo-stripped keys); obsolete anchor records missing from it inherit
    their live anchor's value (see heldout_table).
  * --drop-sigma0 withholds dGbyG's sigma = 0.00 rows (treated as dGbyG
    declining). Those are a model blind spot -- graph-identical sides up to
    stereo, or unchanged 2-bond neighbourhoods -- not confidence, yet they
    take dGbyG's highest p_ok; run it with --out results/grading_nosigma0.
  * dGbyG's proxy reference, eQuilibrator at sigma <= 0.70, is itself fitted
    to TECRDB, so the proxy tier rewards agreement near shared training data.
  * sigma is the head-ensemble std as printed in reaction_predictions.tsv
    (2 dp). 509 records (428 live) print sigma 0.00; every one has dG = 0
    exactly -- reactions whose two sides dGbyG reads as the same graph
    (mostly epimerases, racemases, cis-trans isomerases; also glucan chain
    transfers) -- and they take dGbyG's highest p_ok (0.98) by construction.

Outputs (results/grading/):
  <run>/reaction_grades.tsv, source_grades.tsv, source_grades_wide.tsv,
  grade_frontier.tsv, grade_calibration.json   shipped formats (+ dGbyG);
      grade_calibration.json also carries validation_cv_grouped_{live,all};
      source_grades_heldout.tsv is not written: it is source_grades.tsv minus
      the openTECR rows (that is how the M12 re-grading below uses it);
      *dgb_after_eq/ has no source_grades.tsv: it equals *dgb_last/'s.
  baseline_check.tsv      expected vs reproduced numbers
  per_source_grades.tsv   per-source grade counts on live reactions, per run
  p_ok_models.tsv         n_anchor, n_proxy, anchor frac within tau, p_ok range
  p_ok_curves.tsv         every run's p_ok isotonic curve (x = sigma kcal/mol)
  validation.tsv          in-sample, shipped CV and grouped CV tables
  best_grade.tsv          live best_grade totals per run
  transition_<run>.tsv    baseline -> run best_grade on live reactions
  best_source.tsv         live best_source counts per run
  best_reason.tsv         reason of the best source by best grade (S03 table)
  m12_regrade.tsv         predictors-alone re-grading of the 363 live anchors
  m12_chosen_source.tsv   which predictor's value that re-grading scored
  other_source_shifts.tsv baseline -> 4-source per-source grade, EQ/DGP/GC
  other_source_reason_shifts.tsv  the same changes by (old reason -> new reason)
  n_src_<run>.tsv         baseline -> run number of graded sources per reaction
  dgbyg_gold_diagnostics.tsv  where the extra gold comes from; exact-zero dGbyG
  reaction_comparison.tsv one row per reaction: baseline vs dGbyG runs
  heldout_substitution.tsv  (heldout runs) how each anchor's value was set
  summary.json, grade_with_dgbyg.log

Run:
  ~/Documents/py_venv/bin/python -B analysis/modelseed/grade_with_dgbyg.py
      [--heldout | --no-heldout] [--heldout-file PATH] [--out DIR] [--strict]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")   # a 48-worker job shares this machine

import __future__
import argparse
import contextlib
import glob
import inspect
import json
import sys
import textwrap
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MSDB = Path(os.environ.get("MSDB_ROOT", REPO.parent / "ModelSEEDDatabase"))
SG = MSDB / "Scripts" / "Thermodynamics" / "SourceGrading"
COMMITTED = MSDB / "Biochemistry" / "Thermodynamics" / "SourceGrading" / "results" / "thermo_grades"
RES = HERE / "results"
PRED = RES / "reaction_predictions.tsv"
HELDOUT_DEFAULT = RES / "cv" / "anchor_heldout.tsv"

K3 = ["GC", "EQ", "DGP"]
K4 = ["GC", "EQ", "DGP", "DGB"]
GRADES = ["GOLD", "SILVER", "BRONZE"]
STATES = GRADES + ["ungraded"]

# the paper's published numbers the baseline must reproduce (M12, S03, and the
# committed grade_calibration.json)
EXPECT_BEST = {"GOLD": 4073, "SILVER": 14475, "BRONZE": 8790, "graded": 27338}
EXPECT_M12 = {"GOLD": (172, 0.930), "SILVER": (186, 0.892), "BRONZE": (5, None)}
# S03 Table (reason of the best source, live), lumped as the table lumps them
EXPECT_S03 = OrderedDict([
    (("GOLD", "measured"), (363, ["measured"])),
    (("GOLD", "self-certain"), (3710, ["self-certain", "self-certain-corroborated"])),
    (("SILVER", "self-certain disputed"), (31, ["self-certain-disputed"])),
    (("SILVER", "self-certain unpaired"), (370, ["self-certain-unpaired"])),
    (("SILVER", "self-confident"), (6361, ["self-confident", "self-confident-corroborated",
                                           "self-confident-unpaired"])),
    (("SILVER", "unconfident corroborated"), (7713, ["unconfident-corroborated"])),
    (("BRONZE", "self-confident disputed"), (79, ["self-confident-disputed"])),
    (("BRONZE", "unconfident disputed"), (2563, ["unconfident-disputed"])),
    (("BRONZE", "unconfident"), (6148, ["unconfident", "unconfident-unpaired"])),
])

O = G = None   # the SourceGrading modules, imported in main() after the env is set
ORIG = {}      # original function objects, kept for the fidelity check
STATE = {"dgb": None}
BASE_DB = None


# ------------------------------------------------------------------ logging
class Tee:
    def __init__(self, path):
        self.f = open(path, "w")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def show(title, df):
    print(f"\n=== {title} ===")
    print(df.to_string() if len(df) else "(empty)")


# ----------------------------------------------------------------- patching
def redefine(module, name, edits):
    """Re-define ``module.<name>`` from its own source with exact text edits.

    The new function is compiled into the module's own namespace, so it
    resolves every global (K, LABEL, load_db, fit_p_ok, ...) through the live
    module exactly like the original -- including later monkeypatches.
    """
    src = textwrap.dedent(inspect.getsource(getattr(module, name)))
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            raise RuntimeError(f"{module.__name__}.{name}: expected exactly one "
                               f"occurrence of {old!r}, found {n}")
        src = src.replace(old, new)
    code = compile(src, f"<{module.__name__}.{name} (grade_with_dgbyg)>", "exec",
                   flags=__future__.annotations.compiler_flag, dont_inherit=True)
    exec(code, module.__dict__)
    return getattr(module, name)


PROXY_OLD = 'proxy = {"GC": "EQ", "DGP": "EQ", "EQ": "DGP"}'
PROXY_NEW = 'proxy = {"GC": "EQ", "DGP": "EQ", "EQ": "DGP", "DGB": "EQ"}'


def load_db_with_dgbyg():
    """The original load_db frame (cached), plus dGbyG columns in 4-source runs."""
    db = BASE_DB.copy()
    if "DGB" in G.K:
        db = db.merge(STATE["dgb"], on="rxn", how="left", validate="one_to_one")
        assert len(db) == len(BASE_DB)
    return db


def install_patches():
    ORIG.update(load_db=O.load_db, fit_error_models=O.fit_error_models,
                fit_p_ok=G.fit_p_ok, build=G.build)
    fem = redefine(O, "fit_error_models", [(PROXY_OLD, PROXY_NEW)])
    G.fit_error_models = fem          # G imported it by value
    redefine(G, "fit_p_ok", [(PROXY_OLD, PROXY_NEW)])
    redefine(G, "build", [
        ('PRECEDENCE = ["openTECR", "EQ", "DGP", "GC"]',
         'PRECEDENCE = list(GRADE_PRECEDENCE)'),
        ('wide["best_source"] = ranks.idxmin(axis=1)',
         'wide["best_source"] = ranks.fillna(np.inf).idxmin(axis=1)'),
    ])
    O.load_db = G.load_db = load_db_with_dgbyg


def configure(sources, order, dgb=None):
    """Point the SourceGrading modules at one source set / precedence."""
    G.K[:] = sources
    if "DGB" in sources:
        O.SOURCES["dGbyG"] = "DGB"
        G.LABEL["DGB"] = "dGbyG"
        assert dgb is not None
    else:
        O.SOURCES.pop("dGbyG", None)
    G.GRADE_PRECEDENCE = ["openTECR"] + list(order)
    STATE["dgb"] = dgb


@contextlib.contextmanager
def original_functions():
    """Temporarily restore the shipped fit_error_models / fit_p_ok / build.

    The shipped build() needs pandas<3 for idxmin on all-NaN rows; a scoped
    shim gives the pandas-2 result (NaN) so the original code can run.
    """
    saved = dict(fem=G.fit_error_models, ofem=O.fit_error_models,
                 fpk=G.fit_p_ok, build=G.build)
    orig_idxmin = pd.DataFrame.idxmin

    def idxmin(self, axis=0, skipna=True, numeric_only=False):
        if axis in (1, "columns"):
            allna = self.isna().all(axis=1)
            res = orig_idxmin(self.fillna(np.inf), axis=1, skipna=skipna,
                              numeric_only=numeric_only).astype(object)
            return res.where(~allna, np.nan)
        return orig_idxmin(self, axis=axis, skipna=skipna, numeric_only=numeric_only)

    G.fit_error_models = O.fit_error_models = ORIG["fit_error_models"]
    G.fit_p_ok, G.build = ORIG["fit_p_ok"], ORIG["build"]
    pd.DataFrame.idxmin = idxmin
    try:
        yield
    finally:
        pd.DataFrame.idxmin = orig_idxmin
        G.fit_error_models, O.fit_error_models = saved["fem"], saved["ofem"]
        G.fit_p_ok, G.build = saved["fpk"], saved["build"]


# ------------------------------------------------------------------- inputs
def reaction_meta():
    """id -> (is_obsolete, linked_reaction) from the same JSON load_db reads."""
    obs, linked = {}, {}
    for path in sorted(glob.glob(str(O.BIOCHEM / "reaction_*.json"))):
        for e in json.load(open(path)):
            obs[e["id"]] = e.get("is_obsolete") in (1, "1")
            linked[e["id"]] = str(e.get("linked_reaction") or "")
    return obs, linked


def anchor_groups(obs, linked):
    """The 801 anchor records -> their live anchor (the CV / heldout grouping)."""
    tec = G.load_opentecr()
    a = tec[tec.match_tier.eq("stereo_exact") & tec.opentecr_dg.notna()
            & tec.rxn.isin(set(BASE_DB.rxn))]
    live = sorted(r for r in a.rxn if not obs[r])
    live_set = set(live)
    group = {r: r for r in live}
    for r in a.rxn:
        if r in live_set:
            continue
        cand = [x for x in linked[r].split(";") if x.strip() in live_set]
        if len(cand) != 1:
            raise RuntimeError(f"obsolete anchor {r} links to {len(cand)} live anchors")
        group[r] = cand[0].strip()
    tdg = dict(zip(a.rxn, a.opentecr_dg))
    return group, live, tdg


DROP_SIGMA0 = False


def dgbyg_table():
    """dGbyG's primary estimate, admitted exactly as load_db admits a source."""
    rp = pd.read_csv(PRED, sep="\t", usecols=["rxn", "status", "dG_kcal", "sd_kcal", "op"],
                     dtype={"op": str}, keep_default_na=False, na_values=[""])
    ok = rp[rp.status.eq("ok")].copy()
    t = pd.DataFrame({"rxn": ok.rxn,
                      "dg_DGB": pd.to_numeric(ok.dG_kcal, errors="coerce"),
                      "sig_DGB": pd.to_numeric(ok.sd_kcal, errors="coerce").abs(),
                      "op_DGB": ok.op})
    good = np.isfinite(t.dg_DGB) & np.isfinite(t.sig_DGB) & (t.dg_DGB != O.GC_SENTINEL_DG)
    if DROP_SIGMA0:
        good &= t.sig_DGB > 0
    info = {"status_ok": int(len(ok)), "admitted": int(good.sum()),
            "in_load_db": int(t[good].rxn.isin(set(BASE_DB.rxn)).sum()),
            "sigma_zero_2dp": int((t[good].sig_DGB == 0).sum()),
            "sigma_quantiles": {str(q): float(t[good].sig_DGB.quantile(q))
                                for q in (0, .05, .25, .5, .75, .95, 1)}}
    return t[good].reset_index(drop=True), info


def heldout_table(base, path, group, obs, tdg):
    """Replace dGbyG's anchor values by the held-out fold predictions.

    A record listed in the file takes its own value. An obsolete anchor that is
    not listed takes its linked live anchor's held-out value, with the sign of
    the orientation between the two records -- read off the shipped dGbyG
    values (equal -> same orientation, negated -> reversed; when both are ~0
    the openTECR values decide). Anything that cannot be mapped loses its
    dGbyG value at that anchor rather than keep an in-sample one.
    """
    h = pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""])
    hv = {r: (float(d), abs(float(s))) for r, d, s in
          zip(h.rxn, h.heldout_dG_kcal, h.heldout_sd_kcal)}
    t = base.set_index("rxn").copy()
    rows = []
    tol = 0.015   # dG_kcal is printed to 2 dp
    for r in sorted(group):
        g = group[r]
        old = t.dg_DGB.get(r, np.nan)
        how, val = None, (np.nan, np.nan)
        if r in hv:
            how, val = "file", hv[r]
        elif obs[r] and g in hv:
            a, b = t.dg_DGB.get(r, np.nan), t.dg_DGB.get(g, np.nan)
            same, rev = abs(a - b) <= tol, abs(a + b) <= tol
            if same and rev:
                same = abs(tdg[r] - tdg[g]) <= abs(tdg[r] + tdg[g])
                rev = not same
            if same:
                how, val = "linked_same", hv[g]
            elif rev:
                how, val = "linked_reversed", (-hv[g][0], hv[g][1])
            else:
                how = "unmapped"
        else:
            how = "not_in_file"
        if r in t.index:
            t.loc[r, ["dg_DGB", "sig_DGB"]] = val
            t.loc[r, "op_DGB"] = None   # the stored operator came from the shipped value
        rows.append({"rxn": r, "live": not obs[r], "group": g, "how": how,
                     "shipped_dg": old, "heldout_dg": val[0], "heldout_sd": val[1],
                     "opentecr_dg": tdg[r]})
    t = t[t.dg_DGB.notna() & t.sig_DGB.notna()].reset_index()
    return t, pd.DataFrame(rows)


# --------------------------------------------------------------- validation
def cv_rows(pooled, note):
    rows = []
    for k in G.K:
        for v in (G.GOLD, G.SILVER, G.BRONZE):
            e = np.array(pooled[(k, v)])
            r = {"source": G.LABEL[k], "grade": G.NAME[v], "n": int(len(e))}
            if len(e):
                r.update(median_abs_err=float(np.median(e)), mean_abs_err=float(e.mean()),
                         frac_within_1=float((e <= 1).mean()), frac_within_2=float((e <= 2).mean()),
                         p90_abs_err=float(np.percentile(e, 90)), note=note)
            rows.append(r)
    return rows


def validate_cv_grouped(db, tec, veto_eq, group, live_set, n_folds=5, n_reps=4, seed=11):
    """grade_thermo_sources.validate_cv with folds over the 363 live anchors.

    Identical refitting (fit_error_models / fit_p_ok on the training anchors,
    test reactions removed from the proxy tier, pooling and grading over the
    whole database); only the split differs, so no reaction is in both train
    and test through an obsolete duplicate. Returns (live-scored, all-scored).
    """
    m0 = tec["match_tier"].eq("stereo_exact")
    anchor_idx = db.index[m0.to_numpy() & db["opentecr_dg"].notna()]
    grp = db.loc[anchor_idx, "rxn"].map(group)
    assert grp.notna().all()
    is_live = db.loc[anchor_idx, "rxn"].isin(live_set).to_numpy()
    groups = np.array(sorted(set(grp)))
    rng = np.random.default_rng(seed)
    pooled = {s: {(k, v): [] for k in G.K for v in (G.GOLD, G.SILVER, G.BRONZE)}
              for s in ("live", "all")}
    err = {k: (db[f"dg_{k}"] - db["opentecr_dg"]).abs() for k in G.K}
    for _ in range(n_reps):
        gfolds = np.array_split(rng.permutation(groups), n_folds)
        for f in range(n_folds):
            in_te = grp.isin(set(gfolds[f])).to_numpy()
            te, tr = anchor_idx[in_te], anchor_idx[~in_te]
            db_tr = db[~db.rxn.isin(set(db.loc[te, "rxn"]))]
            eh = G.predict_error(db, G.fit_error_models(
                db.loc[tr].rename(columns={"opentecr_dg": "opentecr"}), db_tr))
            pk = G.predict_p_ok(db, G.fit_p_ok(db.loc[tr], db_tr), veto_eq)
            gr, _ = G.grade_predictors(db, pk, G.pool_sources(db, eh, pk), tec)
            for scope, idx in (("all", te), ("live", anchor_idx[in_te & is_live])):
                for k in G.K:
                    for v in (G.GOLD, G.SILVER, G.BRONZE):
                        g = gr[k].loc[idx]
                        e = err[k].loc[idx]
                        pooled[scope][(k, v)].extend(e[(g == v) & e.notna()].tolist())
    note = f"{n_folds}-fold x {n_reps} reps grouped by live anchor; "
    return (cv_rows(pooled["live"], note + "scored on the 363 live anchors"),
            cv_rows(pooled["all"], note + "scored on all anchor records"))


# ------------------------------------------------------------------ one run
def run(name, sources, order, dgb, outd, group, live_set, twin=None):
    """build() + CV for one configuration; write the shipped file set to outd/name.

    ``twin`` is an earlier run of the same data under another precedence: its
    per-source grades must be identical (asserted), so its CV tables are reused
    and its source_grades.tsv is not duplicated.
    """
    configure(sources, order, dgb)
    d = outd / name
    d.mkdir(parents=True, exist_ok=True)
    G.OUT = d
    print(f"\n################ run {name}: sources {sources}, precedence "
          f"{G.GRADE_PRECEDENCE}")
    t0 = time.time()
    long, wide, calib, ctx = G.build()
    db, tec, veto = ctx
    calib["grade_with_dgbyg"] = {"run": name, "sources": list(sources),
                                 "precedence": G.GRADE_PRECEDENCE}
    out = {"long": long, "wide": wide, "calib": calib, "db": db, "tec": tec, "veto": veto,
           "sources": list(sources), "order": list(order)}
    cv_kinds = ("validation_cv", "validation_cv_grouped_live", "validation_cv_grouped_all")
    if twin is None:
        calib["validation_cv"] = G.validate_cv(db, tec, veto)
        calib["validation_cv_grouped_live"], calib["validation_cv_grouped_all"] = \
            validate_cv_grouped(db, tec, veto, group, live_set)
        long.to_csv(d / "source_grades.tsv", sep="\t", index=False, float_format="%.4f")
    else:
        same_grades(twin, out)
        tname = twin["calib"]["grade_with_dgbyg"]["run"]
        for kind in cv_kinds:
            calib[kind] = twin["calib"][kind]
        calib["grade_with_dgbyg"]["source_grades"] = f"identical to ../{tname}/source_grades.tsv"
        (d / "source_grades.tsv").unlink(missing_ok=True)
        print(f"  {name}: per-source grades identical to {tname} (precedence changes "
              f"best_source only); CV tables reused, source_grades.tsv not duplicated")
    wide.to_csv(d / "source_grades_wide.tsv", sep="\t", index=False, float_format="%.4f")
    G.frontier(long).to_csv(d / "grade_frontier.tsv", sep="\t", index=False)
    (d / "grade_calibration.json").write_text(json.dumps(calib, indent=1))
    print(f"  run {name}: {time.time() - t0:.1f} s, wrote {d}")
    return out


# ---------------------------------------------------------------- summaries
def per_source_counts(res, live):
    w = res["wide"][res["wide"].rxn.isin(live)]
    rows = []
    for k in res["sources"]:
        c = w[f"grade_{k}"].value_counts()
        rows.append({"source": G.LABEL[k], **{g: int(c.get(g, 0)) for g in GRADES},
                     "graded": int(w[f"grade_{k}"].notna().sum())})
    return rows


def best_counts(res, live):
    w = res["wide"][res["wide"].rxn.isin(live)]
    c = w.best_grade.value_counts()
    return {**{g: int(c.get(g, 0)) for g in GRADES}, "graded": int(w.best_grade.notna().sum())}


def predictors_alone(res, live_anchors, order):
    """M12: best grade per anchored reaction with openTECR withheld.

    source_grades_heldout.tsv is source_grades.tsv minus the openTECR rows, so
    the openTECR rows are simply dropped; ties at the best grade go to the
    first source in ``order``.
    """
    pr = res["long"][res["long"].source != "openTECR"].copy()
    rank = {G.LABEL[k]: i for i, k in enumerate(order)}
    pr["_g"] = pr.grade.map({g: i for i, g in enumerate(GRADES)})
    pr["_p"] = pr.source.map(rank)
    best = pr.sort_values(["rxn", "_g", "_p"]).groupby("rxn", as_index=False).first()
    best = best[best.rxn.isin(live_anchors)].copy()
    truth = dict(zip(res["db"].rxn, res["db"].opentecr_dg))
    best["err"] = (best.dg - best.rxn.map(truth)).abs()
    return best


def m12_rows(best, run_name, order):
    rows = []
    for g in GRADES + ["ALL"]:
        x = best if g == "ALL" else best[best.grade == g]
        rows.append({"run": run_name, "precedence": ">".join(order), "grade": g,
                     "n": len(x), "within_2": int((x.err <= 2).sum()),
                     "frac_within_2": float((x.err <= 2).mean()) if len(x) else np.nan,
                     "median_abs_err": float(x.err.median()) if len(x) else np.nan})
    return rows


def best_reason(res, live):
    w = res["wide"][res["wide"].rxn.isin(live) & res["wide"].best_grade.notna()]
    w = w[["rxn", "best_grade", "best_source"]].copy()
    w["source"] = w.best_source.map(G.LABEL)
    lg = res["long"][["rxn", "source", "reason"]]
    m = w.merge(lg, on=["rxn", "source"], how="left")
    return m.groupby(["best_grade", "reason"]).size()


def validation_frame(res, run_name):
    out = []
    for kind in ("validation", "validation_cv", "validation_cv_grouped_live",
                 "validation_cv_grouped_all"):
        for r in res["calib"].get(kind, []):
            out.append({"run": run_name, "kind": {"validation": "in-sample (801 records)",
                                                  "validation_cv": "shipped CV (801 records)",
                                                  "validation_cv_grouped_live": "grouped CV (363 live)",
                                                  "validation_cv_grouped_all": "grouped CV (801 records)"}[kind],
                        **{c: r.get(c, np.nan) for c in ("source", "grade", "n", "median_abs_err",
                                                         "mean_abs_err", "frac_within_1",
                                                         "frac_within_2", "p90_abs_err")}})
    return out


def pok_rows(res, run_name):
    rows, curves = [], []
    for k, m in res["calib"]["p_ok_models"].items():
        grid = np.linspace(0.0, m["x"][-1] * 1.01, 200001)
        pg = np.interp(grid, m["x"], m["y"], left=m["y"][0], right=m["y"][-1])
        smax = {t: (float(grid[pg >= t].max()) if (pg >= t).any() else np.nan)
                for t in (G.P_GOLD, G.P_SILVER)}
        rows.append({"run": run_name, "source": G.LABEL.get(k, k), "n_anchor": m["n_anchor"],
                     "n_proxy": m["n_proxy"], "anchor_frac_within_tau": m["anchor_frac_within_tau"],
                     "p_ok_at_min_sigma": m["y"][0], "p_ok_at_max_sigma": m["y"][-1],
                     "sigma_min": m["x"][0], "sigma_max": m["x"][-1], "n_steps": len(m["x"]),
                     "sigma_max_p_ok_ge_0.90": smax[G.P_GOLD],
                     "sigma_max_p_ok_ge_0.70": smax[G.P_SILVER]})
        curves += [{"run": run_name, "source": G.LABEL.get(k, k), "sigma_kcal": x, "p_ok": y}
                   for x, y in zip(m["x"], m["y"])]
    return rows, curves


def transition(a, b, live):
    wa = a["wide"].set_index("rxn").best_grade
    wb = b["wide"].set_index("rxn").best_grade
    idx = [r for r in wa.index if r in live]
    t = pd.crosstab(wa.reindex(idx).fillna("ungraded").rename("baseline"),
                    wb.reindex(idx).fillna("ungraded").rename("with_dGbyG"))
    return t.reindex(index=STATES, columns=STATES).fillna(0).astype(int)


def same_grades(a, b):
    """Two runs of one data set must agree on everything but best_source."""
    la = a["long"].reset_index(drop=True)
    lb = b["long"].reset_index(drop=True)
    pd.testing.assert_frame_equal(la, lb)
    cols = [c for c in a["wide"].columns if c != "best_source"]
    pd.testing.assert_frame_equal(a["wide"][cols], b["wide"][cols])


# --------------------------------------------------------------------- main
def main():
    global O, G, BASE_DB
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RES / "grading"), help="output directory")
    hg = ap.add_mutually_exclusive_group()
    hg.add_argument("--heldout", action="store_true",
                    help="also run with dGbyG's anchor values from --heldout-file "
                         "(default: run it whenever that file exists)")
    hg.add_argument("--no-heldout", action="store_true", help="skip the held-out runs")
    ap.add_argument("--heldout-file", default=str(HELDOUT_DEFAULT))
    ap.add_argument("--strict", action="store_true", help="stop if the baseline does not reproduce")
    ap.add_argument("--drop-sigma0", action="store_true",
                    help="treat dGbyG rows with sigma 0.00 as declined (sensitivity run)")
    args = ap.parse_args()
    global DROP_SIGMA0
    DROP_SIGMA0 = args.drop_sigma0

    outd = Path(args.out).resolve()
    outd.mkdir(parents=True, exist_ok=True)
    for bad in (MSDB.resolve(), (REPO.parent / "MSDB_thermo_paper").resolve()):
        if outd == bad or bad in outd.parents:
            raise SystemExit(f"refusing to write under {bad}")
    sys.stdout = Tee(outd / "grade_with_dgbyg.log")
    # MUST precede the SourceGrading imports: they read these at import time,
    # and the defaults point into ModelSEEDDatabase.
    os.environ["THERMO_GRADING_OUT"] = str(outd)
    os.environ["GRADES_OUT"] = str(outd / "baseline")
    os.environ["EQDGP_OUT"] = str(outd / "eq_vs_dgp")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SG))
    import optimize_thermo_source_assignment as O_
    import grade_thermo_sources as G_
    O, G = O_, G_
    assert Path(G.ANALYSIS_DIR) == outd and Path(O.OUT) == outd / "eq_vs_dgp"
    print(f"SourceGrading code {SG}\nMSDB_ROOT {O.MSDB_ROOT}\nout {outd}")

    t0 = time.time()
    BASE_DB = O.load_db()          # the shipped loader, 3 sources, cached
    print(f"load_db: {len(BASE_DB)} rows ({time.time() - t0:.1f} s)")
    obs, linked = reaction_meta()
    live = {r for r in BASE_DB.rxn if not obs[r]}
    group, live_anchors, tdg = anchor_groups(obs, linked)
    live_set = set(live_anchors)
    print(f"live reactions in load_db {len(live)}, obsolete {len(BASE_DB) - len(live)}; "
          f"anchor records {len(group)} -> {len(live_anchors)} live anchors")
    install_patches()
    summary = OrderedDict(n_load_db=len(BASE_DB), n_live=len(live), n_anchor_records=len(group),
                          n_live_anchors=len(live_anchors))

    # ================================================== 1. baseline (3 sources)
    configure(K3, ["EQ", "DGP", "GC"])
    G.OUT = outd / "baseline"
    G.OUT.mkdir(parents=True, exist_ok=True)
    print("\n################ fidelity: ORIGINAL shipped functions, 3 sources")
    with original_functions():
        o_long, o_wide, o_calib, _ = G.build()
    res = {"baseline": run("baseline", K3, ["EQ", "DGP", "GC"], None, outd, group, live_set)}
    b = res["baseline"]
    pd.testing.assert_frame_equal(o_long.reset_index(drop=True), b["long"].reset_index(drop=True))
    obj = {"best_source": object}    # the shim yields object, pandas 3 a str dtype
    pd.testing.assert_frame_equal(o_wide.astype(obj), b["wide"].astype(obj))
    assert o_calib["p_ok_models"] == b["calib"]["p_ok_models"]
    print("fidelity: patched copies == shipped functions on 3 sources (long, wide, p_ok) OK")

    checks = []

    def check(metric, expected, got, ok):
        checks.append({"metric": metric, "expected": expected, "got": got, "ok": bool(ok)})

    bc = best_counts(b, live)
    for g in GRADES + ["graded"]:
        check(f"live best_grade {g}", EXPECT_BEST[g], bc[g], bc[g] == EXPECT_BEST[g])
    m12 = predictors_alone(b, live_set, ["EQ", "DGP", "GC"])
    for g in GRADES:
        x = m12[m12.grade == g]
        n_exp, f_exp = EXPECT_M12[g]
        check(f"M12 predictors-alone {g} n", n_exp, len(x), len(x) == n_exp)
        if f_exp is not None:
            f = float((x.err <= 2).mean())
            check(f"M12 predictors-alone {g} within 2 kcal", f_exp, round(f, 4), round(f, 3) == f_exp)
    cj = json.load(open(COMMITTED / "grade_calibration.json"))
    for r_exp, r_got in zip(cj.get("validation_cv", []), b["calib"]["validation_cv"]):
        same = (r_exp["source"], r_exp["grade"], r_exp["n"]) == (r_got["source"], r_got["grade"], r_got["n"])
        dm = abs(r_exp.get("median_abs_err", 0) - r_got.get("median_abs_err", 0))
        df2 = abs(r_exp.get("frac_within_2", 0) - r_got.get("frac_within_2", 0))
        check(f"validation_cv {r_exp['source']} {r_exp['grade']} (n, median, <=2)",
              f"{r_exp['n']} {r_exp.get('median_abs_err', np.nan):.3f} {r_exp.get('frac_within_2', np.nan):.3f}",
              f"{r_got['n']} {r_got.get('median_abs_err', np.nan):.3f} {r_got.get('frac_within_2', np.nan):.3f}",
              same and dm < 1e-9 and df2 < 1e-9)
    for k, m in cj["p_ok_models"].items():
        mg = b["calib"]["p_ok_models"][k]
        check(f"p_ok model {k} (n_anchor, n_proxy)", f"{m['n_anchor']} {m['n_proxy']}",
              f"{mg['n_anchor']} {mg['n_proxy']}",
              (m["n_anchor"], m["n_proxy"]) == (mg["n_anchor"], mg["n_proxy"])
              and np.allclose(m["y"], mg["y"]) and np.allclose(m["x"], mg["x"]))
    rg = pd.read_csv(COMMITTED / "reaction_grades.tsv", sep="\t", low_memory=False)
    mine = pd.read_csv(outd / "baseline" / "reaction_grades.tsv", sep="\t", low_memory=False)
    j = rg.merge(mine, on="rxn", how="outer", suffixes=("_c", "_m"), indicator=True)
    check("reaction_grades.tsv rows (committed vs reproduced)", len(rg), len(mine),
          (j._merge == "both").all())
    for c in ("best_grade", "best_source", "n_src"):
        mis = int((j[f"{c}_c"].fillna("NA").astype(str) != j[f"{c}_m"].fillna("NA").astype(str)).sum())
        check(f"reaction_grades.tsv {c} mismatches", 0, mis, mis == 0)
    bd = float(np.nanmax(np.abs(j.birge_c - j.birge_m)))
    check("reaction_grades.tsv max |birge diff|", 0.0, bd, bd < 1e-9)
    br = best_reason(b, live)
    for (g, label), (n_exp, reasons) in EXPECT_S03.items():
        n = int(sum(br.get((g, r), 0) for r in reasons))
        check(f"S03 table {g} {label}", n_exp, n, n == n_exp)
    chk = pd.DataFrame(checks)
    chk.to_csv(outd / "baseline_check.tsv", sep="\t", index=False)
    show("BASELINE CHECK (3 sources, before dGbyG)", chk)
    summary["baseline_reproduced"] = bool(chk.ok.all())
    if not chk.ok.all():
        print("\n!!! BASELINE DID NOT REPRODUCE -- see baseline_check.tsv")
        if args.strict:
            raise SystemExit(1)

    # ================================================== 2. with dGbyG
    dgb, dgb_info = dgbyg_table()
    summary["dgbyg_input"] = dgb_info
    print(f"\ndGbyG rows admitted: {dgb_info}")
    variants = [("dgb_last", ["EQ", "DGP", "GC", "DGB"]),
                ("dgb_after_eq", ["EQ", "DGB", "DGP", "GC"])]
    data_sets = [("", dgb)]
    hpath = Path(args.heldout_file)
    if not args.no_heldout:
        if hpath.exists():
            dgb_h, sub = heldout_table(dgb, hpath, group, obs, tdg)
            sub.to_csv(outd / "heldout_substitution.tsv", sep="\t", index=False, float_format="%.4f")
            show("held-out substitution of dGbyG anchor values (records)",
                 sub.groupby(["live", "how"]).size().rename("n").reset_index())
            summary["heldout_substitution"] = sub.groupby("how").size().to_dict()
            lv = sub[sub.live]
            e_in = (lv.shipped_dg - lv.opentecr_dg).abs()
            e_ho = (lv.heldout_dg - lv.opentecr_dg).abs()
            summary["heldout_vs_shipped_live_anchors"] = {
                "n": int(e_ho.notna().sum()),
                "shipped_median_abs_err": float(e_in[e_ho.notna()].median()),
                "heldout_median_abs_err": float(e_ho.median()),
                "shipped_within_2": float((e_in[e_ho.notna()] <= 2).mean()),
                "heldout_within_2": float((e_ho.dropna() <= 2).mean())}
            print("live anchors, shipped vs held-out dGbyG:", summary["heldout_vs_shipped_live_anchors"])
            data_sets.append(("heldout_", dgb_h))
        else:
            msg = (f"held-out anchor file {hpath} not found -- heldout runs skipped "
                   "(make it with cv_predict.py)")
            print(("\n!!! " if args.heldout else "\nnote: ") + msg)
            summary["heldout_substitution"] = msg

    for prefix, table in data_sets:
        first = None
        for vname, order in variants:
            name = prefix + vname
            res[name] = run(name, K4, order, table, outd, group, live_set, twin=first)
            first = first or res[name]

    # ================================================== 3. summaries
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)
    names = list(res)
    ps, pk, pc, val, m12r, m12c, bsrc, bg, reasons = [], [], [], [], [], [], [], [], {}
    for n in names:
        r = res[n]
        ps += [{"run": n, **x} for x in per_source_counts(r, live)]
        a, c = pok_rows(r, n)
        pk += a
        pc += c
        val += validation_frame(r, n)
        order = r["order"]
        best = predictors_alone(r, live_set, order)
        m12r += m12_rows(best, n, order)
        cs = best.groupby(["grade", "source"]).agg(n=("err", "size"),
                                                   within_2=("err", lambda e: int((e <= 2).sum())))
        m12c += [{"run": n, "grade": g, "source": s, **v} for (g, s), v in cs.to_dict("index").items()]
        w = r["wide"][r["wide"].rxn.isin(live)]
        bsrc.append({"run": n, **w.best_source.map(G.LABEL).value_counts().to_dict()})
        bg.append({"run": n, **best_counts(r, live)})
        reasons[n] = best_reason(r, live)

    tabs = OrderedDict()
    tabs["per_source_grades"] = pd.DataFrame(ps)
    tabs["p_ok_models"] = pd.DataFrame(pk)
    tabs["validation"] = pd.DataFrame(val)
    tabs["best_grade"] = pd.DataFrame(bg)
    tabs["best_source"] = pd.DataFrame(bsrc).fillna(0)
    tabs["m12_regrade"] = pd.DataFrame(m12r)
    tabs["m12_chosen_source"] = pd.DataFrame(m12c)
    tabs["best_reason"] = pd.DataFrame(reasons).fillna(0).astype(int).reset_index().rename(
        columns={"level_0": "best_grade", "level_1": "reason"})
    pd.DataFrame(pc).to_csv(outd / "p_ok_curves.tsv", sep="\t", index=False)

    # how adding dGbyG moves the OTHER sources (n_src, Birge, z all change)
    shifts = []
    bw = res["baseline"]["wide"].set_index("rxn")
    lv = sorted(live)
    for n in names:
        if n == "baseline" or "after_eq" in n:
            continue
        w = res[n]["wide"].set_index("rxn")
        for k in K3:
            t = pd.crosstab(bw.loc[lv, f"grade_{k}"].fillna("ungraded").rename("baseline"),
                            w.loc[lv, f"grade_{k}"].fillna("ungraded").rename("with_dGbyG"))
            for (g0, g1), v in t.stack().items():
                shifts.append({"run": n, "source": G.LABEL[k], "baseline": g0, "with_dGbyG": g1, "n": int(v)})
        ns = pd.crosstab(bw.loc[lv, "n_src"].rename("n_src_baseline"), w.loc[lv, "n_src"].rename("n_src_with_dGbyG"))
        tabs[f"n_src_{n}"] = ns
        only = w.loc[lv]
        dgb_only = int((only.grade_DGB.notna() & only[[f"grade_{k}" for k in K3]].isna().all(axis=1)).sum())
        summary[f"{n}_live_dgbyg_only_graded"] = dgb_only
        summary[f"{n}_live_dgbyg_only_grades"] = only.loc[
            only.grade_DGB.notna() & only[[f"grade_{k}" for k in K3]].isna().all(axis=1),
            "grade_DGB"].value_counts().to_dict()
        moved = (bw.loc[lv, "birge"].round(6).fillna(-1) != w.loc[lv, "birge"].round(6).fillna(-1))
        summary[f"{n}_live_birge_changed"] = int(moved.sum())
        summary[f"{n}_live_n_src_changed"] = int((bw.loc[lv, "n_src"] != w.loc[lv, "n_src"]).sum())
    tabs["other_source_shifts"] = pd.DataFrame(shifts)

    # ... and the mechanism: per-source grade changes by (old reason -> new reason)
    rs = []
    base_long = res["baseline"]["long"]
    base_long = base_long[base_long.rxn.isin(live) & (base_long.source != "openTECR")]
    for n in names:
        if n == "baseline" or "after_eq" in n:
            continue
        nl = res[n]["long"]
        nl = nl[nl.rxn.isin(live) & nl.source.isin(base_long.source.unique())]
        m = base_long[["rxn", "source", "grade", "reason"]].merge(
            nl[["rxn", "source", "grade", "reason"]], on=["rxn", "source"], suffixes=("_baseline", "_with_dGbyG"))
        m = m[m.grade_baseline != m.grade_with_dGbyG]
        t = m.groupby(["source", "grade_baseline", "grade_with_dGbyG", "reason_baseline",
                       "reason_with_dGbyG"]).size().rename("n").reset_index()
        rs.append(t.assign(run=n).sort_values(["source", "n"], ascending=[True, False]))
    tabs["other_source_reason_shifts"] = pd.concat(rs, ignore_index=True)[
        ["run", "source", "grade_baseline", "grade_with_dGbyG", "reason_baseline", "reason_with_dGbyG", "n"]]

    # where the extra gold comes from, and dGbyG's exact-zero predictions
    diag = []
    for n in names:
        if n == "baseline" or "after_eq" in n:
            continue
        w = res[n]["wide"].set_index("rxn").loc[lv]
        lg = res[n]["long"]
        d = lg[(lg.source == "dGbyG") & lg.rxn.isin(live)].set_index("rxn")
        new_gold = (w.best_grade == "GOLD") & (bw.loc[lv, "best_grade"] != "GOLD")
        others_gold = w[[f"grade_{k}" for k in K3]].eq("GOLD").any(axis=1)
        zero = (d.sigma == 0) & (d.dg.abs() < 0.005)
        diag.append(OrderedDict(
            run=n, live_gold_baseline=int((bw.loc[lv, "best_grade"] == "GOLD").sum()),
            live_gold_with_dgbyg=int((w.best_grade == "GOLD").sum()),
            newly_gold=int(new_gold.sum()),
            newly_gold_only_dgbyg_gold=int((new_gold & ~others_gold).sum()),
            newly_gold_other_source_gold=int((new_gold & others_gold).sum()),
            lost_gold=int(((w.best_grade != "GOLD") & (bw.loc[lv, "best_grade"] == "GOLD")).sum()),
            dgbyg_gold=int((d.grade == "GOLD").sum()),
            dgbyg_gold_sigma0_dg0=int((zero & (d.grade == "GOLD")).sum()),
            dgbyg_sigma0_dg0=int(zero.sum()),
            newly_gold_only_dgbyg_sigma0_dg0=int((new_gold & ~others_gold).reindex(d.index[zero]).fillna(False).sum()),
            dgbyg_only_graded=int((w.grade_DGB.notna() & w[[f"grade_{k}" for k in K3]].isna().all(axis=1)).sum())))
    tabs["dgbyg_gold_diagnostics"] = pd.DataFrame(diag)

    for n in names:
        if n != "baseline":
            tabs[f"transition_{n}"] = transition(res["baseline"], res[n], live)

    for name, df in tabs.items():
        df.to_csv(outd / f"{name}.tsv", sep="\t", index=name.startswith(("transition_", "n_src_")),
                  float_format="%.4f")
        show(name, df)

    # one row per reaction, for downstream use
    rc = res["baseline"]["wide"][["rxn", "is_transport", "n_src", "best_grade", "best_source"]].rename(
        columns={"n_src": "n_src_baseline", "best_grade": "best_grade_baseline",
                 "best_source": "best_source_baseline"})
    rc.insert(1, "live", rc.rxn.isin(live).astype(int))
    for n in names:
        if n == "baseline":
            continue
        w = res[n]["wide"]
        add = w[["rxn", "best_grade", "best_source"]].rename(
            columns={"best_grade": f"best_grade_{n}", "best_source": f"best_source_{n}"})
        if "after_eq" not in n:
            add = add.join(w[["n_src", "grade_DGB"]].rename(
                columns={"n_src": f"n_src_{n}", "grade_DGB": f"grade_DGB_{n}"}))
        rc = rc.merge(add, on="rxn", how="left")
    rc.to_csv(outd / "reaction_comparison.tsv", sep="\t", index=False)

    summary["tables"] = {k: v.reset_index().to_dict("records") if k.startswith(("transition_", "n_src_"))
                         else v.to_dict("records") for k, v in tabs.items()}
    (outd / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"\nwrote {outd}  ({time.time() - t0:.0f} s total)")


if __name__ == "__main__":
    main()
