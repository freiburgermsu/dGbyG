"""Re-checks behind the October 2026 lit-review deck (Lit review_research presentation X).

Four numbers the deck quotes that are not printed elsewhere in results/:

  A. dGbyG's grouped 10-fold CV over all 671 training rows, in the units of
     the paper's Table S4 (kJ/mol), from results/cv/training_heldout.tsv.
  B. Anchor accuracy on the 348 non-degenerate live anchors (the PR #323
     population): every source including group contribution, the truth-choice
     comparison (raw anchors vs the same measurements transformed to pH 7 /
     I 0.25 / pMg 14) and the novelty classes of anchor_novelty.py.
  C. eQuilibrator's direction accuracy by quartile of its own sigma on the
     801 anchor records, under the README protocol and under the shipped rule
     sets against the RI-rule truth (results/direction/anchor_ops.tsv).
  D. optimize_thermo_source_assignment.py's held-out MAGNITUDE validation
     (argmin expected error vs always-one-source), which the script runs on a
     single 70/30 split, repeated over 20 splits. Reads the ModelSEEDDatabase
     given by --msdb; writes nothing there (the script's outputs are redirected
     to a temporary directory).

Writes results/deck_checks/deck_checks.txt.

    ~/Documents/py_venv/bin/python -B analysis/modelseed/deck_checks.py \
        --msdb ~/Documents/ModelSEEDDatabase-dgbyg
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "deck_checks"
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def table_s4_rerun():
    t = pd.read_csv(RES / "cv" / "training_heldout.tsv", sep="\t")
    say("A. Grouped 10-fold CV of dGbyG (20 heads per fold), kJ/mol; paper Table S4 dGbyG: "
        "Pearson 0.997, MAE 9.18, RMSE 25.88")
    for lab, s in [("all rows", t), ("reaction rows", t[t.type != "formation dg"]),
                   ("formation rows", t[t.type == "formation dg"])]:
        e = s.pred_kJ - s.measured_kJ
        say(f"  {lab:15s} n={len(s):3d}  Pearson {np.corrcoef(s.pred_kJ, s.measured_kJ)[0, 1]:.4f}  "
            f"MAE {e.abs().mean():5.2f}  RMSE {np.sqrt((e ** 2).mean()):5.2f}  median|e| {e.abs().median():.2f}")


def anchors_348():
    a = pd.read_csv(RES / "comparison" / "anchors.tsv", sep="\t").set_index("rxn")
    nv = pd.read_csv(RES / "cv" / "anchor_novelty.tsv", sep="\t").set_index("rxn").reindex(a.index)
    nd = a.dgbyg_sd.round(2) != 0           # PR #323 withholds sigma = 0 as 'degenerate'
    tgt = nv.target_kcal
    say()
    say(f"B. Anchors: {len(a)} live, {int(nd.sum())} non-degenerate (dGbyG sigma > 0 at 2 dp); kcal/mol")
    say(f"  {'source':28s} {'n':>4s} {'MAE':>5s} {'within 2':>8s} | {'MAE vs transformed':>18s} {'within 2':>8s}")
    for lab, c in [("eQuilibrator", "eq_dg"), ("dGPredictor", "dgp_dg"), ("Group contribution", "gc_dg"),
                   ("dGbyG", "dgbyg_dg"), ("dGbyG KEGG structures", "dgbyg_kegg_dg"),
                   ("dGbyG (held-out)", "dgbyg_heldout_dg"),
                   ("dGbyG KEGG (held-out)", "dgbyg_heldout_kegg_dg"),
                   ("dGbyG repaired (held-out)", "dgbyg_heldout_fixed_dg")]:
        m = nd & a[c].notna()
        e = (a.loc[m, c] - a.loc[m, "dG_exp"]).abs()
        mt = m & tgt.notna()
        et = (a.loc[mt, c] - tgt[mt]).abs()
        say(f"  {lab:28s} {int(m.sum()):4d} {e.mean():5.2f} {100 * (e <= 2).mean():7.1f}% | "
            f"{et.mean():18.2f} {100 * (et <= 2).mean():7.1f}%")
    d = (tgt - a.dG_exp).abs()[nd]
    say(f"  raw anchors vs transformed targets: MAE {d.mean():.2f}, {100 * (d > 1).mean():.1f}% differ by > 1; "
        f"measured entirely within pH 6.5-7.5: {int(nv.ph_all_near_7[nd].sum())} of {int(nd.sum())}")
    cls = np.where(nv.wl2_zero == 1, "predicted exactly 0",
                   np.where(nv.wl2_twin_in_train > 0, "indistinguishable twin in training",
                            np.where(nv.all_form_in_train == 1, "all formation energies in training",
                                     "novel to the fold model")))
    cols = [("KEGG in-sample", "dgbyg_kegg_dg"), ("KEGG held-out", "dgbyg_heldout_kegg_dg"),
            ("ModelSEED held-out", "dgbyg_heldout_dg"), ("repaired held-out", "dgbyg_heldout_fixed_dg"),
            ("eQuilibrator", "eq_dg"), ("dGPredictor", "dgp_dg")]
    say("  novelty class (non-degenerate)        n  " + "  ".join(f"{lab:>18s}" for lab, _ in cols))
    for c in ("predicted exactly 0", "indistinguishable twin in training", "all formation energies in training",
              "novel to the fold model"):
        idx = a.index[nd.values & (cls == c)]
        say(f"  {c:36s} {len(idx):4d}  " + "  ".join(
            f"{(a.loc[idx, col] - a.loc[idx, 'dG_exp']).abs().mean():18.2f}" for _, col in cols))


def eq_sigma_quartiles():
    a = pd.read_csv(RES / "direction" / "anchor_ops.tsv", sep="\t", keep_default_na=False, na_values=[""])
    a = a[(a.feas_EQ == 1) & a.sig_EQ.notna()].copy()
    a["q"] = pd.qcut(a.sig_EQ.rank(method="first"), 4, labels=["Q1 (smallest)", "Q2", "Q3", "Q4"])
    say()
    say(f"C. Direction accuracy by quartile of eQuilibrator's own sigma ({len(a)} anchor records, full set)")
    for proto, ope, opd, truth in [("README protocol (GC cascade)", "gc_op_EQ", "gc_op_DGP", "truth_gc"),
                                   ("shipped rule sets vs RI-rule truth", "own_op_EQ", "own_op_DGP", "truth_ri")]:
        say(f"  {proto}")
        for q, s in a.groupby("q", observed=True):
            say(f"    {q:14s} n={len(s):3d}  median sigma {s.sig_EQ.median():.2f}  "
                f"eQuilibrator {100 * (s[ope] == s[truth]).mean():5.1f}%  "
                f"dGPredictor {100 * (s[opd] == s[truth]).mean():5.1f}%")


def magnitude_multisplit(msdb, n_splits=20):
    tmp = tempfile.mkdtemp(prefix="deck_checks_")
    os.environ.update(MSDB_ROOT=str(msdb), MSDB_CODE=str(msdb), THERMO_GRADING_OUT=tmp,
                      EQDGP_OUT=str(Path(tmp) / "eq_vs_dgp"))
    sys.path.insert(0, str(Path(msdb) / "Scripts" / "Thermodynamics" / "SourceGrading"))
    import optimize_thermo_source_assignment as o
    db = o.load_db()
    truth = o.load_truth(db)
    rows = []
    for seed in range(n_splits):
        rng = np.random.default_rng(1000 + seed)
        idx = rng.permutation(len(truth))
        cut = int(len(truth) * 0.7)
        tr, te = truth.iloc[idx[:cut]].copy(), truth.iloc[idx[cut:]].copy()
        models = o.fit_error_models(tr, db[~db.rxn.isin(te.rxn)])
        asg = o.assign(te, o.predict_error(te, models), tol=np.inf)
        strategies = {
            "argmin expected error": asg["chosen_source"].to_numpy(),
            "always eQuilibrator": np.where(te["dg_EQ"].notna() & (te["sig_EQ"] <= o.EQ_SENTINEL), "EQ", None),
            "always dGPredictor": np.where(te["dg_DGP"].notna(), "DGP", None),
            "always group contribution": np.where(te["dg_GC"].notna(), "GC", None),
            "dev priority (EQ>GC, then ML)": o.baseline_priority(te),
        }
        for lab, ch in strategies.items():
            err = o.realized_error(te, ch)
            ok = np.isfinite(err)
            rows.append(dict(seed=seed, strategy=lab, n=int(ok.sum()), mean=float(np.mean(err[ok])),
                             median=float(np.median(err[ok]))))
    r = pd.DataFrame(rows)
    g = r.groupby("strategy").agg(n=("n", "mean"), mean_abs=("mean", "mean"), sd=("mean", "std"),
                                  median_abs=("median", "mean")).sort_values("mean_abs")
    say()
    say(f"D. Magnitude source selection, {n_splits} x 70/30 splits of {len(truth)} anchors "
        f"(optimize_thermo_source_assignment.py functions, {msdb}); |chosen - measured|, kcal/mol. "
        f"Docstring claim: argmin 1.03 vs always-eQuilibrator 1.75")
    for lab, x in g.iterrows():
        say(f"  {lab:30s} n={x.n:5.1f}  mean {x.mean_abs:.2f} +/- {x.sd:.2f}  median {x.median_abs:.2f}")
    w = r.pivot(index="seed", columns="strategy", values="mean")
    for other in ("always eQuilibrator", "always dGPredictor"):
        say(f"  argmin expected error beats {other} in "
            f"{int((w['argmin expected error'] < w[other]).sum())} of {n_splits} splits")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msdb", default=str(Path.home() / "Documents" / "ModelSEEDDatabase-dgbyg"))
    ap.add_argument("--skip-msdb", action="store_true", help="skip D (needs the database checkout)")
    args = ap.parse_args()
    table_s4_rerun()
    anchors_348()
    eq_sigma_quartiles()
    if not args.skip_msdb:
        magnitude_multisplit(args.msdb)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "deck_checks.txt").write_text("\n".join(LINES) + "\n")
    print(f"\nwrote {OUT / 'deck_checks.txt'}")


if __name__ == "__main__":
    main()
