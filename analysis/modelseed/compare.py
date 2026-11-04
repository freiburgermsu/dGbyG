#!/usr/bin/env python
"""Stage 4: compare dGbyG with eQuilibrator, dGPredictor, Group contribution,
openTECR and the LLM ensemble, using the MSDB thermo paper's own definitions.

Every definition below is the manuscript's, reproduced from
MSDB_thermo_paper/analysis/{figure_common,population_basis_table,
review_transport_and_llm}.py (identical copies live under
ModelSEEDDatabase/Papers/NAR_Update_2026/analysis):

  population    live records only (is_obsolete != 1): 48,384 reactions
  covers        energy present, dg < 1e7 and |sigma| < 2500 (the dg test is
                one-sided, as shipped; no dg <= -1e7 exists)
  resolves      operator in STATED = ('>', '<', '=') among covered
  commits       operator in COMMITTED = ('>', '<')
  undetermined  operator '?' among covered
  median sigma  sorted(sigma)[n // 2]
  agreement     share of equal operators where BOTH sources commit
  unique        set difference of STATED reaction-id sets
  union         reactions with a STATED call from >= 1 predictor (LLMs excluded)

dGbyG enters as a fourth predictor whose operator came from the paper's
eQuilibrator/dGPredictor rule set (predict_reactions.py).

Anchors: the 363 distinct live stereo-exact openTECR reactions of
opentecr_comparison.csv, opentecr_dG_kJ / 4.184, ALREADY in ModelSEED
orientation (no flip), raw measurement conditions (median over measurements).

Outputs: results/comparison/*.tsv and numbers.json.
"""
import csv
import json
import math
import os
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MSDB = Path(os.environ.get("MSDB_ROOT", REPO.parent / "ModelSEEDDatabase"))
RES = HERE / "results"
OUTD = RES / "comparison"
TAU = 2.0
STATED = (">", "<", "=")
COMMITTED = (">", "<")
SOURCES = OrderedDict([("eQuilibrator", "eq"), ("dGPredictor", "dgp"),
                       ("Group contribution", "gc"), ("dGbyG", "dgbyg")])
PAPER3 = ("eQuilibrator", "dGPredictor", "Group contribution")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def load():
    rp = pd.read_csv(RES / "reaction_predictions.tsv", sep="\t", keep_default_na=False,
                     dtype=str)
    rp["is_obsolete"] = rp["is_obsolete"].astype(int)
    live = rp[rp["is_obsolete"] != 1].copy()
    # dGbyG triple, present only where Stage 2 produced a primary estimate
    live["dgbyg_dG_kcal"] = live["dG_kcal"]
    live["dgbyg_sd_kcal"] = live["sd_kcal"]
    live["dgbyg_op"] = live["op"]
    rep = RES / "repaired" / "reaction_predictions.tsv"
    if rep.exists() and "dGbyG (repaired)" not in SOURCES:
        SOURCES["dGbyG (repaired)"] = "dgbyr"
    if "dGbyG (repaired)" in SOURCES:
        r2 = pd.read_csv(rep, sep="\t", keep_default_na=False, dtype=str).set_index("rxn")
        for col, src in (("dgbyr_dG_kcal", "dG_kcal"), ("dgbyr_sd_kcal", "sd_kcal"), ("dgbyr_op", "op"),
                         ("dgbyr_status", "status")):
            live[col] = r2[src].reindex(live["rxn"]).fillna("").to_numpy()
    for lab, k in SOURCES.items():
        dg = live[f"{k}_dG_kcal"].map(fnum)
        sd = live[f"{k}_sd_kcal"].map(fnum).abs()
        live[f"{k}_dg"], live[f"{k}_sd"] = dg, sd
        # the shipped energy-valid test; a NaN sigma fails it (float('') raises there)
        live[f"{k}_cov"] = dg.notna() & sd.notna() & (dg < 1e7) & (sd < 2500)
        live[f"{k}_opv"] = live[f"{k}_op"].where(live[f"{k}_op"] != "", None)
    return rp, live.set_index("rxn", drop=False)


def median_paper(v):
    v = sorted(v)
    return v[len(v) // 2] if v else math.nan


def coverage_table(live):
    rows = []
    n_live = len(live)
    for lab, k in SOURCES.items():
        cov = live[live[f"{k}_cov"]]
        ops = cov[f"{k}_op"]
        sd = cov[f"{k}_sd"].tolist()
        allops = live[f"{k}_op"][live[f"{k}_op"] != ""]
        rows.append(OrderedDict(
            source=lab, covered=len(cov), covered_pct_of_live=100 * len(cov) / n_live,
            resolved=int(ops.isin(STATED).sum()), resolved_pct=100 * ops.isin(STATED).mean(),
            committed=int(ops.isin(COMMITTED).sum()), committed_pct=100 * ops.isin(COMMITTED).mean(),
            undetermined=int((ops == "?").sum()), undetermined_pct=100 * (ops == "?").mean(),
            n_fwd=int((allops == ">").sum()), n_rev_ok=int((allops == "=").sum()),
            n_bwd=int((allops == "<").sum()), n_unknown=int((allops == "?").sum()),
            n_entries=len(allops),
            median_sigma=median_paper(sd),
            sigma_p05=sorted(sd)[int(0.05 * len(sd))] if sd else math.nan,
            sigma_p95=sorted(sd)[int(0.95 * len(sd))] if sd else math.nan))
    return pd.DataFrame(rows)


def union_table(live):
    st = {lab: set(live.index[live[f"{k}_op"].isin(STATED)]) for lab, k in SOURCES.items()}
    cm = {lab: set(live.index[live[f"{k}_op"].isin(COMMITTED)]) for lab, k in SOURCES.items()}
    u3 = set().union(*(st[s] for s in PAPER3))
    u4 = u3 | st["dGbyG"]
    u4r = u3 | st["dGbyG (repaired)"] if "dGbyG (repaired)" in st else None
    none3 = set(live.index) - u3
    llm_stated = set(live.index[live["llm_op"].isin(STATED)])
    out = OrderedDict(
        live=len(live), union_paper3=len(u3), none_paper3=len(none3),
        union_with_dgbyg=len(u4), none_with_dgbyg=len(live) - len(u4),
        dgbyg_stated_in_none3=len(st["dGbyG"] & none3),
        dgbyg_committed_in_none3=len(cm["dGbyG"] & none3),
        dgbyg_covered_in_none3=int(live.loc[list(none3), "dgbyg_cov"].sum()),
        llm_stated_in_none3=len(llm_stated & none3),
        llm_stated_in_none4=len(llm_stated & (set(live.index) - u4)),
    )
    if u4r is not None:
        out["union_with_dgbyg_repaired"] = len(u4r)
        out["dgbyg_repaired_stated_in_none3"] = len(st["dGbyG (repaired)"] & none3)
        out["dgbyg_repaired_committed_in_none3"] = len(cm["dGbyG (repaired)"] & none3)
    # why the none3 reactions stay undirected by dGbyG
    out["none3_by_dgbyg_status"] = dict(Counter(live.loc[list(none3), "status"]))
    out["none3_dgbyg_ops"] = dict(Counter(live.loc[list(none3 & set(live.index[live['dgbyg_cov']])), "dgbyg_op"]))
    return out, st, cm


def pairwise_direction(live, st, cm):
    rows = []
    for a in SOURCES:
        for b in SOURCES:
            if a >= b:
                continue
            ka, kb = SOURCES[a], SOURCES[b]
            both = sorted(cm[a] & cm[b])
            agree = int((live.loc[both, f"{ka}_op"] == live.loc[both, f"{kb}_op"]).sum()) if both else 0
            both_st = sorted(st[a] & st[b])
            agree_st = int((live.loc[both_st, f"{ka}_op"] == live.loc[both_st, f"{kb}_op"]).sum()) if both_st else 0
            rows.append(OrderedDict(
                a=a, b=b, both_commit=len(both), agree=agree,
                agree_pct=100 * agree / len(both) if both else math.nan,
                opposed=len(both) - agree,
                both_stated=len(both_st), agree_stated=agree_st,
                agree_stated_pct=100 * agree_st / len(both_st) if both_st else math.nan,
                unique_a=len(st[a] - st[b]), unique_b=len(st[b] - st[a])))
    return pd.DataFrame(rows)


def energy_stats(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y
    if len(d) < 3:
        return OrderedDict(n=len(d))
    return OrderedDict(
        n=len(d), pearson=stats.pearsonr(x, y)[0], spearman=stats.spearmanr(x, y)[0],
        mae=np.mean(np.abs(d)), median_ae=np.median(np.abs(d)), rmse=np.sqrt(np.mean(d ** 2)),
        # rounded before the tau test: both inputs carry 2 dp, and a float
        # difference of exactly 2.00 can land a hair above TAU
        bias=np.mean(d), within_tau_pct=100 * np.mean(np.round(np.abs(d), 6) <= TAU),
        sign_agree_pct=100 * np.mean(np.sign(x) == np.sign(y)))


def pairwise_energy(live):
    rows = []
    variants = [("dGbyG", "dgbyg")] + ([("dGbyG (repaired)", "dgbyr")] if "dGbyG (repaired)" in SOURCES else [])
    for (alab, ka), b in [(v, b) for v in variants for b in PAPER3]:
        kb = SOURCES[b]
        m = live[f"{ka}_cov"] & live[f"{kb}_cov"]
        sub = live[m]
        r = energy_stats(sub[f"{ka}_dg"], sub[f"{kb}_dg"])
        rows.append(OrderedDict(a=alab, b=b, subset="all", **r))
        # transport reactions are compartment-collapsed by every source; report separately
        # GC's shipped reaction energies drift ~5 kcal/mol per net H+ against both
        # dGbyG and eQuilibrator (Convention A bookkeeping); calibrate_sigma.py
        # restricts GC to no net bound-H change for the same reason
        nh0 = sub["n_proton"].map(fnum) == 0
        for lab, mm in (("non-transport", sub["is_transport"] != "1"), ("transport", sub["is_transport"] == "1"),
                        ("net H+ = 0", nh0), ("net H+ != 0", ~nh0)):
            r = energy_stats(sub.loc[mm, f"{ka}_dg"], sub.loc[mm, f"{kb}_dg"])
            rows.append(OrderedDict(a=alab, b=b, subset=lab, **r))
    # the paper's own pair for reference
    m = live["eq_cov"] & live["dgp_cov"]
    rows.append(OrderedDict(a="dGPredictor", b="eQuilibrator", subset="all",
                            **energy_stats(live.loc[m, "dgp_dg"], live.loc[m, "eq_dg"])))
    m = live["eq_cov"] & live["gc_cov"]
    rows.append(OrderedDict(a="Group contribution", b="eQuilibrator", subset="all",
                            **energy_stats(live.loc[m, "gc_dg"], live.loc[m, "eq_dg"])))
    m0 = m & (live["n_proton"].map(fnum) == 0)
    rows.append(OrderedDict(a="Group contribution", b="eQuilibrator", subset="net H+ = 0",
                            **energy_stats(live.loc[m0, "gc_dg"], live.loc[m0, "eq_dg"])))
    m = live["eq_cov"] & live["dgp_cov"] & (live["n_proton"].map(fnum) == 0)
    rows.append(OrderedDict(a="dGPredictor", b="eQuilibrator", subset="net H+ = 0",
                            **energy_stats(live.loc[m, "dgp_dg"], live.loc[m, "eq_dg"])))
    return pd.DataFrame(rows)


def anchors(live):
    t = pd.read_csv(MSDB / "Biochemistry/Thermodynamics/SourceGrading/opentecr_comparison.csv")
    a = t[(t.match_tier == "stereo_exact") & t.modelseed_rxn.isin(live.index)]
    an = pd.DataFrame({"rxn": a.modelseed_rxn.values,
                       "dG_exp": (a.opentecr_dG_kJ / 4.184).values,
                       "sd_exp": (a.opentecr_dG_sd_kJ / 4.184).values,
                       "n_measurements": a.n_measurements.values,
                       "opentecr_reaction": a.opentecr_reaction.values}).set_index("rxn")
    assert len(an) == 363 and an.index.is_unique, len(an)
    for lab, k in SOURCES.items():
        an[f"{k}_dg"] = live.loc[an.index, f"{k}_dg"].where(live.loc[an.index, f"{k}_cov"])
        an[f"{k}_sd"] = live.loc[an.index, f"{k}_sd"].where(live.loc[an.index, f"{k}_cov"])
        an[f"{k}_op"] = live.loc[an.index, f"{k}_op"]
    an["is_transport"] = live.loc[an.index, "is_transport"]
    an["dgbyg_status"] = live.loc[an.index, "status"]
    an["llm_op"] = live.loc[an.index, "llm_op"]
    return an


def anchor_accuracy(an, extra=None):
    """Accuracy of each source's shipped value against the anchors: per-source
    over the anchors it covers, and on the common set every source covers."""
    cols = dict((lab, (f"{k}_dg", f"{k}_sd")) for lab, k in SOURCES.items())
    if extra:
        cols.update(extra)
    cols = OrderedDict((lab, c) for lab, c in cols.items() if c[0] in an)
    rows = []
    common = an.index[np.all([an[c[0]].notna() for c in cols.values()], axis=0)]
    for lab, (dgc, sdc) in cols.items():
        for subset, idx in (("own coverage", an.index[an[dgc].notna()]), ("common", common)):
            r = energy_stats(an.loc[idx, dgc], an.loc[idx, "dG_exp"])
            rows.append(OrderedDict(source=lab, subset=subset, **r))
    return pd.DataFrame(rows), common


def calibration(an, extra=None):
    """M11's statistic: z = (prediction - measured) / reported sigma, per source,
    one prediction per anchored reaction. Reported: RMS of z, median of |z|,
    share with |z| < 1 and < 2 (strict, as calibrate_sigma.report()). Reactions with sigma == 0 are excluded (z
    undefined) and counted."""
    cols = dict((lab, (f"{k}_dg", f"{k}_sd")) for lab, k in SOURCES.items())
    if extra:
        cols.update(extra)
    cols = OrderedDict((lab, c) for lab, c in cols.items() if c[0] in an)
    rows = []
    for lab, (dgc, sdc) in cols.items():
        s = an[an[dgc].notna() & an[sdc].notna()]
        zero = int((s[sdc] <= 0).sum())
        s = s[s[sdc] > 0]
        z = (s[dgc] - s["dG_exp"]) / s[sdc]
        rows.append(OrderedDict(source=lab, n=len(z), n_sigma_zero=zero,
                                rms_z=float(np.sqrt(np.mean(z ** 2))), median_abs_z=float(np.median(np.abs(z))),
                                within_1sd_pct=100 * float(np.mean(np.abs(z) < 1)),
                                within_2sd_pct=100 * float(np.mean(np.abs(z) < 2)),
                                median_sigma=float(np.median(s[sdc]))))
    return pd.DataFrame(rows)


def kappa(a, b, labels=COMMITTED):
    """Agreement, chance agreement sum_d p_a(d) p_b(d), and Cohen's kappa."""
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    if not n:
        return 0, math.nan, math.nan, math.nan
    agree = float(np.mean(a == b))
    chance = sum(float(np.mean(a == d)) * float(np.mean(b == d)) for d in labels)
    return n, agree, chance, (agree - chance) / (1 - chance) if chance < 1 else math.nan


def llm_layer(live, st):
    """review_transport_and_llm.py's LLM analysis (live basis), with dGbyG added."""
    rows = []
    llm = live["llm_op"]
    for lab, k in SOURCES.items():
        ops = live[f"{k}_op"]
        both = live.index[ops.isin(COMMITTED) & llm.isin(COMMITTED)]
        n, agree, chance, kap = kappa(ops[both], llm[both])
        allc = ops[ops != ""]
        com = ops[ops.isin(COMMITTED)]
        rows.append(OrderedDict(source=lab, both_commit=n, agree_pct=100 * agree, chance_pct=100 * chance,
                                kappa=kap, fwd_share_all_calls_pct=100 * float((allc == ">").mean()),
                                fwd_share_committed_pct=100 * float((com == ">").mean())))
    allc = llm[llm != ""]
    com = llm[llm.isin(COMMITTED)]
    rows.append(OrderedDict(source="LLMs", both_commit=math.nan, agree_pct=math.nan, chance_pct=math.nan,
                            kappa=math.nan, fwd_share_all_calls_pct=100 * float((allc == ">").mean()),
                            fwd_share_committed_pct=100 * float((com == ">").mean())))
    tab = pd.DataFrame(rows)

    none3 = set(live.index) - set().union(*(st[s] for s in PAPER3))
    none4 = none3 - st["dGbyG"]
    tiers = OrderedDict()
    for name, pop in (("none_paper3", none3), ("none_with_dgbyg", none4)):
        sub = llm.loc[list(pop)]
        tiers[name] = OrderedDict(n=len(pop), llm_any=int((sub != "").sum()),
                                  llm_stated=int(sub.isin(STATED).sum()),
                                  llm_committed=int(sub.isin(COMMITTED).sum()))
    # where dGbyG is the ONLY thermodynamic voice: its agreement with the LLMs
    newly = sorted(none3 & st["dGbyG"])
    both = [r for r in newly if live.at[r, "dgbyg_op"] in COMMITTED and llm[r] in COMMITTED]
    n, agree, chance, kap = kappa(live.loc[both, "dgbyg_op"], llm[both])
    tiers["dgbyg_new_vs_llm"] = OrderedDict(dgbyg_newly_stated=len(newly), both_commit=n,
                                            agree_pct=100 * agree, chance_pct=100 * chance, kappa=kap)
    return tab, tiers


def proxy_rms(live):
    """M11's dGPredictor 'RMS 1.15' sentence: z = (dg_src - dg_EQ) / sigma_src over
    reactions with a tight eQuilibrator reference (sigma_EQ <= 2.0), live basis."""
    rows = []
    for lab, k in SOURCES.items():
        if k == "eq":
            continue
        m = live[f"{k}_cov"] & live["eq_cov"] & (live["eq_sd"] <= 2.0) & (live[f"{k}_sd"] > 0)
        z = (live.loc[m, f"{k}_dg"] - live.loc[m, "eq_dg"]) / live.loc[m, f"{k}_sd"]
        # the paper's convention puts only the source's sigma in the denominator;
        # eQuilibrator's own sigma is comparable to dGbyG's on this subset, so the
        # combined-sigma z is reported alongside
        zc = (live.loc[m, f"{k}_dg"] - live.loc[m, "eq_dg"]) / np.sqrt(live.loc[m, f"{k}_sd"] ** 2
                                                                        + live.loc[m, "eq_sd"] ** 2)
        rows.append(OrderedDict(source=lab, n=int(m.sum()), rms_z=float(np.sqrt(np.mean(z ** 2))),
                                p90_abs_z=float(np.quantile(np.abs(z), 0.9)),
                                within_1sd_pct=100 * float(np.mean(np.abs(z) < 1)),
                                rms_z_combined=float(np.sqrt(np.mean(zc ** 2))),
                                within_1sd_combined_pct=100 * float(np.mean(np.abs(zc) < 1)),
                                n_sigma_zero=int((live[f"{k}_cov"] & live["eq_cov"] & (live["eq_sd"] <= 2.0)
                                                  & (live[f"{k}_sd"] <= 0)).sum())))
    return pd.DataFrame(rows)


def silver_band(live):
    """figure_common._silver_sigma_band, plus dGbyG's sigma on the reactions the
    EXISTING (3-source) scheme grades silver -- dGbyG took no part in that grading."""
    g = pd.read_csv(MSDB / "Biochemistry/Thermodynamics/SourceGrading/results/thermo_grades/reaction_grades.tsv",
                    sep="\t", keep_default_na=False, dtype=str)
    silver = set(g.loc[g["best_grade"] == "SILVER", "rxn"]) & set(live.index)
    out = OrderedDict()
    for lab, k in SOURCES.items():
        v = sorted(live.loc[[r for r in silver if live.at[r, f"{k}_cov"]], f"{k}_sd"].tolist())
        out[lab] = (v[int(0.05 * len(v))], v[int(0.95 * len(v))], len(v)) if len(v) >= 20 else None
    return out


def own_training_cv():
    """calibrate_sigma.report(): z = |mu - y| / sigma over held-out predictions of a
    model's OWN training rows (here dGbyG's grouped 10-fold CV), median |z| against
    the calibrated 0.674, share |z| < 1 against 68%, plus RMS and |z| < 2, and the
    sigma band table with edges [0,0.5,1,2,5,10,25,inf]."""
    p = RES / "cv" / "training_heldout.tsv"
    if not p.exists():
        return None, None
    t = pd.read_csv(p, sep="\t")
    out, bands = [], []
    for subset, sub in (("all rows", t), ("reaction rows", t[t["type"] == "reaction dg"]),
                        ("formation rows", t[t["type"] == "formation dg"])):
        s, e = sub["pred_sd_kcal"].to_numpy(), np.abs(sub["pred_kcal"] - sub["measured_kcal"]).to_numpy()
        fin = np.isfinite(s) & np.isfinite(e) & (s > 0)
        z = e[fin] / s[fin]
        # the target's own measurement scale (the notebook's weight scale), added in quadrature
        meas = sub["scale_kJ"].to_numpy() / 4.184
        zc = e[fin] / np.sqrt(s[fin] ** 2 + meas[fin] ** 2)
        out.append(OrderedDict(subset=subset, n=int(fin.sum()), n_sigma_zero=int((s <= 0).sum()),
                               median_sigma=float(np.median(s[fin])), median_abs_err=float(np.median(e[fin])),
                               mae=float(np.mean(e[fin])), rmse=float(np.sqrt(np.mean(e[fin] ** 2))),
                               within_tau_pct=100 * float(np.mean(e[fin] <= TAU)),
                               median_abs_z=float(np.median(z)), rms_z=float(np.sqrt(np.mean(z ** 2))),
                               within_1sd_pct=100 * float(np.mean(z < 1)), within_2sd_pct=100 * float(np.mean(z < 2)),
                               median_abs_z_with_meas=float(np.median(zc)),
                               within_1sd_with_meas_pct=100 * float(np.mean(zc < 1))))
        if subset == "reaction rows":
            edges = [0, 0.5, 1, 2, 5, 10, 25, np.inf]
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = fin & (s >= lo) & (s < hi)
                if m.sum() >= 5:
                    bands.append(OrderedDict(sigma_lo=lo, sigma_hi=hi, n=int(m.sum()),
                                             median_abs_err=float(np.median(e[m])),
                                             p90_abs_err=float(np.quantile(e[m], 0.9))))
                elif m.sum():
                    bands.append(OrderedDict(sigma_lo=lo, sigma_hi=hi, n=int(m.sum()),
                                             median_abs_err=math.nan, p90_abs_err=math.nan))
    return pd.DataFrame(out), pd.DataFrame(bands)


def coverage_overlap(live):
    """Energy coverage of dGbyG against the three shipped predictors (live)."""
    d, e, p, g = (live[f"{k}_cov"] for k in ("dgbyg", "eq", "dgp", "gc"))
    any3 = e | p | g
    return OrderedDict(
        dgbyg=int(d.sum()), any_paper3=int(any3.sum()), union_with_dgbyg=int((any3 | d).sum()),
        dgbyg_only=int((d & ~any3).sum()), dgbyg_and_eq=int((d & e).sum()), dgbyg_not_eq=int((d & ~e).sum()),
        eq_not_dgbyg=int((e & ~d).sum()), dgbyg_and_dgp=int((d & p).sum()), dgp_not_dgbyg=int((p & ~d).sum()),
        dgbyg_not_eq_not_dgp=int((d & ~e & ~p).sum()))


def decline_reasons(live):
    """Why dGbyG has no primary estimate, against whether eQ / dGP cover it."""
    rows = []
    for st, sub in live.groupby("status"):
        rows.append(OrderedDict(status=st, n=len(sub), eq_covers=int(sub["eq_cov"].sum()),
                                dgp_covers=int(sub["dgp_cov"].sum()), gc_covers=int(sub["gc_cov"].sum()),
                                transport=int((sub["is_transport"] == "1").sum())))
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def zero_sigma(live):
    """dGbyG's exactly-zero sigma: every one of its 100 heads is a 2-layer MPNN
    with no chirality feature, so a reaction whose sides are the same graph up
    to stereo, or that leaves every atom's 2-bond neighbourhood unchanged,
    sums to exactly 0 on every head. The rule set reads sigma 0 as certainty
    ('=' via the tight-sigma branch), so report what excluding them changes."""
    out = OrderedDict()
    for lab, k in (("dGbyG", "dgbyg"), ("dGbyG (repaired)", "dgbyr")):
        if f"{k}_cov" not in live:
            continue
        ok = live[live[f"{k}_cov"]]
        z = ok[ok[f"{k}_sd"] == 0]
        rest = ok[ok[f"{k}_sd"] > 0]
        out[lab] = OrderedDict(
            n_sigma_zero=len(z), n_dg_zero_too=int((z[f"{k}_dg"] == 0).sum()),
            ops=dict(Counter(z[f"{k}_op"])),
            eq_covers=int(z["eq_cov"].sum()), eq_abs_gt1=int((z["eq_cov"] & (z["eq_dg"].abs() > 1)).sum()),
            resolved_pct_all=100 * float(ok[f"{k}_op"].isin(STATED).mean()),
            resolved_pct_excl_sigma0=100 * float(rest[f"{k}_op"].isin(STATED).mean()),
            median_sigma_excl_sigma0=median_paper(rest[f"{k}_sd"].tolist()),
            examples=z.loc[z.index[:8], "name"].tolist())
    return out


def charge_imbalanced(live):
    """dGbyG-scored reactions that ModelSEED itself marks charge-imbalanced (CI);
    dropping H+ hides that from dGbyG's balance check (predict_reactions.py)."""
    ci = live["rxn_status"].str.contains("CI")
    out = OrderedDict()
    for lab, k in SOURCES.items():
        out[lab] = int((live[f"{k}_cov"] & ci).sum())
    out["dGbyG_ok_and_CI"] = int((live["dgbyg_cov"] & ci).sum())
    return out


def anchor_diagnostics(an):
    """Two checks on what the anchor tables can and cannot say.

    1. Novelty (anchor_novelty.py): held-out means the anchor's own rows are
       withheld, not its reaction class. Accuracy by whether the fold model
       has seen an indistinguishable 2-bond change, or every participant's
       formation energy.
    2. Truth: the anchors are medians of raw measurements at their own pH and
       T. dGbyG's training targets for the same reactions are those
       measurements transformed to pH 7 / I 0.25 / pMg 14. Scoring every source
       against both shows how much the ranking depends on that choice."""
    p = RES / "cv" / "anchor_novelty.tsv"
    if not p.exists():
        return None, None, None
    nv = pd.read_csv(p, sep="\t").set_index("rxn").reindex(an.index)
    cls = np.where(nv.wl2_zero == 1, "predicted exactly 0",
                   np.where(nv.wl2_twin_in_train > 0, "indistinguishable twin in training",
                            np.where(nv.all_form_in_train == 1, "all formation energies in training",
                                     "neither (novel to the fold model)")))
    rows = []
    cols = [("dGbyG KEGG structures", "dgbyg_kegg_dg"), ("dGbyG KEGG structures (held-out)", "dgbyg_heldout_kegg_dg"),
            ("dGbyG (held-out)", "dgbyg_heldout_dg"), ("dGbyG (repaired, held-out)", "dgbyg_heldout_fixed_dg"),
            ("eQuilibrator", "eq_dg"), ("dGPredictor", "dgp_dg")]
    for c in ("predicted exactly 0", "indistinguishable twin in training", "all formation energies in training",
              "neither (novel to the fold model)"):
        idx = an.index[cls == c]
        r = OrderedDict(anchor_class=c, n=len(idx))
        for lab, col in cols:
            if col in an:
                r[f"MAE {lab}"] = float((an.loc[idx, col] - an.loc[idx, "dG_exp"]).abs().mean()) if len(idx) else math.nan
        rows.append(r)
    novelty = pd.DataFrame(rows)

    rows = []
    srcs = [("eQuilibrator", "eq_dg"), ("dGPredictor", "dgp_dg"), ("Group contribution", "gc_dg"),
            ("dGbyG", "dgbyg_dg"), ("dGbyG (repaired)", "dgbyr_dg"), ("dGbyG KEGG structures", "dgbyg_kegg_dg"),
            ("dGbyG (held-out)", "dgbyg_heldout_dg"), ("dGbyG KEGG structures (held-out)", "dgbyg_heldout_kegg_dg")]
    tgt = nv["target_kcal"]
    for lab, col in srcs:
        if col not in an:
            continue
        m = an[col].notna() & tgt.notna()
        rows.append(OrderedDict(source=lab, n=int(m.sum()),
                                mae_vs_anchor=float((an.loc[m, col] - an.loc[m, "dG_exp"]).abs().mean()),
                                mae_vs_transformed=float((an.loc[m, col] - tgt[m]).abs().mean()),
                                within_tau_vs_anchor_pct=100 * float(((an.loc[m, col] - an.loc[m, "dG_exp"]).abs() <= TAU).mean()),
                                within_tau_vs_transformed_pct=100 * float(((an.loc[m, col] - tgt[m]).abs() <= TAU).mean())))
    truth = pd.DataFrame(rows)
    d = (tgt - an["dG_exp"]).abs()
    info = OrderedDict(n=int(d.notna().sum()), anchor_vs_transformed_mae=float(d.mean()),
                       anchor_vs_transformed_median=float(d.median()), share_gt1_pct=100 * float((d > 1).mean()),
                       share_gt2_pct=100 * float((d > 2).mean()), ph_all_near_7=int(nv.ph_all_near_7.sum()),
                       t_all_near_298=int(nv.t_all_near_298.sum()),
                       classes=dict(Counter(cls)))
    return novelty, truth, info


def extra_diagnostics(live, st, cm):
    """Numbers the report's prose leans on, computed here so none is typed in."""
    out = OrderedDict()
    # GC against dGbyG and eQuilibrator by the reaction's net H+
    nh = live["n_proton"].map(fnum).round()
    rows = []
    for other, k in (("dGbyG", "dgbyg"), ("eQuilibrator", "eq")):
        m = live["gc_cov"] & live[f"{k}_cov"]
        d = (live.loc[m, "gc_dg"] - live.loc[m, f"{k}_dg"])
        for h, g in d.groupby(nh[m].clip(-3, 3)):
            rows.append(OrderedDict(other=other, net_H=int(h), n=len(g), median_gc_minus_other=float(g.median()),
                                    median_abs=float(g.abs().median())))
    out["gc_by_net_H"] = rows
    # how large the energies are where dGbyG and eQuilibrator both commit
    both = sorted(cm["dGbyG"] & cm["eQuilibrator"])
    out["dgbyg_eq_both_commit_median_abs_eq_dg"] = float(live.loc[both, "eq_dg"].abs().median())
    m = live["dgbyg_cov"] & live["eq_cov"]
    out["dgbyg_eq_share_absdiff_gt5_pct"] = 100 * float(((live.loc[m, "dgbyg_dg"] - live.loc[m, "eq_dg"]).abs() > 5).mean())
    # ModelSEED's own balance flags on the reactions dGbyG's check rejects
    rej = live["status"].isin(["unbalanced", "unbalanced_water"])
    out["balance_rejected"] = int(rej.sum())
    out["balance_rejected_modelseed_MI"] = int((rej & live["rxn_status"].str.contains("MI")).sum())
    out["balance_rejected_dgp_covers"] = int((rej & live["dgp_cov"]).sum())
    # newly directed reactions that ModelSEED marks charge-imbalanced
    none3 = set(live.index) - set().union(*(st[s] for s in PAPER3))
    new = sorted(none3 & st["dGbyG"])
    newc = sorted(none3 & cm["dGbyG"])
    ci = live["rxn_status"].str.contains("CI")
    out["newly_stated_CI"] = int(ci[new].sum())
    out["newly_committed_CI"] = int(ci[newc].sum())
    out["newly_stated_sigma0"] = int((live.loc[new, "dgbyg_sd"] == 0).sum())
    # what the structure repair newly commits, against eQuilibrator
    if "dgbyr_op" in live:
        nc = live["dgbyr_op"].isin(COMMITTED) & ~live["dgbyg_op"].isin(COMMITTED)
        e = nc & live["eq_op"].isin(COMMITTED)
        out["repair_newly_committed"] = int(nc.sum())
        out["repair_newly_committed_eq_commits"] = int(e.sum())
        out["repair_newly_committed_eq_agree_pct"] = 100 * float((live.loc[e, "dgbyr_op"] == live.loc[e, "eq_op"]).mean())
    # transport: compartment-collapsed values
    tr = live["dgbyg_cov"] & (live["is_transport"] == "1")
    out["transport_scored"] = int(tr.sum())
    out["transport_atp_hydrolysis_value"] = int((tr & (live["dgbyg_dg"] == live.loc[tr, "dgbyg_dg"].mode().iloc[0])).sum())
    out["transport_modal_value"] = float(live.loc[tr, "dgbyg_dg"].mode().iloc[0])
    out["transport_reversible"] = int((tr & (live["dgbyg_op"] == "=")).sum())
    # null-net transport that group contribution nevertheless scores
    nn = live["status"] == "null_net"
    out["null_net_gc_scores"] = int((nn & live["gc_cov"]).sum())
    out["null_net_gc_zero_eq"] = int((nn & live["gc_cov"] & (live["gc_dg"] == 0) & (live["gc_op"] == "=")).sum())
    # nucleotide-phosphate reactions, where the pMg shift concentrates (S2)
    npp = live["equation"].str.contains(r"cpd00002\[|cpd00008\[|cpd00018\[|cpd00012\[", regex=True)
    out["dgbyg_scored_with_atp_adp_amp_ppi"] = int((live["dgbyg_cov"] & npp).sum())
    return out


def training_sigma0_kinds():
    p, f = RES / "cv" / "training_heldout.tsv", RES / "cv" / "folds.tsv"
    if not p.exists():
        return None
    t = pd.read_csv(p, sep="\t")
    k = pd.read_csv(f, sep="\t", keep_default_na=False)["key"]
    z = t["pred_sd_kcal"] == 0
    stereo = k.str.contains("empty")
    return OrderedDict(n=int(z.sum()), pure_stereo=int((z & stereo).sum()), other=int((z & ~stereo).sum()),
                       other_examples=t.loc[z & ~stereo, "description"].tolist())


def compound_agreement():
    """Compound-level dGf'° of dGbyG against the eQuilibrator and dGPredictor
    compound entries (live compounds, no R groups). The paper compares
    reactions, not compounds; absolute formation energies carry each method's
    own reference conventions, which largely cancel in balanced reactions.
    The offset is tested against hydrogen count and charge (least squares on
    formula H, ModelSEED charge, and dGbyG-minus-ModelSEED H after dGbyG's
    normalisation); R^2 says how much of it those explain."""
    import re
    cp = pd.read_csv(RES / "compound_predictions.tsv", sep="\t", keep_default_na=False)
    cp = cp[(cp.status == "ok") & (cp.is_obsolete == 0) & ~cp.smiles_modelseed.str.contains("*", regex=False)]

    def n_h(fm):
        m = re.search(r"H(\d*)(?![a-z])", fm or "")
        return math.nan if not fm else (0 if not m else int(m.group(1) or 1))

    cp = cp.assign(nH=cp.formula.map(n_h), chg=pd.to_numeric(cp.charge, errors="coerce"),
                   bagH=cp.atom_bag_dgbyg.map(lambda b: json.loads(b).get("H", 0) if b else math.nan))
    rows = []
    for lab, k in (("eQuilibrator", "eq"), ("dGPredictor", "dgp")):
        # the paper's energy-valid test: eQuilibrator's "could not estimate"
        # sentinel sigma (>= 2500) carries a number that is not an estimate
        sd = pd.to_numeric(cp[f"{k}_sd_kcal"], errors="coerce").abs()
        valid = (cp[f"{k}_dGf_kcal"] != "") & (sd < 2500)
        sub = cp[valid].dropna(subset=["nH", "chg", "bagH"])
        x, y = sub.dgbyg_dGf_kcal.astype(float), sub[f"{k}_dGf_kcal"].astype(float)
        d = x - y
        X = np.column_stack([np.ones(len(sub)), sub.nH, sub.chg, sub.bagH - sub.nH])
        coef = np.linalg.lstsq(X, d, rcond=None)[0]
        r2 = 1 - float(((d - X @ coef) ** 2).sum() / ((d - d.mean()) ** 2).sum())
        rows.append(OrderedDict(versus=lab, n=len(sub), pearson=float(np.corrcoef(x, y)[0, 1]),
                                median_diff=float(d.median()), median_abs_diff=float(d.abs().median()),
                                within_tau_pct=100 * float((np.round(d.abs(), 6) <= TAU).mean()),
                                r2_offset_vs_H_and_charge=r2))
    return pd.DataFrame(rows)


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    rp, live = load()
    numbers = OrderedDict()
    numbers["live"] = len(live)
    numbers["dgbyg_status_live"] = dict(Counter(live["status"]))

    cov = coverage_table(live)
    cov.to_csv(OUTD / "coverage.tsv", sep="\t", index=False, float_format="%.4f")
    u, st, cm = union_table(live)
    numbers["union"] = u
    pd_dir = pairwise_direction(live, st, cm)
    pd_dir.to_csv(OUTD / "direction_agreement.tsv", sep="\t", index=False, float_format="%.4f")
    pe = pairwise_energy(live)
    pe.to_csv(OUTD / "energy_agreement.tsv", sep="\t", index=False, float_format="%.4f")

    an = anchors(live)
    extra = OrderedDict()
    sk = RES / "structure_sensitivity" / "anchors_kegg.tsv"
    if sk.exists():
        k = pd.read_csv(sk, sep="\t", keep_default_na=False).set_index("rxn")
        an["dgbyg_kegg_dg"] = k["dgbyg_kegg_dg"].reindex(an.index).map(fnum)
        an["dgbyg_kegg_sd"] = k["dgbyg_kegg_sd"].reindex(an.index).map(fnum)
        extra["dGbyG KEGG structures"] = ("dgbyg_kegg_dg", "dgbyg_kegg_sd")
    ho = RES / "cv" / "anchor_heldout.tsv"
    if ho.exists():
        h = pd.read_csv(ho, sep="\t", keep_default_na=False).set_index("rxn")
        for col, src in (("dgbyg_heldout", "heldout"), ("dgbyg_heldout_fixed", "heldout_fixed"),
                         ("dgbyg_heldout_kegg", "heldout_kegg")):
            an[f"{col}_dg"] = h[f"{src}_dG_kcal"].reindex(an.index).map(fnum)
            an[f"{col}_sd"] = h[f"{src}_sd_kcal"].reindex(an.index).map(fnum)
        an["dgbyg_in_training"] = h["in_training"].reindex(an.index)
        extra["dGbyG (held-out)"] = ("dgbyg_heldout_dg", "dgbyg_heldout_sd")
        extra["dGbyG (repaired, held-out)"] = ("dgbyg_heldout_fixed_dg", "dgbyg_heldout_fixed_sd")
        extra["dGbyG KEGG structures (held-out)"] = ("dgbyg_heldout_kegg_dg", "dgbyg_heldout_kegg_sd")
        numbers["anchor_in_training"] = dict(Counter(an["dgbyg_in_training"].fillna("not scored")))
    an.to_csv(OUTD / "anchors.tsv", sep="\t", float_format="%.4f")
    acc, common = anchor_accuracy(an, extra)
    acc.to_csv(OUTD / "anchor_accuracy.tsv", sep="\t", index=False, float_format="%.4f")
    numbers["anchor_common_n"] = len(common)
    cal = calibration(an, extra)
    cal.to_csv(OUTD / "calibration.tsv", sep="\t", index=False, float_format="%.4f")

    llm_tab, tiers = llm_layer(live, st)
    llm_tab.to_csv(OUTD / "llm_agreement.tsv", sep="\t", index=False, float_format="%.4f")
    numbers["llm_tiers"] = tiers
    prx = proxy_rms(live)
    prx.to_csv(OUTD / "proxy_rms.tsv", sep="\t", index=False, float_format="%.4f")
    numbers["silver_sigma_band"] = silver_band(live)
    numbers["zero_sigma"] = zero_sigma(live)
    numbers["covered_and_modelseed_CI"] = charge_imbalanced(live)
    numbers["extra"] = extra_diagnostics(live, st, cm)
    ca = compound_agreement()
    ca.to_csv(OUTD / "compound_agreement.tsv", sep="\t", index=False, float_format="%.4f")
    print(f"\n=== compound agreement\n{ca.round(3).to_string(index=False)}")
    numbers["training_sigma0"] = training_sigma0_kinds()
    nov, truth, tinfo = anchor_diagnostics(an)
    if nov is not None:
        nov.to_csv(OUTD / "anchor_novelty_accuracy.tsv", sep="\t", index=False, float_format="%.4f")
        truth.to_csv(OUTD / "anchor_truth_choice.tsv", sep="\t", index=False, float_format="%.4f")
        numbers["anchor_truth"] = tinfo
        print(f"\n=== anchor novelty\n{nov.round(3).to_string(index=False)}\n\n=== truth choice\n{truth.round(3).to_string(index=False)}")
    numbers["coverage_overlap"] = coverage_overlap(live)
    dr = decline_reasons(live)
    dr.to_csv(OUTD / "dgbyg_status_vs_coverage.tsv", sep="\t", index=False)
    print(f"\n=== dGbyG status vs other sources' coverage\n{dr.to_string(index=False)}")
    otc, bands = own_training_cv()
    if otc is not None:
        otc.to_csv(OUTD / "own_training_cv.tsv", sep="\t", index=False, float_format="%.4f")
        bands.to_csv(OUTD / "own_training_cv_bands.tsv", sep="\t", index=False, float_format="%.4f")

    with open(OUTD / "numbers.json", "w") as fh:
        json.dump(numbers, fh, indent=1, default=str)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    for name, df in (("coverage", cov), ("direction", pd_dir), ("energy", pe), ("anchors", acc),
                     ("calibration", cal), ("llm", llm_tab), ("proxy rms", prx)) + ((("own training cv", otc),) if otc is not None else ()):
        print(f"\n=== {name}\n{df.round(3).to_string(index=False)}")
    print(json.dumps(numbers, indent=1, default=str))


if __name__ == "__main__":
    main()
