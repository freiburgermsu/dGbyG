#!/usr/bin/env python
"""Stage 5: direction accuracy against MEASUREMENT, and the README's source-
selection strategy table, with dGbyG added.

Two questions, each answered first on the paper's own protocol (reproduced
before anything is added) and then with dGbyG as a fourth predictor.

(A) Does each source's direction call agree with the MEASURED direction?
    Replicates MSDB_thermo_paper/analysis/review_transport_and_llm.py
    ``_vs_measured`` exactly:
      * margins 0, 5.0, 11.7 kJ/mol on opentecr_dG_kJ (already in ModelSEED
        orientation -- no flip);
      * measured m = '>' if g < -margin else '<' if g > margin else '=';
      * keep only pairs where measured AND source COMMIT ('>' or '<');
      * agree rate, chance = sum_d p_m(d) p_c(d) over d in {>,<}, Cohen's kappa.
    Basis 'all_rows' is the script's: every opentecr_comparison.csv row (both
    match tiers, 1,361 rows), every record (obsolete duplicates included).
    Basis 'live_anchor_363' is the 363 distinct live stereo-exact reactions.
    Source operators are the SHIPPED ones (thermodynamics[label][2]), i.e. each
    source's own rule set: eQuilibrator/dGPredictor the RI cascade, GC the GC
    cascade; dGbyG's ``op`` from reaction_predictions.tsv was made with the same
    RI cascade (heuristics_for_source('dGPredictor')), only where status=='ok'
    (any other status = dGbyG declined = no call).
    A third view, 'RI truth': the measured energy is put through the SAME RI
    rule set the predictors use, truth = run_reversibility(rxn,
    explicit_energy(dG_exp, max(sd_exp, 0.15)), DGP_HEURISTICS), and each
    source's commit rate, accuracy where both commit, and full confusion
    (>,<,=,? and no-call) are reported on the 363 live anchors.

(B) The README table "Held-out accuracy over 20 random 70/30 splits".
    Produced by SourceGrading/recommend_thermo_source.py ``validate()``. The
    script is first RUN UNCHANGED (via runpy -- it does not import as-is, see
    run_recommend_baseline) to reproduce its numbers on the live database. The
    fixed strategies are then re-implemented here (no fitting needed), checked
    digit-for-digit against that run, and extended with dGbyG:
      eq_only, dgp_only, gc_only, dgb_only,
      priority EQ>DGP>GC (= priority_plain), EQ>DGB>DGP>GC, DGB>EQ>DGP>GC,
      EQ>DGP>GC>DGB.
    README protocol: truth = GC cascade (DEFAULT_HEURISTICS) on
    explicit_energy(opentecr_dg, opentecr_sd or 0); predicted operator = GC
    cascade on each source's own energy; a pick is the first FEASIBLE source
    (sentinels, EQ sigma <= 100, DGP quinone veto) even if its call is '?'; a
    '?' truth is kept and '?' == '?' counts as correct; accuracy = mean(op ==
    truth) over held-out anchors answered; coverage = answered / held-out;
    +/- = SD (ddof 0) over 20 splits drawn from default_rng(20260812).
    Variants, each labelled: operators from each source's OWN rule set (stored
    ops; dGbyG ``op``), scored against the README truth and against the RI
    truth; and a commit-only accuracy (both truth and call in {>,<}), which does
    not depend on how either rule set draws its '=' band.
    Populations: the 801 anchors of the README protocol (438 of them obsolete
    duplicates of live anchors, so 70/30 splits leak copies across train/test;
    harmless for fixed strategies, which fit nothing) and the 363 distinct live
    anchors (leak-free). The 363 basis uses a fresh default_rng(20260812) too.

IN-SAMPLE WARNING. dGbyG's shipped model was trained on TECRDB, the source of
the openTECR anchors, so its anchor accuracy is in-sample. The same holds for
eQuilibrator (component contribution fitted on TECRDB) and dGPredictor (trained
on TECRDB). Group contribution is the only out-of-sample source here.

Writes ONLY to results/direction/ (the SourceGrading env vars are pointed there
before any SourceGrading module is imported; ModelSEEDDatabase is read-only):
  vs_measured.tsv               (A) agree/chance/kappa per basis x margin x source
  vs_measured_confusion.tsv     (A) measured -> call counts at every margin
  ri_truth_summary.tsv          (A) commit rate, accuracy, kappa vs RI truth (363)
  ri_truth_confusion.tsv        (A) truth x call (>,<,=,?,none) per source (363)
  strategy_table.tsv            (B) every population x operator set x truth x strategy
  readme_protocol_confusion.tsv (B) GC-cascade call x GC-cascade truth per source
  anchor_ops.tsv                per-anchor truths, energies, operators (audit trail)
  recommend_baseline/           the unchanged recommend_thermo_source.py run: its log,
                                recommendation_models.json (the validation table) and
                                recommendation_direction.tsv (5.7 MB, a byproduct of
                                running main() unchanged, not an analysis output)
  numbers.json, report.txt      baseline checks and the printed report

Optional input: dGbyG's held-out anchor predictions from cv_predict.py
(results/cv/anchor_heldout.tsv, or the path in $DGBYG_HELDOUT). When present they
add a 'dGbyG held-out CV' row to the 363-anchor tables of (A) and a
dgb_heldout_only strategy to the 363 basis of (B); absent, they are skipped.

Run (about two minutes; reads ModelSEEDDatabase, never writes it):
  ~/Documents/py_venv/bin/python -B analysis/modelseed/direction_accuracy.py
"""
import contextlib
import csv
import json
import math
import os
import runpy
import sys
import types
from collections import Counter, OrderedDict
from pathlib import Path

sys.dont_write_bytecode = True        # never leave __pycache__ in ModelSEEDDatabase

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MSDB = Path(os.environ.get("MSDB_ROOT", REPO.parent / "ModelSEEDDatabase"))
RES = HERE / "results"
OUTD = RES / "direction"
SG = MSDB / "Scripts" / "Thermodynamics" / "SourceGrading"
OPENTECR_CSV = MSDB / "Biochemistry" / "Thermodynamics" / "SourceGrading" / "opentecr_comparison.csv"
# optional: dGbyG's grouped-K-fold held-out anchor predictions (cv_predict.py); used when present
HELDOUT = Path(os.environ.get("DGBYG_HELDOUT", RES / "cv" / "anchor_heldout.tsv"))

# Point every SourceGrading output path at our results dir BEFORE importing it.
# (Without this, grade_thermo_sources.build() would overwrite committed files.)
os.environ["MSDB_ROOT"] = str(MSDB)
os.environ["THERMO_GRADING_OUT"] = str(OUTD / "sourcegrading_scratch")
os.environ["GRADES_OUT"] = str(OUTD / "sourcegrading_scratch" / "thermo_grades")
os.environ["EQDGP_OUT"] = str(OUTD / "sourcegrading_scratch" / "eq_vs_dgp")
os.environ["RECOMMEND_OUT"] = str(OUTD / "recommend_baseline")
sys.path.insert(0, str(SG))
sys.path.insert(0, str(SG.parent))

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
from reversibility_heuristics import (                                # noqa: E402
    DEFAULT_HEURISTICS, DGP_HEURISTICS, explicit_energy, per_source_energy,
    run_reversibility,
)
from optimize_thermo_source_assignment import EQ_SENTINEL, load_db    # noqa: E402
from grade_thermo_sources import load_opentecr, load_vetoes           # noqa: E402

COMMITTED = (">", "<")
CALLS = (">", "<", "=", "?")
MARGINS_KJ = (0.0, 5.0, 11.7)
TRUTH_SD_FLOOR = 0.15          # kcal/mol; recommend_thermo_source.TRUTH_SIGMA
SEED = 20260812                # recommend_thermo_source.RNG
N_SPLITS = 20
LABEL = OrderedDict([("EQ", "eQuilibrator"), ("DGP", "dGPredictor"),
                     ("GC", "Group contribution"), ("DGB", "dGbyG")])

# (A) the script's printed numbers (review_transport_and_llm.py, either mode):
# margin -> source -> (n, agree)
BASELINE_A = {0.0: {"eQuilibrator": (125, 124), "LLMs": (360, 246)},
              5.0: {"eQuilibrator": (125, 124), "LLMs": (263, 200)},
              11.7: {"eQuilibrator": (79, 78), "LLMs": (157, 113)}}
BASELINE_A_CONFUSION = {("<", "<"): 42, ("<", ">"): 44, (">", ">"): 71}
# cross-check from an earlier reader (spec_calibration_llm.md, llm_vs_sources.py --live):
# live records, both tiers, margin 0 -> (n, rate %)
CROSSCHECK_LIVE_BOTH = {"eQuilibrator": (57, 98.2), "dGPredictor": (27, 96.3),
                        "Group contribution": (74, 62.2), "LLMs": (358, 68.2)}
# (B) live-database result of recommend_thermo_source.py --target direction
# (mean accuracy %, SD %, coverage %), rounded as the script prints them
BASELINE_B = {"priority_plain": (87.4, 1.7, 100.0), "eq_only": (87.5, 1.7, 99.8),
              "dgp_only": (90.6, 1.5, 100.0), "gc_only": (60.1, 1.8, 100.0),
              "tau_only_no_risk": (90.9, 1.5, 100.0)}

STRATEGIES = OrderedDict([
    ("eq_only", ["EQ"]), ("dgp_only", ["DGP"]), ("gc_only", ["GC"]), ("dgb_only", ["DGB"]),
    ("priority_plain (EQ>DGP>GC)", ["EQ", "DGP", "GC"]),
    ("EQ>DGB>DGP>GC", ["EQ", "DGB", "DGP", "GC"]),
    ("DGB>EQ>DGP>GC", ["DGB", "EQ", "DGP", "GC"]),
    ("EQ>DGP>GC>DGB", ["EQ", "DGP", "GC", "DGB"]),
    ("null: always '='", ["NULL"]),          # no-information reference: the base rate of '='
])
HELDOUT_STRATEGY = ("dgb_heldout_only", ["DGBCV"])   # 363 basis only, when HELDOUT exists

REPORT = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    REPORT.append(line)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


# --------------------------------------------------------------------- inputs
def load_reactions():
    """{id: entry} over every reaction record, EMPTY and obsolete included."""
    out = {}
    for shard in sorted((MSDB / "Biochemistry").glob("reaction_[0-9][0-9].json")):
        for e in json.load(open(shard)):
            out[e["id"]] = e
    return out


def stored_op(entry, label):
    """review_transport_and_llm.direction(): third slot of the triple, or None."""
    t = (entry.get("thermodynamics") or {}).get(label)
    if not t or len(t) < 3:
        return None
    return t[2] or None


def load_dgbyg():
    """dGbyG's primary estimate per reaction record; only status=='ok' rows."""
    rp = pd.read_csv(RES / "reaction_predictions.tsv", sep="\t", dtype=str, keep_default_na=False)
    ok = rp[rp["status"] == "ok"]
    return {r.rxn: (fnum(r.dG_kcal), fnum(r.sd_kcal), r.op or None, r.op_z0 or None)
            for r in ok.itertuples()}, rp


def is_live(entry):
    return entry is not None and entry.get("is_obsolete") not in (1, "1")


def load_heldout(reactions):
    """dGbyG held-out anchor predictions, if cv_predict.py has written them.

    Rounded to 2 dp in kcal before the direction call, as predict_reactions.py
    does for the primary estimate; operators from the RI rule set (as ``op``)
    and from the GC cascade (for the README protocol)."""
    if not HELDOUT.exists():
        return {}
    h = pd.read_csv(HELDOUT, sep="\t", dtype=str, keep_default_na=False)
    out = {}
    for r in h.itertuples():
        dg, sd = fnum(r.heldout_dG_kcal), fnum(r.heldout_sd_kcal)
        e = reactions.get(r.rxn)
        if e is None or not (np.isfinite(dg) and np.isfinite(sd)):
            continue
        dg, sd = round(dg, 2), round(sd, 2)
        out[r.rxn] = (dg, sd,
                      run_reversibility(e, explicit_energy(dg, sd), DGP_HEURISTICS)[1],
                      run_reversibility(e, explicit_energy(dg, sd), DEFAULT_HEURISTICS)[1],
                      r.in_training)
    return out


# ------------------------------------------------------------ (A) vs measured
def agreement(pairs):
    """review_transport_and_llm._vs_measured statistics over (measured, call)."""
    n = len(pairs)
    if not n:
        return OrderedDict(n=0, agree=0, rate=math.nan, chance=math.nan, kappa=math.nan)
    agree = sum(1 for a, b in pairs if a == b)
    pm, pc = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    chance = sum((pm[d] / n) * (pc[d] / n) for d in COMMITTED)
    kappa = (agree / n - chance) / (1 - chance) if chance < 1 else math.nan
    return OrderedDict(n=n, agree=agree, rate=agree / n, chance=chance, kappa=kappa)


def vs_measured(bases, by_id):
    """Every basis x margin x source; ``bases`` = {basis: (csv rows, callers)}."""
    out, conf = [], []
    for basis, (brows, callers) in bases.items():
        for margin in MARGINS_KJ:
            for src, call in callers.items():
                pairs = []
                for row in brows:
                    rxn = by_id.get(row["modelseed_rxn"])
                    if not rxn:
                        continue
                    g = fnum(row["opentecr_dG_kJ"])
                    if math.isnan(g):
                        continue
                    m = ">" if g < -margin else ("<" if g > margin else "=")
                    c = call(rxn)
                    if m in COMMITTED and c in COMMITTED:
                        pairs.append((m, c))
                out.append(OrderedDict(basis=basis, margin_kJ=margin, source=src, **agreement(pairs)))
                for (a, b), k in sorted(Counter(pairs).items()):
                    conf.append(OrderedDict(basis=basis, margin_kJ=margin, source=src,
                                            measured=a, call=b, n=k))
    return pd.DataFrame(out), pd.DataFrame(conf)


def check_baseline_a(tab, conf):
    ok = True
    for margin, srcs in BASELINE_A.items():
        for src, (n, agree) in srcs.items():
            r = tab[(tab.basis == "all_rows") & (tab.margin_kJ == margin) & (tab.source == src)].iloc[0]
            hit = (int(r.n), int(r.agree)) == (n, agree)
            ok &= hit
            say(f"  margin {margin:<5} {src:<13} expected {n:>4}/{agree:<4} got "
                f"{int(r.n):>4}/{int(r.agree):<4} {100 * r.rate:5.1f}% kappa {r.kappa:.2f}  "
                f"{'OK' if hit else 'MISMATCH'}")
    c = conf[(conf.basis == "all_rows") & (conf.margin_kJ == 11.7) & (conf.source == "LLMs")]
    got = {(a, b): int(k) for a, b, k in zip(c.measured, c.call, c.n)}
    hit = got == BASELINE_A_CONFUSION
    ok &= hit
    say("  LLM confusion at 11.7 kJ: " + ", ".join(f"{a}->{b} {k}" for (a, b), k in sorted(got.items()))
        + f"  {'OK' if hit else 'MISMATCH (expected <->< 42, <->> 44, >->> 71)'}")
    return ok


def ri_truth_tables(anchor_rows, reactions, callers, sd_of):
    """The 363 live anchors scored against the measured energy put through the
    predictors' own RI rule set (z = 1, ln 1000), sd floored at 0.15 kcal."""
    truth = {}
    for row in anchor_rows:
        rid = row["modelseed_rxn"]
        dg = fnum(row["opentecr_dG_kJ"]) / 4.184
        sd = max(fnum(row["opentecr_dG_sd_kJ"]) / 4.184, TRUTH_SD_FLOOR)
        truth[rid] = run_reversibility(reactions[rid], explicit_energy(dg, sd), DGP_HEURISTICS)[1]
    n_all = len(truth)
    summ, conf = [], []
    for src, call in callers.items():
        ops = {rid: call(reactions[rid]) for rid in truth}
        dist = Counter(o if o in CALLS else "none" for o in ops.values())
        both = [(truth[r], ops[r]) for r in truth if truth[r] in COMMITTED and ops[r] in COMMITTED]
        stated = [(truth[r], ops[r]) for r in truth if truth[r] in CALLS[:3] and ops[r] in CALLS[:3]]
        a = agreement(both)
        sds = [s for s in (sd_of[src](r) for r in truth) if s is not None and np.isfinite(s)]
        summ.append(OrderedDict(
            source=src, n_anchors=n_all, n_call=sum(dist[k] for k in CALLS),
            commits=dist[">"] + dist["<"], commit_rate=(dist[">"] + dist["<"]) / n_all,
            n_gt=dist[">"], n_lt=dist["<"], n_eq=dist["="], n_q=dist["?"], n_none=dist["none"],
            both_commit_n=a["n"], both_commit_agree=a["agree"], both_commit_rate=a["rate"],
            both_commit_chance=a["chance"], both_commit_kappa=a["kappa"],
            stated_n=len(stated), stated_exact_rate=(np.mean([x == y for x, y in stated])
                                                     if stated else math.nan),
            median_sd_kcal=float(np.median(sds)) if sds else math.nan))
        cm = Counter((truth[r], ops[r] if ops[r] in CALLS else "none") for r in truth)
        for t in CALLS:
            for c in CALLS + ("none",):
                conf.append(OrderedDict(source=src, truth=t, call=c, n=cm.get((t, c), 0)))
    return pd.DataFrame(summ), pd.DataFrame(conf), truth


# --------------------------------------------- (B) recommend_thermo_source run
def run_recommend_baseline(reactions):
    """Run SourceGrading/recommend_thermo_source.py UNCHANGED (--target direction).

    It does not import as shipped: L106 ``ANALYSIS_DIR = ANALYSIS_DIR`` raises
    NameError (satisfied via runpy init_globals; the module then re-derives the
    real paths from THERMO_GRADING_OUT / RECOMMEND_OUT, both pointed at our
    results dir), and main() imports ``build_graded_direction_maps``, which
    stayed in core_models_analysis (supplied here as an in-memory module whose
    load_reactions() returns the reaction records already loaded)."""
    out = Path(os.environ["RECOMMEND_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    shim = types.ModuleType("build_graded_direction_maps")
    shim.load_reactions = lambda: reactions
    sys.modules["build_graded_direction_maps"] = shim
    argv = sys.argv
    sys.argv = ["recommend_thermo_source.py", "--target", "direction"]
    try:
        with open(out / "recommend.log", "w") as log, contextlib.redirect_stdout(log):
            runpy.run_path(str(SG / "recommend_thermo_source.py"), run_name="__main__",
                           init_globals={"ANALYSIS_DIR": Path(os.environ["THERMO_GRADING_OUT"]),
                                         "MSDB_CODE": MSDB})
    finally:
        sys.argv = argv
    val = json.loads((out / "recommendation_models.json").read_text())["validation"]
    return pd.DataFrame(val)


def check_baseline_b(val):
    ok = True
    v = val.set_index("strategy")
    for s, (acc, sd, cov) in BASELINE_B.items():
        got = (round(100 * v.at[s, "mean_accuracy"], 1), round(100 * v.at[s, "sd_accuracy"], 1),
               round(100 * v.at[s, "mean_coverage"], 1))
        hit = got == (acc, sd, cov)
        ok &= hit
        say(f"  {s:<18} expected {acc:5.1f} +/- {sd:.1f} cov {cov:5.1f} | got {got[0]:5.1f} +/- "
            f"{got[1]:.1f} cov {got[2]:5.1f}  {'OK' if hit else 'MISMATCH'}")
    return ok


# --------------------------------------------- (B) fixed-strategy re-implementation
def build_anchor_frame(reactions, dgb, heldout):
    """The README anchor in load_db row order (the split order), with every
    source's energy, feasibility, both operator sets and both truths."""
    db = load_db().reset_index(drop=True)
    tec = db[["rxn"]].merge(load_opentecr(), on="rxn", how="left")
    tec.index = db.index
    db["opentecr_dg"], db["opentecr_sd"] = tec["opentecr_dg"], tec["opentecr_sd"]
    anchor = db[tec["match_tier"].eq("stereo_exact").to_numpy() & db.opentecr_dg.notna()].copy()
    veto_eq = load_vetoes()

    a = pd.DataFrame({"rxn": anchor.rxn.values, "db_index": anchor.index.values})
    a["is_live"] = [is_live(reactions.get(r)) for r in a.rxn]
    a["dg_exp"], a["sd_exp"] = anchor.opentecr_dg.values, anchor.opentecr_sd.values
    for k in ("EQ", "DGP", "GC"):
        a[f"dg_{k}"], a[f"sig_{k}"] = anchor[f"dg_{k}"].values, anchor[f"sig_{k}"].values
        a[f"own_op_{k}"] = [o if isinstance(o, str) else "" for o in anchor[f"op_{k}"].values]
    a["dg_DGB"] = [dgb.get(r, (np.nan,) * 4)[0] for r in a.rxn]
    a["sig_DGB"] = [dgb.get(r, (np.nan,) * 4)[1] for r in a.rxn]
    a["own_op_DGB"] = [dgb.get(r, (None,) * 4)[2] or "" for r in a.rxn]

    # feasibility exactly as recommend_thermo_source.feasibility(); dGbyG: has a primary estimate
    for k in LABEL:
        f = a[f"dg_{k}"].notna() & a[f"sig_{k}"].notna()
        if k == "EQ":
            f &= (a["sig_EQ"] <= EQ_SENTINEL) & ~a["rxn"].isin(veto_eq)
        if k == "DGP":
            f &= anchor["is_quinone"].values == 0
        a[f"feas_{k}"] = f.values

    # predicted operators, README protocol: GC cascade on each source's own energy
    for k in LABEL:
        col = []
        for r in a.itertuples():
            e = reactions.get(r.rxn)
            if e is None or e.get("status") == "EMPTY":
                col.append("")
                continue
            if k != "DGB":
                src = per_source_energy(LABEL[k])
            elif np.isfinite(r.dg_DGB):
                src = explicit_energy(r.dg_DGB, r.sig_DGB)
            else:                              # dGbyG declined: no energy, no call
                col.append("")
                continue
            col.append(run_reversibility(e, src, DEFAULT_HEURISTICS)[1] or "")
        a[f"gc_op_{k}"] = col

    # the no-information reference strategy, and dGbyG's held-out CV values when present
    a["feas_NULL"], a["gc_op_NULL"], a["own_op_NULL"] = True, "=", "="
    if heldout:
        a["dg_DGBCV"] = [heldout.get(r, (np.nan,) * 5)[0] for r in a.rxn]
        a["sig_DGBCV"] = [heldout.get(r, (np.nan,) * 5)[1] for r in a.rxn]
        a["own_op_DGBCV"] = [heldout.get(r, ("",) * 5)[2] or "" for r in a.rxn]
        a["gc_op_DGBCV"] = [heldout.get(r, ("",) * 5)[3] or "" for r in a.rxn]
        a["in_training_DGBCV"] = [heldout.get(r, ("",) * 5)[4] for r in a.rxn]
        a["feas_DGBCV"] = a["dg_DGBCV"].notna() & a["sig_DGBCV"].notna()

    # truths: README (GC cascade, sd or 0) and RI (predictors' rule set, sd floored at 0.15)
    t2 = load_opentecr()
    se = t2[t2.match_tier == "stereo_exact"].set_index("rxn")
    gc_truth, ri_truth = {}, {}
    for rid, r in se.iterrows():
        e = reactions.get(rid)
        if e is None or e.get("status") == "EMPTY":
            continue
        op = run_reversibility(e, explicit_energy(float(r.opentecr_dg), float(r.opentecr_sd or 0.0)),
                               DEFAULT_HEURISTICS)[1]
        if op:
            gc_truth[rid] = op
        op = run_reversibility(e, explicit_energy(float(r.opentecr_dg),
                                                  max(float(r.opentecr_sd), TRUTH_SD_FLOOR)),
                               DGP_HEURISTICS)[1]
        if op:
            ri_truth[rid] = op
    a["truth_gc"] = a.rxn.map(gc_truth)
    a["truth_ri"] = a.rxn.map(ri_truth)
    return a


def evaluate(a, ops_prefix, ops_label, truth_col, population, strategies, seed=SEED):
    """recommend_thermo_source.validate() for the fixed strategies, over the
    rows of ``a`` (already restricted to the population, in db order)."""
    n_pop = len(a)
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(n_pop) for _ in range(N_SPLITS)]
    truth = a[truth_col].to_numpy(dtype=object)
    rows = []
    for name, order in strategies.items():
        ans = np.full(n_pop, None, dtype=object)
        for i in range(n_pop):
            pick = next((k for k in order if a[f"feas_{k}"].iat[i]), None)
            if pick is None:
                continue
            op = a[f"{ops_prefix}{pick}"].iat[i]
            if op:
                ans[i] = op
        has_truth = np.array([isinstance(t, str) for t in truth])
        answered = has_truth & np.array([x is not None for x in ans])
        correct = answered & (ans == truth)
        commit = answered & np.isin(truth, COMMITTED) & np.isin(ans, COMMITTED)
        acc, cov = [], []
        for p in perms:
            te = p[int(0.7 * n_pop):]
            n = int(answered[te].sum())
            if n:
                acc.append(correct[te].sum() / n)
                cov.append(n / len(te))
        nc = int(commit.sum())
        rows.append(OrderedDict(
            population=population, operators=ops_label,
            truth=truth_col, strategy=name,
            mean_accuracy=float(np.mean(acc)), sd_accuracy=float(np.std(acc)),
            mean_coverage=float(np.mean(cov)),
            full_n=n_pop, full_answered=int(answered.sum()), full_correct=int(correct.sum()),
            full_accuracy=correct.sum() / answered.sum() if answered.sum() else math.nan,
            commit_n=nc, commit_correct=int((commit & correct).sum()),
            commit_accuracy=(commit & correct).sum() / nc if nc else math.nan))
    return rows


def check_reimplementation(tab, val):
    """Our evaluator on the README protocol must equal validate() exactly."""
    ok = True
    v = val.set_index("strategy")
    mine = tab[(tab.population == "801_all_anchors") & (tab.operators == "gc_cascade")
               & (tab.truth == "truth_gc")].set_index("strategy")
    for theirs, ours in (("eq_only", "eq_only"), ("dgp_only", "dgp_only"), ("gc_only", "gc_only"),
                         ("priority_plain", "priority_plain (EQ>DGP>GC)")):
        d = max(abs(v.at[theirs, c] - mine.at[ours, c])
                for c in ("mean_accuracy", "sd_accuracy", "mean_coverage"))
        ok &= d < 1e-12
        say(f"  {theirs:<16} validate() {100 * v.at[theirs, 'mean_accuracy']:.4f}% "
            f"+/- {100 * v.at[theirs, 'sd_accuracy']:.4f} cov {100 * v.at[theirs, 'mean_coverage']:.4f} | "
            f"ours {100 * mine.at[ours, 'mean_accuracy']:.4f}% +/- {100 * mine.at[ours, 'sd_accuracy']:.4f} "
            f"cov {100 * mine.at[ours, 'mean_coverage']:.4f}  max|diff| {d:.1e}  {'OK' if d < 1e-12 else 'MISMATCH'}")
    return ok


# ----------------------------------------------------------------------- main
def fmt_pct(x):
    return "  n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:5.1f}"


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 500)
    numbers = OrderedDict()
    reactions = load_reactions()
    dgb, rp = load_dgbyg()
    say(f"reaction records {len(reactions):,}; dGbyG primary estimates (status ok) {len(dgb):,}")

    # ---------------- (A) vs measured
    csv_rows = list(csv.DictReader(open(OPENTECR_CSV)))
    live_anchor_rows = [r for r in csv_rows if r["match_tier"] == "stereo_exact"
                        and is_live(reactions.get(r["modelseed_rxn"]))]
    assert len(live_anchor_rows) == 363 and len({r["modelseed_rxn"] for r in live_anchor_rows}) == 363
    callers = OrderedDict([
        ("eQuilibrator", lambda e: stored_op(e, "eQuilibrator")),
        ("dGPredictor", lambda e: stored_op(e, "dGPredictor")),
        ("Group contribution", lambda e: stored_op(e, "Group contribution")),
        ("dGbyG", lambda e: dgb.get(e["id"], (None,) * 4)[2]),
        ("dGbyG op_z0 (sensitivity)", lambda e: dgb.get(e["id"], (None,) * 4)[3]),
        ("LLMs", lambda e: stored_op(e, "LLMs")),
    ])
    heldout = load_heldout(reactions)
    anchor_callers = OrderedDict(callers)
    if heldout:
        # anchor-only by construction, so it is scored on the 363 basis only
        anchor_callers["dGbyG held-out CV"] = lambda e: heldout.get(e["id"], (None,) * 5)[2]
        numbers["heldout"] = {"file": str(HELDOUT), "n": len(heldout),
                              "in_training": dict(Counter(v[4] for v in heldout.values()))}
    say(f"dGbyG held-out CV anchor predictions: "
        f"{(str(len(heldout)) + ' from ' + str(HELDOUT)) if heldout else 'not available (' + str(HELDOUT) + ' absent)'}")
    # the TSV's copies of the shipped operators must be the JSON's (they feed compare.py)
    tsv = rp.set_index("rxn")
    for col, lab in (("eq_op", "eQuilibrator"), ("dgp_op", "dGPredictor"),
                     ("gc_op", "Group contribution"), ("llm_op", "LLMs")):
        bad = sum(1 for rid, e in reactions.items() if (tsv.at[rid, col] or None) != stored_op(e, lab))
        assert bad == 0, (col, bad)

    live_rows = [r for r in csv_rows if is_live(reactions.get(r["modelseed_rxn"]))]
    tab_a, conf_a = vs_measured({"all_rows": (csv_rows, callers),
                                 "live_rows_both_tiers": (live_rows, callers),
                                 "live_anchor_363": (live_anchor_rows, anchor_callers)}, reactions)
    say("\n(A0) BASELINE: review_transport_and_llm._vs_measured, all 1,361 CSV rows, all records")
    numbers["baseline_A_reproduced"] = bool(check_baseline_a(tab_a, conf_a))
    say("  cross-check, live records both tiers margin 0 (spec_calibration_llm.md): " + "; ".join(
        f"{s} expected {n}/{p:.1f}% got {int(r.n)}/{100 * r.rate:.1f}%"
        for s, (n, p) in CROSSCHECK_LIVE_BOTH.items()
        for r in [tab_a[(tab_a.basis == "live_rows_both_tiers") & (tab_a.margin_kJ == 0.0)
                        & (tab_a.source == s)].iloc[0]]))
    numbers["crosscheck_live_both_reproduced"] = all(
        (int(r.n), round(100 * r.rate, 1)) == v for s, v in CROSSCHECK_LIVE_BOTH.items()
        for r in [tab_a[(tab_a.basis == "live_rows_both_tiers") & (tab_a.margin_kJ == 0.0)
                        & (tab_a.source == s)].iloc[0]])
    tab_a.to_csv(OUTD / "vs_measured.tsv", sep="\t", index=False, float_format="%.4f")
    conf_a.to_csv(OUTD / "vs_measured_confusion.tsv", sep="\t", index=False)
    for basis, lab in (("all_rows", "script protocol: all 1,361 CSV rows, both tiers, all records"),
                       ("live_rows_both_tiers", f"live records only, both tiers ({len(live_rows)} CSV rows)"),
                       ("live_anchor_363", "363 distinct live stereo-exact anchors")):
        say(f"\n(A1) direction vs MEASURED, {lab}; pairs where both commit")
        say(f"  {'margin':<7}{'source':<27}{'n':>5}{'agree':>7}{'rate':>8}{'chance':>8}{'kappa':>7}")
        for r in tab_a[tab_a.basis == basis].itertuples():
            say(f"  {r.margin_kJ:<7}{r.source:<27}{r.n:>5}{r.agree:>7}{fmt_pct(r.rate):>7}%"
                f"{fmt_pct(r.chance):>7}%{r.kappa:>7.2f}")
        c = conf_a[(conf_a.basis == basis) & (conf_a.margin_kJ == 11.7)]
        say("  confusion at 11.7 kJ (measured->call): " + " | ".join(
            f"{s}: " + ", ".join(f"{x.measured}->{x.call} {x.n}" for x in c[c.source == s].itertuples())
            for s in c.source.unique()))

    # sigma medians: shipped |sigma|, sentinels (GC dg 1e7, eQ sigma >= 2500) excluded
    def shipped_sd(lab):
        def f(rid):
            t = (reactions[rid].get("thermodynamics") or {}).get(lab)
            if not t or len(t) < 2:
                return None
            dg, sd = fnum(t[0]), abs(fnum(t[1]))
            return None if (dg == 1e7 or sd >= 2500) else sd
        return f
    sd_of = {lab: shipped_sd(lab) for lab in ("eQuilibrator", "dGPredictor", "Group contribution")}
    sd_of["dGbyG"] = sd_of["dGbyG op_z0 (sensitivity)"] = lambda rid: dgb.get(rid, (None,) * 4)[1]
    sd_of["LLMs"] = lambda rid: None
    sd_of["dGbyG held-out CV"] = lambda rid: heldout.get(rid, (None,) * 5)[1]
    ri_sum, ri_conf, ri_truth = ri_truth_tables(live_anchor_rows, reactions, anchor_callers, sd_of)
    ri_sum.to_csv(OUTD / "ri_truth_summary.tsv", sep="\t", index=False, float_format="%.4f")
    ri_conf.to_csv(OUTD / "ri_truth_confusion.tsv", sep="\t", index=False)
    td = Counter(ri_truth.values())
    numbers["ri_truth_distribution_363"] = dict(td)
    say(f"\n(A2) RI-rule truth on the 363 live anchors: truth = RI cascade (DGP_HEURISTICS, z=1) on "
        f"explicit_energy(dG_exp, max(sd_exp, 0.15)); truth distribution {dict(td)}")
    say(f"  {'source':<27}{'call':>5}{'commit':>8}{'rate':>7}{'>':>5}{'<':>5}{'=':>5}{'?':>5}{'none':>6}"
        f"{'both-commit n':>15}{'agree':>7}{'rate':>7}{'kappa':>7}{'stated n':>10}{'exact':>7}{'med sd':>8}")
    for r in ri_sum.itertuples():
        say(f"  {r.source:<27}{r.n_call:>5}{r.commits:>8}{fmt_pct(r.commit_rate):>7}{r.n_gt:>5}{r.n_lt:>5}"
            f"{r.n_eq:>5}{r.n_q:>5}{r.n_none:>6}{r.both_commit_n:>15}{r.both_commit_agree:>7}"
            f"{fmt_pct(r.both_commit_rate):>7}{r.both_commit_kappa:>7.2f}{r.stated_n:>10}"
            f"{fmt_pct(r.stated_exact_rate):>7}{r.median_sd_kcal:>8.2f}")
    say("  confusion (rows truth >,<,=,?; columns call >,<,=,?,none):")
    for s in anchor_callers:
        c = ri_conf[ri_conf.source == s].pivot(index="truth", columns="call", values="n")
        c = c.reindex(index=list(CALLS), columns=list(CALLS) + ["none"])
        say(f"   {s}: " + " | ".join(f"{t}: " + " ".join(f"{int(c.at[t, x]):>3}" for x in c.columns)
                                    for t in CALLS))

    # ---------------- (B) README strategy table
    say("\n(B0) BASELINE: recommend_thermo_source.py --target direction, run unchanged "
        "(801 anchors, 20 x 70/30, default_rng(20260812))")
    val = run_recommend_baseline(reactions)
    numbers["baseline_B_reproduced"] = bool(check_baseline_b(val))
    say("  all strategies: " + "; ".join(f"{r.strategy} {100 * r.mean_accuracy:.1f}+/-{100 * r.sd_accuracy:.1f}"
                                         f" cov {100 * r.mean_coverage:.1f}"
                                         for r in val.sort_values("mean_accuracy", ascending=False).itertuples()))
    numbers["recommend_validation"] = val.to_dict("records")

    a = build_anchor_frame(reactions, dgb, heldout)
    a.drop(columns=["feas_NULL", "gc_op_NULL", "own_op_NULL"]).to_csv(
        OUTD / "anchor_ops.tsv", sep="\t", index=False, float_format="%.4f")
    live = a[a.is_live].reset_index(drop=True)
    numbers["anchors"] = {"all": len(a), "live": len(live)}
    assert len(a) == 801 and len(live) == 363, (len(a), len(live))
    for lab, d in (("801", a), ("363", live)):
        numbers[f"truth_gc_distribution_{lab}"] = dict(Counter(d.truth_gc.dropna()))
        numbers[f"feasible_{lab}"] = {k: int(d[f"feas_{k}"].sum()) for k in LABEL}
        numbers[f"gc_cascade_ops_{lab}"] = {k: dict(Counter(d[f"gc_op_{k}"])) for k in LABEL}
        numbers[f"own_ops_{lab}"] = {k: dict(Counter(d[f"own_op_{k}"])) for k in LABEL}
    rows = []
    strat_live = OrderedDict(STRATEGIES)
    if heldout:
        strat_live[HELDOUT_STRATEGY[0]] = HELDOUT_STRATEGY[1]
    for pop, d, strat in (("801_all_anchors", a, STRATEGIES), ("363_live_anchors", live, strat_live)):
        rows += evaluate(d, "gc_op_", "gc_cascade", "truth_gc", pop, strat)
        rows += evaluate(d, "own_op_", "own_rule_set", "truth_gc", pop, strat)
        rows += evaluate(d, "own_op_", "own_rule_set", "truth_ri", pop, strat)
    tab_b = pd.DataFrame(rows)
    tab_b.to_csv(OUTD / "strategy_table.tsv", sep="\t", index=False, float_format="%.4f")
    say("\n(B1) re-implementation check against validate() (README protocol, 801 anchors)")
    numbers["reimplementation_matches_validate"] = bool(check_reimplementation(tab_b, val))

    say(f"\n  truth distribution: GC-cascade truth 801 {numbers['truth_gc_distribution_801']}, "
        f"363 {numbers['truth_gc_distribution_363']}")
    say(f"  feasible 801 {numbers['feasible_801']}; 363 {numbers['feasible_363']}")
    for lab in ("801", "363"):
        say(f"  GC-cascade operators {lab}: {numbers[f'gc_cascade_ops_{lab}']}")
        say(f"  own-rule-set operators {lab}: {numbers[f'own_ops_{lab}']}")
    # where the README-protocol misses come from: the call x truth table per source,
    # next to each source's median |dG - dG_exp| on the same anchors
    diag = []
    for pop, d in (("801_all_anchors", a), ("363_live_anchors", live)):
        say(f"\n  README-protocol confusion, {pop} (rows GC-cascade truth >,<,=,?; columns call >,<,=,?,none)"
            f" and median |dG - dG_exp| (kcal/mol)")
        for k in LABEL:
            call = d[f"gc_op_{k}"].where(d[f"gc_op_{k}"] != "", "none")
            cm = Counter(zip(d.truth_gc, call))
            for t in CALLS:
                for c in CALLS + ("none",):
                    diag.append(OrderedDict(population=pop, source=LABEL[k], truth=t, call=c,
                                            n=cm.get((t, c), 0)))
            err = (d[f"dg_{k}"] - d.dg_exp).abs()
            say(f"   {LABEL[k]:<19} median|err| {err.median():5.2f} median sigma {d[f'sig_{k}'].median():5.2f} | "
                + " | ".join(f"{t}: " + " ".join(f"{cm.get((t, c), 0):>3}" for c in CALLS + ("none",))
                             for t in CALLS))
            numbers.setdefault(f"median_abs_err_{pop}", {})[LABEL[k]] = float(err.median())
    pd.DataFrame(diag).to_csv(OUTD / "readme_protocol_confusion.tsv", sep="\t", index=False)

    titles = {("gc_cascade", "truth_gc"): "README protocol: GC-cascade operators vs GC-cascade truth",
              ("own_rule_set", "truth_gc"): "own-rule-set operators (stored; dGbyG op) vs GC-cascade truth",
              ("own_rule_set", "truth_ri"): "own-rule-set operators vs RI-rule truth (sd floor 0.15)"}
    for pop in ("801_all_anchors", "363_live_anchors"):
        for (ops, tr), title in titles.items():
            t = tab_b[(tab_b.population == pop) & (tab_b.operators == ops) & (tab_b.truth == tr)]
            say(f"\n(B2) {pop} | {title}")
            say(f"  {'strategy':<28}{'acc 20x70/30':>14}{'cov':>7}{'full acc':>10}{'answered':>10}"
                f"{'commit-only acc':>17}")
            for r in t.itertuples():
                say(f"  {r.strategy:<28}{fmt_pct(r.mean_accuracy):>7} +/-{100 * r.sd_accuracy:4.1f}"
                    f"{fmt_pct(r.mean_coverage):>7}{fmt_pct(r.full_accuracy):>10}"
                    f"{r.full_correct:>5}/{r.full_answered:<4}{fmt_pct(r.commit_accuracy):>9} "
                    f"({r.commit_correct}/{r.commit_n})")

    with open(OUTD / "numbers.json", "w") as fh:
        json.dump(numbers, fh, indent=1, default=str)
    (OUTD / "report.txt").write_text("\n".join(REPORT) + "\n")
    say(f"\nwrote {OUTD}")


if __name__ == "__main__":
    main()
