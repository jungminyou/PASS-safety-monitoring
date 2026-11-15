"""Table 2: one primary P per feature; BH correction across 21 primary P values."""

import os

# Keep model warnings visible.
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from config import PRIVATE_DATA, RESULTS
from scipy import stats
from statsmodels.stats.multitest import multipletests

DATA = str(PRIVATE_DATA)
OUTR = str(RESULTS)
z = np.load(os.path.join(DATA, "valid_cube.npz"), allow_pickle=True)
PIDS, LOCS = (z["pids"], list(z["locs"]))
ir_rate, valid_min, installed = (
    z["ir_rate"],
    z["valid_min"],
    z["installed"],
)
NP, T = ir_rate.shape[:2]
N = NP * T
WT_H = list(range(8, 20))
ST_H = list(range(20, 24)) + list(range(8))
L2I = {l: j for j, l in enumerate(LOCS)}
ic = pd.read_csv(
    os.path.join(DATA, "safety_dataset_ircount_20230728.csv"), encoding="utf-8"
)
assert ic.P_ID.nunique() == 67, "Expected the approved 67-dyad analysis cohort."
assert len(ic) == 938 and (not ic.duplicated(["P_ID", "Days"]).any()), (
    "Expected 14 unique days per dyad."
)
assert ic.groupby("P_ID").Days.apply(lambda x: set(x) == set(range(1, 15))).all()
assert ic.Event.isin([0, 1]).all()
assert ic.groupby("P_ID").Event.max().sum() == 24
ic = ic.sort_values(["P_ID", "Days"]).reset_index(drop=True)
assert np.array_equal(PIDS, np.sort(ic.P_ID.unique())), (
    "Cube and CSV dyad order differ."
)
pid = ic.P_ID.values
y = ic["Event"].astype(int).values
SLEEP = [
    "TST",
    "WASO",
    "Efficiency",
    "NoA",
    "MAL",
    "TST_night",
    "WASO_night",
    "Efficiency_night",
    "NoA_night",
    "MAL_night",
    "Onset Latency",
]
SLEEP_LAB = {
    "TST": "Total sleep time (min)",
    "WASO": "WASO (min)",
    "Efficiency": "Sleep efficiency (%)",
    "NoA": "No. of awakenings",
    "MAL": "Mean awakening length (min)",
    "TST_night": "TST at night (min)",
    "WASO_night": "WASO at night (min)",
    "Efficiency_night": "Sleep efficiency at night (%)",
    "NoA_night": "No. of awakenings at night",
    "MAL_night": "Mean awakening length at night (min)",
    "Onset Latency": "Sleep onset latency (min)",
}
ROOM = {
    "bed": "Bedroom",
    "living": "Living room",
    "bath": "Bathroom",
    "door": "Front door",
    "kitchen": "Kitchen",
}


def loc_rates():
    out, names, meta = ([], [], [])
    for l in LOCS:
        j = L2I[l]
        for per, hrs in [("WT", WT_H), ("ST", ST_H)]:
            cnt = np.nansum(
                np.nan_to_num(ir_rate[:, :, j, :][:, :, hrs])
                * valid_min[:, :, hrs],
                axis=2,
            )
            vm = np.nansum(
                valid_min[:, :, hrs] * ~np.isnan(ir_rate[:, :, j, :][:, :, hrs]),
                axis=2,
            )
            v = np.where(vm > 0, cnt / np.maximum(vm, 1e-09), np.nan)
            v = np.where(installed[:, j][:, None] == 1, v, np.nan)
            out.append(v.reshape(N))
            names.append(f"{ROOM[l]} {('daytime' if per == 'WT' else 'nighttime')}")
            meta.append(int(installed[:, j].sum()))
    return (np.column_stack(out), names, meta)


RATE, LOCN, LOCMETA = loc_rates()
SLPV = ic[SLEEP].values.astype(float)


def welch(v, yv):
    a, b = (v[(yv == 1) & ~np.isnan(v)], v[(yv == 0) & ~np.isnan(v)])
    if len(a) < 3 or len(b) < 3:
        return (np.nan, np.nan, np.nan)
    return (a.mean(), b.mean(), stats.ttest_ind(a, b, equal_var=False).pvalue)


def mixed_p(v, yv, gp):
    d = pd.DataFrame({"v": v, "e": yv, "g": gp}).dropna()
    if d.v.nunique() < 3 or d.g.nunique() < 5:
        return (np.nan, np.nan)
    try:
        m = smf.mixedlm("v ~ e", d, groups=d["g"]).fit(reml=False, method="lbfgs")
        if not m.converged:
            return np.nan, np.nan
        return (float(m.params.get("e", np.nan)), float(m.pvalues.get("e", np.nan)))
    except Exception:
        return (np.nan, np.nan)


def gee_p(v, yv, gp):
    d = pd.DataFrame({"v": v, "e": yv, "g": gp}).dropna()
    if d.v.nunique() < 3 or d.g.nunique() < 5:
        return (np.nan, np.nan)
    try:
        m = sm.GEE(
            d.v,
            sm.add_constant(d[["e"]]),
            groups=d.g,
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
        return (float(m.params["e"]), float(m.pvalues["e"]))
    except Exception:
        return (np.nan, np.nan)


rows, pw_all, pm_all = ([], [], [])
raw_means = []
primary_sources = []
for j, c in enumerate(SLEEP):
    v = SLPV[:, j]
    ma, mb, pw = welch(v, y)
    raw_means.append((ma, mb))
    bm, pm = mixed_p(v, y, pid)
    _, pg = gee_p(v, y, pid)
    nobs = int((~np.isnan(v)).sum())
    rows.append(
        [
            "Sleep (actigraphy)",
            SLEEP_LAB[c],
            NP,
            nobs,
            f"{ma:.2f}" if not np.isnan(ma) else "-",
            f"{mb:.2f}" if not np.isnan(mb) else "-",
            f"{pw:.3f}" if not np.isnan(pw) else "-",
            f"{bm:+.3f}" if not np.isnan(bm) else "-",
            f"{pm:.3f}" if not np.isnan(pm) else "-",
            f"{pg:.3f}" if not np.isnan(pg) else "-",
        ]
    )
    primary_sources.append("mixed model" if not np.isnan(pm) else "GEE fallback")
    pw_all.append(pw)
    pm_all.append(pm if not np.isnan(pm) else pg)
for j, nm in enumerate(LOCN):
    v = RATE[:, j]
    ninst = LOCMETA[j]
    ma, mb, pw = welch(v, y)
    raw_means.append((ma, mb))
    bm, pm = mixed_p(v, y, pid)
    _, pg = gee_p(v, y, pid)
    nobs = int((~np.isnan(v)).sum())
    rows.append(
        [
            "IR sensor (per valid actigraphy minute)",
            nm,
            ninst,
            nobs,
            f"{ma:.4f}" if not np.isnan(ma) else "-",
            f"{mb:.4f}" if not np.isnan(mb) else "-",
            f"{pw:.3f}" if not np.isnan(pw) else "-",
            f"{bm:+.4f}" if not np.isnan(bm) else "-",
            f"{pm:.3f}" if not np.isnan(pm) else "-",
            f"{pg:.3f}" if not np.isnan(pg) else "-",
        ]
    )
    primary_sources.append("mixed model" if not np.isnan(pm) else "GEE fallback")
    pw_all.append(pw)
    pm_all.append(pm if not np.isnan(pm) else pg)
t2 = pd.DataFrame(
    rows,
    columns=[
        "Domain",
        "Variable",
        "n participants contributing",
        "n days observed",
        "Incident days (mean)",
        "Non-incident days (mean)",
        "P (Welch t, clustering ignored)",
        "Mixed-model β",
        "P (mixed model)",
        "P (GEE)",
    ],
)
t2["Incident mean unrounded"] = [x[0] for x in raw_means]
t2["Non-incident mean unrounded"] = [x[1] for x in raw_means]
t2["Primary P"] = pm_all
t2["Primary model"] = primary_sources
pw_arr = np.array([p if p is not None else np.nan for p in pw_all], float)
pm_arr = np.array([p if p is not None else np.nan for p in pm_all], float)
for arr, col in [(pw_arr, "q (FDR, Welch)"), (pm_arr, "q (FDR, clustered)")]:
    q = np.full(len(arr), np.nan)
    ok = ~np.isnan(arr)
    if ok.sum():
        q[ok] = multipletests(arr[ok], alpha=0.05, method="fdr_bh")[1]
    t2[col] = [f"{x:.3f}" if not np.isnan(x) else "-" for x in q]
t2.to_csv(
    os.path.join(OUTR, "TableAE_table2_corrected.csv"),
    index=False,
    encoding="utf-8-sig",
)
