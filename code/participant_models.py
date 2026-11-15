"""Table 3: nested participant-level models, matched permutation tests, and descriptive thresholds."""
import os, pickle, warnings
from config import RESULTS, STATE
from collections import Counter
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import raw_features as R
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score, confusion_matrix, roc_curve, matthews_corrcoef
RNG = 42
SEEDS = [42, 7]
N_PERM = int(os.environ.get('PASS_PERMUTATIONS', '2000'))
if N_PERM < 0:
    raise ValueError('Permutation count must be nonnegative.')
INNER_K = 4
OUTR = str(RESULTS)
y, pid, day = (R.y, R.pid, R.day)
PIDS = np.unique(pid)
NP = len(PIDS)
ROWI = {p: np.where(pid == p)[0] for p in PIDS}
ORD = {p: ROWI[p][np.argsort(day[ROWI[p]])] for p in PIDS}
yp = np.array([int(y[ROWI[p]].sum() > 0) for p in PIDS])

def slope(v):
    m = ~np.isnan(v)
    if m.sum() < 3:
        return np.nan
    t = np.arange(len(v), dtype=float)[m]
    x = v[m]
    t = t - t.mean()
    d = (t ** 2).sum()
    return np.nan if d == 0 else float((t * (x - x.mean())).sum() / d)
CLIN_P = np.array([np.asarray(R.CLIN, float)[ORD[p][0]] for p in PIDS])
CLIN_N = list(R.BASE)
INV = np.asarray(R.INV, float)
INV_N = list(R.INV_COLS)
SENS_P, SENS_N = ([], [])
for j, nm in enumerate(INV_N):
    M = np.array([INV[ORD[p], j] for p in PIDS])
    SENS_P.append(np.nanmean(M, 1))
    SENS_N.append(f'{nm}__mean14')
    SENS_P.append(np.nanstd(M, 1))
    SENS_N.append(f'{nm}__sd14')
    SENS_P.append(np.array([slope(M[i]) for i in range(NP)]))
    SENS_N.append(f'{nm}__slope14')
SENS_P = np.column_stack(SENS_P)
COMPACT = ['ir_per_awmin__mean14', 'ir_night_frac__mean14', 'ir_hour_entropy__mean14', 'ir_active_night_hours__mean14', 'ir_night_bouts__mean14', 'ir_per_awmin__sd14', 'ir_night_frac__sd14', 'ir_per_awmin__slope14', 'aw_move_minutes__mean14']
ci_ = [SENS_N.index(c) for c in COMPACT]
SENSC_P, SENSC_N = (SENS_P[:, ci_], COMPACT)
SLP = np.asarray(R.sleep_raw, float)
SLP_N = list(R.SLEEP)
SLEEP_P, SLEEP_N = ([], [])
for j, nm in enumerate(SLP_N):
    M = np.array([SLP[ORD[p], j] for p in PIDS])
    SLEEP_P.append(np.nanmean(M, 1))
    SLEEP_N.append(f'{nm}__mean14')
    SLEEP_P.append(np.nanstd(M, 1))
    SLEEP_N.append(f'{nm}__sd14')
SLEEP_P = np.column_stack(SLEEP_P)
FS = {'P1  Clinical baseline only': (CLIN_P, CLIN_N), 'P2  Clinical + sensor summary (compact)': (np.hstack([CLIN_P, SENSC_P]), CLIN_N + SENSC_N), 'P4  Clinical + sleep summary': (np.hstack([CLIN_P, SLEEP_P]), CLIN_N + SLEEP_N), 'P5  Clinical + sensor + sleep': (np.hstack([CLIN_P, SENSC_P, SLEEP_P]), CLIN_N + SENSC_N + SLEEP_N), 'P6  Sensor summary only (no clinical)': (SENSC_P, SENSC_N)}
CAND = [('L2', c) for c in (0.01, 0.03, 0.1, 0.3, 1.0)] + [('L1', c) for c in (0.03, 0.1, 0.3)]

def make(kind, c, seed):
    return Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', MinMaxScaler()), ('c', LogisticRegression(penalty='l1' if kind == 'L1' else 'l2', C=c, solver='liblinear', class_weight='balanced', max_iter=5000, random_state=seed))])

def nested_oof(X, seed, yv=yp, picks=None):
    o = np.full(NP, np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, yv, groups=PIDS):
        best, bsc = (CAND[0], -np.inf)
        for cand in CAND:
            oi = np.full(len(tr), np.nan)
            ok = True
            for itr, ite in StratifiedGroupKFold(INNER_K, shuffle=True, random_state=seed + 1).split(X[tr], yv[tr], groups=PIDS[tr]):
                if yv[tr][itr].sum() < 2:
                    ok = False
                    break
                m = make(*cand, seed)
                m.fit(X[tr][itr], yv[tr][itr])
                oi[ite] = m.predict_proba(X[tr][ite])[:, 1]
            if not ok or len(np.unique(yv[tr])) < 2:
                continue
            try:
                sc = roc_auc_score(yv[tr], oi)
            except Exception:
                continue
            if sc > bsc:
                bsc, best = (sc, cand)
        if picks is not None:
            picks.append(f'{best[0]}(C={best[1]})')
        m = make(*best, seed)
        m.fit(X[tr], yv[tr])
        o[te] = m.predict_proba(X[te])[:, 1]
    return o

def statistic(X, yv=yp, picks=None):
    return np.mean([nested_oof(X, s, yv, picks) for s in SEEDS], 0)

def boot(fn, yv, o, nb=2000, seed=RNG):
    rs = np.random.RandomState(seed)
    n = len(yv)
    v = []
    for _ in range(nb):
        idx = rs.choice(n, n, replace=True)
        if yv[idx].sum() in (0, len(idx)):
            continue
        try:
            v.append(fn(yv[idx], o[idx]))
        except Exception:
            pass
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
rows, OOF = ([], {})
for k, (X, _) in FS.items():
    picks = []
    o = statistic(X, picks=picks)
    OOF[k] = o
    a = roc_auc_score(yp, o)
    ap = average_precision_score(yp, o)
    alo, ahi = boot(roc_auc_score, yp, o)
    plo, phi = boot(average_precision_score, yp, o)
    ts = np.unique(np.round(o, 3))
    tf = float(ts[int(np.argmax([f1_score(yp, (o >= t).astype(int), zero_division=0) for t in ts]))])
    pr = (o >= tf).astype(int)
    rows.append([k, X.shape[1], f'{a:.3f} [{alo:.3f},{ahi:.3f}]', f'{ap:.3f} [{plo:.3f},{phi:.3f}]', round(precision_score(yp, pr, zero_division=0), 3), round(recall_score(yp, pr, zero_division=0), 3), round(f1_score(yp, pr, zero_division=0), 3), str(dict(Counter(picks)))])
t1 = pd.DataFrame(rows, columns=['Feature set', 'n features', 'AUROC (95% CI)', 'AUPRC (95% CI)', 'Precision', 'Recall', 'F1', 'Models chosen by inner CV'])
t1.to_csv(os.path.join(OUTR, 'TableBI_participant_sensor.csv'), index=False, encoding='utf-8-sig')
KEYS = ['P1  Clinical baseline only', 'P2  Clinical + sensor summary (compact)', 'P6  Sensor summary only (no clinical)']
SENSOR_KEYS = ['P2  Clinical + sensor summary (compact)']
rs = np.random.RandomState(RNG)
perms = [rs.permutation(NP) for _ in range(N_PERM)]
NULLS = {k: [] for k in KEYS}
for b, pm in enumerate(perms):
    ys = yp[pm]
    if ys.sum() in (0, NP):
        continue
    for k in KEYS:
        try:
            NULLS[k].append(roc_auc_score(ys, statistic(FS[k][0], yv=ys)))
        except Exception:
            NULLS[k].append(np.nan)
    if (b + 1) % 20 == 0:
        print(f'Completed {b + 1}/{N_PERM} matched permutations', flush=True)
rows2 = []
for k in KEYS:
    nl = np.array([v for v in NULLS[k] if not np.isnan(v)])
    if nl.size == 0:
        continue
    obs = roc_auc_score(yp, OOF[k])
    q = float(np.percentile(nl, 97.5))
    p = float((nl >= obs).mean())
    se = float(np.sqrt(max(p * (1 - p), 1e-12) / len(nl)))
    rows2.append([k, round(obs, 4), round(float(nl.mean()), 4), round(float(nl.std()), 4), round(q, 4), f'{p:.4f}', f'±{1.96 * se:.4f}', len(nl), 'exceeds the matched null' if obs > q else 'inside the matched null'])
t2 = pd.DataFrame(rows2, columns=['Feature set', 'Observed AUROC', 'Null mean', 'Null SD', 'Null 97.5th', 'Permutation p', 'p 95% MC interval', 'n permutations', 'Verdict'])
rows3 = []
for k in SENSOR_KEYS:
    a = np.array(NULLS['P1  Clinical baseline only'], float)
    b_ = np.array(NULLS[k], float)
    m = ~np.isnan(a) & ~np.isnan(b_)
    if not m.any():
        continue
    d_obs = roc_auc_score(yp, OOF[k]) - roc_auc_score(yp, OOF[KEYS[0]])
    d_null = b_[m] - a[m]
    p = float((d_null >= d_obs).mean())
    q = float(np.percentile(d_null, 97.5))
    rows3.append([f'{k} minus P1', round(d_obs, 4), round(float(d_null.mean()), 4), round(q, 4), f'{p:.4f}', int(m.sum()), 'exceeds null' if d_obs > q else 'inside null'])
t3 = pd.DataFrame(rows3, columns=['Contrast', 'Observed delta', 'Null mean delta', 'Null 97.5th', 'Permutation p', 'n permutations', 'Verdict'])
pd.concat([t2, pd.DataFrame([[''] * len(t2.columns)], columns=t2.columns)], ignore_index=True).to_csv(os.path.join(OUTR, 'TableBJ_participant_null.csv'), index=False, encoding='utf-8-sig')
t3.to_csv(os.path.join(OUTR, 'TableBJb_participant_delta.csv'), index=False, encoding='utf-8-sig')
BEST_S = max(SENSOR_KEYS, key=lambda k: roc_auc_score(yp, OOF[k]))
rows4 = []
for k in [KEYS[0], BEST_S]:
    o = OOF[k]
    ts = np.unique(np.round(o, 3))
    tf = float(ts[int(np.argmax([f1_score(yp, (o >= t).astype(int), zero_division=0) for t in ts]))])
    fpr, tpr, t2_ = roc_curve(yp, o)
    tj = float(t2_[np.argmax(tpr - fpr)])
    for th, lab in [(0.5, '0.500 (default)'), (tf, f'{tf:.3f} (F1-optimal)'), (tj, f'{tj:.3f} (Youden J)')]:
        pr = (o >= th).astype(int)
        tn, fp, fn_, tp = confusion_matrix(yp, pr, labels=[0, 1]).ravel()
        rows4.append([k, lab, tn, fp, fn_, tp, round(accuracy_score(yp, pr), 3), round(balanced_accuracy_score(yp, pr), 3), round(precision_score(yp, pr, zero_division=0), 3), round(recall_score(yp, pr, zero_division=0), 3), round(tn / max(tn + fp, 1), 3), round(f1_score(yp, pr, zero_division=0), 3), round(matthews_corrcoef(yp, pr), 3)])
t4 = pd.DataFrame(rows4, columns=['Feature set', 'Threshold', 'TN', 'FP', 'FN', 'TP', 'Accuracy', 'Balanced accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'MCC'])
t4.to_csv(os.path.join(OUTR, 'TableBK_participant_confusion.csv'), index=False, encoding='utf-8-sig')
pickle.dump(dict(method_variant="PASS_analysis", n_permutations=N_PERM, OOF=OOF, FS={k: v[1] for k, v in FS.items()}, yp=yp, PIDS=PIDS, NULLS={k: np.array(v, float) for k, v in NULLS.items()}, t1=t1, t2=t2, t3=t3, t4=t4, BEST_S=BEST_S, XS={k: v[0] for k, v in FS.items()}), open(str(STATE / 'stage37_state.pkl'), 'wb'))
