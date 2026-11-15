"""Figures 1 to 3: ROC and precision-recall curves, descriptive confusion matrices, and out-of-fold SHAP."""
import os, pickle, warnings
from config import STATE, FIGURES
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, confusion_matrix
MM = 1 / 25.4
W2 = 183 * MM
HMAX = 247 * MM
rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'xtick.major.size': 2.6, 'ytick.major.size': 2.6})
C1, C2 = ('#2166ac', '#c0392b')
DEST = str(FIGURES)
os.makedirs(DEST, exist_ok=True)
S37 = pickle.load(open(str(STATE / 'stage37_state.pkl'), 'rb'))
if S37.get('method_variant') != 'PASS_analysis':
    raise ValueError('Run participant_models.py in this work directory before figures.py.')
yp, OOF, XS, FSN, t1 = (S37['yp'], S37['OOF'], S37['XS'], S37['FS'], S37['t1'])
K1 = 'P1  Clinical baseline only'
K2 = 'P2  Clinical + sensor summary (compact)'
L1, L2 = ('Clinical baseline (9)', 'Clinical + sensor summary (18)')
LAB = {'Age': 'Age (person with dementia)', 'C_Age': 'Age (caregiver)', 'CIRS_G_S_index': 'CIRS-G severity index', 'Med_total': 'Number of medications', 'CSDD_total': 'K-CSDD (depression)', 'NPI_totalscore': 'K-NPI (BPSD)', 'ADL_total': 'K-ADL', 'IADL_total': 'K-IADL', 'MMSE_total': 'K-MMSE', 'ir_per_awmin__mean14': 'Infrared rate per active min, mean', 'ir_per_awmin__sd14': 'Infrared rate per active min, SD', 'ir_per_awmin__slope14': 'Infrared rate per active min, trend', 'ir_night_frac__mean14': 'Nocturnal share of infrared, mean', 'ir_night_frac__sd14': 'Nocturnal share of infrared, SD', 'ir_hour_entropy__mean14': 'Entropy of hourly activity, mean', 'ir_active_night_hours__mean14': 'Active night-time hours, mean', 'ir_night_bouts__mean14': 'Nocturnal activity bouts, mean', 'aw_move_minutes__mean14': 'Actigraphy movement minutes, mean'}

def lab(c):
    return LAB.get(c, c)

def save(fig, n):
    w, h = fig.get_size_inches()
    assert w <= W2 + 1e-06
    assert h <= HMAX + 1e-06
    fig.savefig(os.path.join(DEST, f'{n}.png'), dpi=600, bbox_inches=None)
    fig.savefig(os.path.join(DEST, f'{n}.pdf'), bbox_inches=None)
    plt.close(fig)

def panel(ax, s, dx=-0.13, dy=1.03):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')
import shap
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
_CAND = [('L2', c) for c in (0.01, 0.03, 0.1, 0.3, 1.0)] + [('L1', c) for c in (0.03, 0.1, 0.3)]

def _mk(kind, c, seed):
    return Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', MinMaxScaler()), ('c', LogisticRegression(penalty='l1' if kind == 'L1' else 'l2', C=c, solver='liblinear', class_weight='balanced', max_iter=5000, random_state=seed))])

def oof_shap(X, seeds=(42, 7)):
    sh = np.zeros(X.shape)
    for seed in seeds:
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, yp, groups=S37['PIDS']):
            best, bsc = (_CAND[0], -np.inf)
            for cd in _CAND:
                oi = np.full(len(tr), np.nan)
                ok = True
                for itr, ite in StratifiedGroupKFold(4, shuffle=True, random_state=seed + 1).split(X[tr], yp[tr], groups=S37['PIDS'][tr]):
                    if yp[tr][itr].sum() < 2:
                        ok = False
                        break
                    m = _mk(*cd, seed)
                    m.fit(X[tr][itr], yp[tr][itr])
                    oi[ite] = m.predict_proba(X[tr][ite])[:, 1]
                if not ok:
                    continue
                try:
                    sc = roc_auc_score(yp[tr], oi)
                except Exception:
                    continue
                if sc > bsc:
                    bsc, best = (sc, cd)
            m = _mk(*best, seed)
            m.fit(X[tr], yp[tr])
            Xi = m[:-1].transform(X)
            sh[te] += shap.LinearExplainer(m[-1], Xi[tr]).shap_values(Xi[te])
    return sh / len(seeds)

def ci_of(k, col):
    r = t1[t1['Feature set'] == k].iloc[0]
    return r[col].replace('[', '(').replace(']', ')').replace(',', '–')
fig, axes = plt.subplots(1, 2, figsize=(W2, 3.05))
ax = axes[0]
for k, c, lb in [(K1, C1, L1), (K2, C2, L2)]:
    f, t, _ = roc_curve(yp, OOF[k])
    ax.plot(f, t, color=c, lw=1.5, label=f"{lb}\nAUROC {ci_of(k, 'AUROC (95% CI)')}")
ax.plot([0, 1], [0, 1], ls=(0, (3, 2.5)), color='#aaa', lw=0.8)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_xlabel('1 − Specificity')
ax.set_ylabel('Sensitivity')
ax.legend(loc='lower right', frameon=False, labelspacing=0.7, handlelength=1.4, borderpad=0.1, handletextpad=0.5)
panel(ax, 'a', dx=-0.17)
ax = axes[1]
for k, c, lb in [(K1, C1, L1), (K2, C2, L2)]:
    pr, rc, _ = precision_recall_curve(yp, OOF[k])
    ax.step(rc, pr, where='post', color=c, lw=1.5, label=f"{lb}\nAUPRC {ci_of(k, 'AUPRC (95% CI)')}")
ax.axhline(yp.mean(), ls=(0, (3, 2.5)), color='#aaa', lw=0.8)
ax.text(0.02, yp.mean() + 0.025, f'No-skill baseline {yp.mean():.3f}', fontsize=6.5, color='#666')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend(loc='lower left', frameon=False, labelspacing=0.7, handlelength=1.4, borderpad=0.1, handletextpad=0.5)
panel(ax, 'b', dx=-0.17)
fig.tight_layout(pad=0.6, w_pad=2.2, rect=[0.015, 0, 1, 0.955])
save(fig, 'Fig1_participant_ROC_PR')
CM = LinearSegmentedColormap.from_list('bl', ['#ffffff', C1])
fig, axes = plt.subplots(2, 2, figsize=(W2 * 0.92, 5.2))
PL = iter('abcd')
for row, (k, mlab) in enumerate([(K1, 'Clinical baseline (9)'), (K2, 'Clinical + sensor summary (18)')]):
    o = OOF[k]
    fpr, tpr, th_ = roc_curve(yp, o)
    tj = float(th_[np.argmax(tpr - fpr)])
    for col, (thr, tname) in enumerate([(0.5, 'default'), (tj, 'Youden')]):
        ax = axes[row, col]
        cm = confusion_matrix(yp, (o >= thr).astype(int), labels=[0, 1])
        fr = cm / cm.sum(1, keepdims=True)
        ax.imshow(fr, cmap=CM, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                r_, g_, b_, _ = CM(fr[i, j])
                lum = 0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_
                tc = 'white' if lum < 0.45 else 'black'
                ax.text(j, i - 0.08, f'{cm[i, j]}', ha='center', va='center', fontsize=12, color=tc, fontweight='bold')
                ax.text(j, i + 0.24, f'{fr[i, j] * 100:.0f}%', ha='center', va='center', fontsize=6.5, color=tc)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Predicted\nno incident', 'Predicted\nincident'], fontsize=6.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Observed\nno incident', 'Observed\nincident'], fontsize=6.5)
        ax.tick_params(length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        tn, fp, fn_, tp = cm.ravel()
        ax.set_title(f'{mlab}\n{tname} threshold {thr:.3f}', fontsize=7.5, pad=4)
        ax.set_xlabel(f'Sensitivity {tp / max(tp + fn_, 1):.3f} · Specificity {tn / max(tn + fp, 1):.3f}\nPPV {tp / max(tp + fp, 1):.3f} · NPV {tn / max(tn + fn_, 1):.3f}', fontsize=6.5, labelpad=4)
        panel(ax, next(PL), dx=-0.34, dy=1.2)
fig.subplots_adjust(left=0.135, right=0.99, top=0.895, bottom=0.125, wspace=0.62, hspace=0.85)
save(fig, 'Fig2_participant_confusion')
SHAPC = shap.plots.colors.red_blue
SH = {K1: oof_shap(XS[K1]), K2: oof_shap(XS[K2])}

def beeswarm(ax, sh, Xv, names, nmax):
    order = np.argsort(np.abs(sh).mean(0))[::-1][:nmax]
    rs = np.random.RandomState(0)
    for k, j in enumerate(order[::-1]):
        v, xv = (sh[:, j], Xv[:, j])
        ok = ~np.isnan(xv)
        q = np.full(len(xv), np.nan)
        if ok.sum() > 1:
            lo, hi = np.nanpercentile(xv[ok], [5, 95])
            q[ok] = np.clip((xv[ok] - lo) / max(hi - lo, 1e-09), 0, 1)
        ax.axhline(k, color='#dddddd', lw=0.5, ls=(0, (1, 2)), zorder=0)
        off = np.zeros(len(v))
        bins = np.linspace(v.min() - 1e-09, v.max() + 1e-09, 26)
        bi = np.clip(np.digitize(v, bins) - 1, 0, len(bins) - 2)
        for bb in np.unique(bi):
            sel = np.where(bi == bb)[0]
            n_ = len(sel)
            if n_ == 1:
                continue
            sp = min(0.3, 0.05 * np.sqrt(n_))
            off[sel] = np.linspace(-sp, sp, n_)[rs.permutation(n_)]
        ax.scatter(v[ok], k + off[ok], c=SHAPC(q[ok]), s=8, alpha=0.9, linewidths=0, zorder=3)
    ax.axvline(0, color='#999999', lw=0.8, zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([lab(names[j]) for j in order[::-1]], fontsize=6.8)
    ax.set_ylim(-0.8, len(order) - 0.2)
    for sp_ in ('left', 'top', 'right'):
        ax.spines[sp_].set_visible(False)
    ax.tick_params(axis='y', length=0)
fig, axes = plt.subplots(1, 2, figsize=(W2, 3.6))
XLIM = (-1.65, 2.05)
for ax, k, pl, ttl in [(axes[0], K1, 'a', 'Clinical baseline (9)'), (axes[1], K2, 'b', 'Clinical + sensor summary (18)')]:
    beeswarm(ax, SH[k], XS[k], FSN[k], 18)
    ax.set_xlim(*XLIM)
    ax.set_xlabel('SHAP value (impact on model output)')
    ax.set_title(ttl, fontsize=8, pad=4)
    panel(ax, pl, dx=-0.46, dy=1.03)
sm = plt.cm.ScalarMappable(cmap=SHAPC)
sm.set_array([])
cb = fig.colorbar(sm, ax=axes, fraction=0.016, pad=0.012, aspect=24)
cb.set_ticks([0, 1])
cb.set_ticklabels(['Low', 'High'])
cb.ax.tick_params(labelsize=6.5, length=0)
cb.set_label('Feature value', fontsize=7.5)
cb.outline.set_visible(False)
fig.subplots_adjust(left=0.215, right=0.925, top=0.9, bottom=0.155, wspace=0.95)
save(fig, 'Fig3_SHAP_beeswarm')
