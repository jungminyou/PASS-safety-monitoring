"""Table 1: baseline characteristics. The private input must already contain only the approved analysis cohort."""
import os, warnings
from math import comb
from config import PRIVATE_DATA, RESULTS
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats


def fisher_2xk(table):
    """Exact conditional, probability-ordered two-sided Fisher test for 2-by-K tables."""
    table = np.asarray(table)
    if table.ndim != 2 or table.shape[0] != 2:
        raise ValueError('Expected a 2-by-K contingency table.')
    if not np.isfinite(table).all() or (table < 0).any() or (table != np.floor(table)).any():
        raise ValueError('Counts must be finite nonnegative integers.')
    table = table.astype(int)
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] <= 1 or (table.sum(axis=1) == 0).any():
        return 1.0
    if table[0].sum() > table[1].sum():
        table = table[::-1]
    margins = [int(x) for x in table.sum(axis=0)]
    target = int(table[0].sum())
    observed = 1
    for total, count in zip(margins, table[0]):
        observed *= comb(total, int(count))
    remaining_capacity = [sum(margins[i:]) for i in range(len(margins) + 1)]
    tail_weight = 0

    def enumerate_tables(j, left, weight):
        nonlocal tail_weight
        if j == len(margins) - 1:
            if 0 <= left <= margins[j]:
                final_weight = weight * comb(margins[j], left)
                if final_weight <= observed:
                    tail_weight += final_weight
            return
        low = max(0, left - remaining_capacity[j + 1])
        high = min(left, margins[j])
        for count in range(low, high + 1):
            enumerate_tables(j + 1, left - count, weight * comb(margins[j], count))

    enumerate_tables(0, target, 1)
    return min(1.0, tail_weight / comb(sum(margins), target))


DATA = str(PRIVATE_DATA)
RES = str(RESULTS)
ic = pd.read_csv(os.path.join(DATA, 'safety_dataset_ircount_20230728.csv'), encoding='utf-8')
assert ic.P_ID.nunique() == 67, 'Expected the approved 67-dyad analysis cohort.'
assert len(ic) == 938 and (not ic.duplicated(['P_ID', 'Days']).any()), 'Expected 14 unique days per dyad.'
assert ic.groupby('P_ID').Days.apply(lambda x: set(x) == set(range(1, 15))).all()
assert ic.Event.isin([0, 1]).all()
assert ic.groupby('P_ID').Event.max().sum() == 24
d = ic.groupby('P_ID').first().reset_index()
d['inc'] = ic.groupby('P_ID').Event.max().values
A, B = (d[d.inc == 1], d[d.inc == 0])
nA, nB = (len(A), len(B))
LBL = {'P_Gender': {1: 'Male', 2: 'Female'}, 'P_Marriage': {1: 'Never married', 2: 'Married', 3: 'Separated', 4: 'Widowed', 5: 'Divorced'}, 'P_Edu': {1: 'Elementary or less', 2: 'Middle school', 3: 'High school', 4: 'College or above'}, 'MMSE_group': {0: 'Normal / questionable', 1: 'Mild to moderate', 2: 'Severe'}, 'Polypharmacy': {0: 'No', 1: 'Yes'}, 'C_Gender': {1: 'Male', 2: 'Female'}, 'Relationship': {1: 'Spouse', 2: 'Son', 3: 'Daughter', 4: 'Son-in-law', 5: 'Daughter-in-law'}, 'C_Edu': {1: 'Elementary or less', 2: 'Middle school', 3: 'High school', 4: 'College or above'}, 'Income': {1: '< 0.5', 2: '0.5–1.0', 3: '1.0–2.0', 4: '2.0–3.0', 5: '3.0–4.0', 6: '4.0–5.0', 7: '≥ 5.0'}, 'Help': {0: 'No', 1: 'Yes'}, 'C_Job_re': {0: 'Not employed', 1: 'Employed'}, 'P_Job_re': {0: 'Not employed', 1: 'Employed'}}
rows = []

def add(var, cat, a, b, p):
    rows.append([var, cat, a, b, p])

def cont(name, col, valid=None):
    a, b = (A[col].astype(float), B[col].astype(float))
    if valid is not None:
        a, b = (a[valid(a)], b[valid(b)])
    p = stats.ttest_ind(a, b, equal_var=False).pvalue
    add(name, '', f'{a.mean():.2f} ± {a.std(ddof=1):.2f}', f'{b.mean():.2f} ± {b.std(ddof=1):.2f}', f'{p:.3f}')

def cat(name, col):
    lv = sorted(set(d[col].dropna().astype(int)))
    tab = np.array([[int((A[col] == k).sum()) for k in lv], [int((B[col] == k).sum()) for k in lv]])
    keep = tab.sum(0) > 0
    tab, lv = (tab[:, keep], [k for k, m in zip(lv, keep) if m])
    try:
        _, p, _, exp = stats.chi2_contingency(tab)
        if (exp < 5).any():
            if tab.shape[1] == 2:
                p = stats.fisher_exact(tab)[1]
            else:
                p = fisher_2xk(tab)
    except Exception:
        p = np.nan
    for i, k in enumerate(lv):
        lab = LBL.get(col, {}).get(k, str(k))
        add(name if i == 0 else '', lab, f'{tab[0, i]} ({100 * tab[0, i] / nA:.1f})', f'{tab[1, i]} ({100 * tab[1, i] / nB:.1f})', f'{p:.3f}' if i == 0 else '')
add('Person living with dementia', '', '', '', '')
cont('Age, years', 'Age')
cat('Sex', 'P_Gender')
cat('Education', 'P_Edu')
cat('Marital status', 'P_Marriage')
cat('Employment', 'P_Job_re')
cont('CIRS-G total score', 'CIRS_G_totalscore')
cont('CIRS-G severity index', 'CIRS_G_S_index')
cont('Number of medications', 'Med_total')
cat('Polypharmacy (≥ 5 medications)', 'Polypharmacy')
cont('K-CSDD (depression)', 'CSDD_total')
cont('K-NPI (BPSD)', 'NPI_totalscore')
cont('K-ADL', 'ADL_total')
cont('K-IADL', 'IADL_total')
cont('K-MMSE, completers only', 'MMSE_total', valid=lambda s: s >= 0)
na, nb = (int((A.MMSE_total < 0).sum()), int((B.MMSE_total < 0).sum()))
tab = np.array([[na, nA - na], [nb, nB - nb]])
p_un = stats.fisher_exact(tab)[1]
add('Unable to complete K-MMSE', '', f'{na} ({100 * na / nA:.1f})', f'{nb} ({100 * nb / nB:.1f})', f'{p_un:.3f}')
cat('Cognitive impairment stage', 'MMSE_group')
add('Family caregiver', '', '', '', '')
cont('Age, years', 'C_Age')
cat('Sex', 'C_Gender')
cat('Relationship to the person with dementia', 'Relationship')
cat('Education', 'C_Edu')
cat('Employment', 'C_Job_re')
cat('Monthly household income, million KRW', 'Income')
cat('Receives formal care support', 'Help')
t1 = pd.DataFrame(rows, columns=['Characteristic', 'Category', f'Incident (n = {nA})', f'No incident (n = {nB})', 'P'])
t1.to_csv(os.path.join(RES, 'Table1_participant_characteristics.csv'), index=False, encoding='utf-8-sig')
