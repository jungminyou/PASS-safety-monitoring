"""Generate a synthetic dataset with the same format as the restricted study inputs.

All values are random draws. They are not derived from, fitted to, or resampled
from any participant data, and analysis results obtained with them are meaningless.
The output satisfies the structural checks in the analysis code: 67 dyads,
14 days each (938 rows), and 24 dyads with at least one incident day.

Usage: python make_synthetic_data.py [output_dir]   (default: this directory)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
rs = np.random.RandomState(20260101)
N_DYADS, N_DAYS, N_INCIDENT = 67, 14, 24
LOCS = np.array(['bed', 'living', 'bath', 'door', 'kitchen'])
pids = np.arange(1001, 1001 + N_DYADS)
incident = np.zeros(N_DYADS, int)
incident[rs.choice(N_DYADS, N_INCIDENT, replace=False)] = 1


def sleep_day():
    tst = rs.normal(560, 80)
    tst_n = rs.normal(450, 70)
    return {
        'TST': tst, 'WASO': rs.gamma(2, 12), 'Efficiency': rs.uniform(88, 99),
        'NoA': rs.poisson(4), 'MAL': rs.gamma(2, 3.5),
        'TST_night': tst_n, 'WASO_night': rs.gamma(2, 8), 'Efficiency_night': rs.uniform(88, 99),
        'NoA_night': rs.poisson(3), 'MAL_night': rs.gamma(2, 3.5), 'Onset Latency': rs.gamma(2, 7),
    }


rows = []
for i, p in enumerate(pids):
    med = rs.randint(1, 18)
    mmse = -1 if rs.rand() < 0.06 else rs.randint(3, 25)
    base = {
        'Age': rs.randint(66, 95), 'C_Age': rs.randint(40, 90),
        'CIRS_G_totalscore': rs.randint(0, 20), 'CIRS_G_S_index': round(rs.uniform(0.5, 3), 2),
        'Med_total': med, 'CSDD_total': rs.randint(0, 25), 'NPI_totalscore': rs.randint(0, 100),
        'ADL_total': rs.randint(7, 22), 'IADL_total': rs.randint(10, 34), 'MMSE_total': mmse,
        'P_Gender': rs.randint(1, 3), 'C_Gender': rs.randint(1, 3), 'P_Marriage': rs.choice([2, 4]),
        'P_Edu': rs.randint(1, 5), 'C_Edu': rs.randint(1, 5), 'P_Job_re': rs.randint(0, 2),
        'C_Job_re': rs.randint(0, 2), 'Help': rs.randint(0, 2), 'Relationship': rs.choice([1, 2, 3, 5]),
        'Income': rs.randint(1, 8), 'Polypharmacy': int(med >= 5),
        'MMSE_group': 2 if mmse < 10 else rs.randint(0, 2),
    }
    events = np.zeros(N_DAYS, int)
    if incident[i]:
        events[rs.choice(N_DAYS, rs.randint(1, 4), replace=False)] = 1
    for d in range(N_DAYS):
        row = {'P_ID': p, 'Days': d + 1, 'Event': events[d], **base}
        for k, v in sleep_day().items():
            row[k] = np.nan if rs.rand() < 0.15 else round(float(v), 2)
        rows.append(row)
pd.DataFrame(rows).to_csv(OUT / 'safety_dataset_ircount_20230728.csv', index=False, encoding='utf-8')

installed = np.zeros((N_DYADS, 5), int)
for i in range(N_DYADS):
    installed[i, rs.choice(5, rs.randint(1, 4), replace=False)] = 1
absent = np.broadcast_to((installed == 0)[:, None, :, None], (N_DYADS, N_DAYS, 5, 24))
cube = rs.poisson(1.5, (N_DYADS, N_DAYS, 5, 24)).astype(float)
cube[absent] = np.nan
aw_minutes = rs.randint(200, 900, (N_DYADS, N_DAYS)).astype(float)
np.savez_compressed(
    OUT / 'sensor_cube.npz', pids=pids, locs=LOCS, cube=cube, installed=installed,
    aw_minutes=aw_minutes, aw_worn=np.ones((N_DYADS, N_DAYS)), obs_day=np.ones((N_DYADS, N_DAYS)),
)

valid_min = rs.randint(0, 61, (N_DYADS, N_DAYS, 24)).astype(float)
with np.errstate(invalid='ignore', divide='ignore'):
    ir_rate = np.where(valid_min[:, :, None, :] > 0, cube / valid_min[:, :, None, :], np.nan)
ir_rate[absent] = np.nan
np.savez_compressed(
    OUT / 'valid_cube.npz', pids=pids, locs=LOCS, ir_rate=ir_rate,
    valid_min=valid_min, installed=installed,
)
print(f'Synthetic dataset written to {OUT}')
