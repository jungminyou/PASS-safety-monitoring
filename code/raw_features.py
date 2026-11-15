"""Build the retained feature blocks from an approved 67-dyad dataset and private sensor cube."""
import os, warnings
from config import PRIVATE_DATA
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
DATA_DIR = str(PRIVATE_DATA)
IRCOUNT = os.path.join(DATA_DIR, 'safety_dataset_ircount_20230728.csv')
ic = pd.read_csv(IRCOUNT, encoding='utf-8')
assert ic.P_ID.nunique() == 67, 'Expected the approved 67-dyad analysis cohort.'
assert len(ic) == 938 and (not ic.duplicated(['P_ID', 'Days']).any()), 'Expected 14 unique days per dyad.'
assert ic.groupby('P_ID').Days.apply(lambda x: set(x) == set(range(1, 15))).all()
assert ic.Event.isin([0, 1]).all()
assert ic.groupby('P_ID').Event.max().sum() == 24
ic = ic.sort_values(['P_ID', 'Days']).reset_index(drop=True)
N = len(ic)
pid = ic.P_ID.values
day = ic.Days.values
PIDS = np.array(sorted(ic.P_ID.unique()))
ROW = np.array([[np.where((pid == p) & (day == d))[0][0] for d in range(1, 15)] for p in PIDS])
BASE = ['Age', 'CIRS_G_S_index', 'Med_total', 'CSDD_total', 'NPI_totalscore', 'ADL_total', 'IADL_total', 'MMSE_total', 'C_Age']
SLEEP = ['TST', 'WASO', 'Efficiency', 'NoA', 'MAL', 'TST_night', 'WASO_night', 'Efficiency_night', 'NoA_night', 'MAL_night', 'Onset Latency']
CLIN = ic[BASE].values.astype(float)
# Daily sleep observations are used without forward filling.
sleep_raw = ic[SLEEP].values.astype(float)
y = ic['Event'].astype(int).values
_z = np.load(os.path.join(DATA_DIR, 'sensor_cube.npz'), allow_pickle=True)
_cube_pids = list(_z['pids'])
_keep = np.array([_cube_pids.index(p) for p in PIDS])
CUBE = _z['cube'][_keep]
INSTALLED = _z['installed'][_keep]
AW_MIN = _z['aw_minutes'][_keep]
OBS_DAY = _z['obs_day'][_keep]
assert CUBE.shape[0] == len(PIDS)
NIGHT = list(range(20, 24)) + list(range(0, 8))
DAYH = list(range(8, 20))

def _flat(M3):
    out = np.empty((N,) + M3.shape[2:], dtype=float)
    for i in range(len(PIDS)):
        out[ROW[i]] = M3[i]
    return out

def build_invariant():
    NP, T = (CUBE.shape[0], CUBE.shape[1])
    F_, names = ([], [])
    tot = np.nansum(CUBE, axis=(2, 3))
    nsens = INSTALLED.sum(1)
    night = np.nansum(CUBE[:, :, :, NIGHT], axis=(2, 3))
    dayc = np.nansum(CUBE[:, :, :, DAYH], axis=(2, 3))
    hourly = np.nansum(CUBE, axis=2)
    with np.errstate(invalid='ignore', divide='ignore'):
        per_sensor = tot / nsens[:, None]
        night_frac = np.where(tot > 0, night / np.maximum(tot, 1e-09), np.nan)
        per_awmin = np.where(AW_MIN > 0, tot / np.maximum(AW_MIN, 1e-09), np.nan)
    act_hours = (hourly > 0).sum(2).astype(float)
    act_night = (hourly[:, :, NIGHT] > 0).sum(2).astype(float)
    p = hourly / np.maximum(hourly.sum(2, keepdims=True), 1e-09)
    ent = -np.where(p > 0, p * np.log(p + 1e-12), 0).sum(2)
    ent = np.where(hourly.sum(2) > 0, ent, np.nan)
    peak_h = np.argmax(hourly, axis=2).astype(float)
    peak_frac = np.where(hourly.sum(2) > 0, hourly.max(2) / np.maximum(hourly.sum(2), 1e-09), np.nan)
    nb = np.zeros((NP, T))
    for i in range(NP):
        for t in range(T):
            v = (hourly[i, t, NIGHT] > 0).astype(int)
            nb[i, t] = int(np.sum((v[1:] == 1) & (v[:-1] == 0)) + (v[0] == 1))
    for arr, nm in [(tot, 'ir_total'), (per_sensor, 'ir_per_sensor'), (night, 'ir_night'), (dayc, 'ir_day'), (night_frac, 'ir_night_frac'), (per_awmin, 'ir_per_awmin'), (act_hours, 'ir_active_hours'), (act_night, 'ir_active_night_hours'), (ent, 'ir_hour_entropy'), (peak_h, 'ir_peak_hour'), (peak_frac, 'ir_peak_frac'), (nb, 'ir_night_bouts'), (AW_MIN, 'aw_move_minutes'), (OBS_DAY, 'sensor_log_present')]:
        F_.append(_flat(arr)[:, None])
        names.append(nm)
    return (np.hstack(F_), names)
INV, INV_COLS = build_invariant()
