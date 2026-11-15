# Restricted input contracts

All inputs described here remain private. No example participant rows are supplied.

## Daily CSV

The CSV fields and their types are listed in `DATA_DICTIONARY.csv`. Use a unique `(P_ID, Days)` key and all days 1–14 per dyad. Baseline values repeat within each dyad. Missing numeric values are represented as empty CSV cells. `Event` is the daily composite diary outcome; the participant outcome is its maximum over 14 days.

Table 1 reports negative `MMSE_total` sentinel values as unable to complete and excludes them from the completer mean. The prediction models retain this sentinel as numeric; they do not replace it with missingness.

## sensor_cube.npz

| Key | Shape | Type and definition |
| --- | --- | --- |
| pids | (n,) | Integer keys matching the CSV dyads; extra preselection dyads may exist and are not analyzed |
| locs | (5,) | Room labels: bed, living, bath, door, kitchen; actual array order follows these labels |
| cube | (n,14,5,24) | Numeric infrared counts after same-minute actigraphy movement filtering; structurally absent sensors are NaN |
| installed | (n,5) | Binary sensor installation indicators |
| aw_minutes | (n,14) | Daily wrist movement minutes used as the denominator of location-invariant infrared rates |
| aw_worn | (n,14) | Binary indicator of available valid actigraphy data that day; may be present but is not read by this code |
| obs_day | (n,14) | Binary sensor log presence indicator |

The final sensor axis follows `locs`; hours are 0–23 and days are 1–14. Night comprises hours 20–23 followed by 0–7; daytime comprises 8–19.

## valid_cube.npz

Use exactly the same approved 67 dyads as the CSV, in ascending dyad order.

| Key | Shape | Type and definition |
| --- | --- | --- |
| pids | (67,) | Integer dyad keys in the same order as the CSV's dyad blocks |
| locs | (5,) | Room labels as above |
| ir_rate | (67,14,5,24) | Retained infrared counts per valid actigraphy minute; unobserved hours and absent sensors are NaN |
| valid_min | (67,14,24) | Worn actigraphy minutes with Activity > 0 in each hour, used as exposure for room-specific rates |
| installed | (67,5) | Binary sensor installation indicators |

The preprocessing of device exports is upstream of this release. These contracts describe the retained intermediate inputs and do not provide a raw-device ingestion pipeline.

## Participant feature expansions

The nine clinical variables are `Age`, `CIRS_G_S_index`, `Med_total`, `CSDD_total`, `NPI_totalscore`, `ADL_total`, `IADL_total`, `MMSE_total`, and `C_Age`.

Every one of the 14 daily sensor features in the dictionary produces `__mean14`, `__sd14`, and `__slope14`: available-day mean, population SD (`ddof=0`), and ordinary least-squares slope against zero-based day index, requiring at least three observed days for a slope. This yields 42 sensor summaries. Day positions are retained when values are missing.

The compact nine summaries are `ir_per_awmin__mean14`, `ir_night_frac__mean14`, `ir_hour_entropy__mean14`, `ir_active_night_hours__mean14`, `ir_night_bouts__mean14`, `ir_per_awmin__sd14`, `ir_night_frac__sd14`, `ir_per_awmin__slope14`, and `aw_move_minutes__mean14`.

The 11 sleep variables are not forward-filled. Each is summarized by its available-day mean and population SD (`ddof=0`) within dyad, producing 22 sleep summaries. A summary remains missing when a dyad has no observed day for that variable. Training-fold median imputation and min-max scaling occur downstream inside cross-validation.

Table 2 uses the unfilled daily sleep observations. Each room's daytime and nighttime rate is an exposure-weighted aggregate of hourly rates over its observed hours. An absent sensor or zero observed exposure yields missingness, never a structural zero.
