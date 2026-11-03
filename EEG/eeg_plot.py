"""
Three-panel figure of HMC EEG phase estimate:

(A) Uncalibrated phase error distribution (polar)
(B) Calibrated phase error distribution (polar)
(C) Mean phase error (uncal & cal) vs. IAF CV per recording,
    with paired points and regression lines.

Expected inputs
---------------
1) NPZ file containing arrays (degrees):
   - phase_err_unc_deg_all
   - phase_err_cal_deg_all

2) CSV with per-recording phase-error means (degrees):
   - required: file, mean_unc_deg, mean_cal_deg
   - optional: a weight column (e.g., n_windows, n_samples, ...)

3) CSV with per-segment IAF estimates used to compute CV per recording:
   - required: file, segment_index, had_alpha, paf_hz
"""

import numpy as np
import pandas as pd

from utils import (
    make_figure,
)


_NPZ_PATH = "phase_error_all.npz"
_IAF_CSV = "iaf_per_segment.csv"
SUBJ_CSV = "phase_error_per_subject.csv"


# IAF helpers
# Rodrigues2017
def compute_iaf_cv_per_file(iaf_csv):
    df = pd.read_csv(iaf_csv)

    mask_valid = (
        (df["segment_index"] >= 0)
        & (df["had_alpha"] == 1)
        & np.isfinite(df["paf_hz"])
    )
    df_valid = df.loc[mask_valid].copy()
    if df_valid.empty:
        raise RuntimeError("No valid segments with alpha found")

    stats = (
        df_valid.groupby("subject")["paf_hz"]          # explicit id, no regex
        .agg(mean_iaf_hz="mean", std_iaf_hz="std", n_segments="count")
        .reset_index()
    )
    stats["cv_iaf"] = stats["std_iaf_hz"] / stats["mean_iaf_hz"]
    return stats


def plot_phase_error(
    phase_err_unc_deg_all,
    phase_err_cal_deg_all,
    subj_csv_path=SUBJ_CSV,
    iaf_csv_path=_IAF_CSV,
    save_base="phase_error_rod17",
):
    phase_err_unc_rad = np.radians(np.asarray(phase_err_unc_deg_all))
    phase_err_cal_rad = np.radians(np.asarray(phase_err_cal_deg_all))

    # Subject-level stats: EXACT pooled circular means (Fix B)
    subj_df = pd.read_csv(subj_csv_path)
    iaf_stats = compute_iaf_cv_per_file(iaf_csv_path)

    merged = subj_df.merge(iaf_stats, on="subject", how="left")
    if merged.empty:
        raise RuntimeError("No overlapping subjects after merging.")

    # Everything below comes from the SAME frame -> always aligned
    x_cv = merged["cv_iaf"].to_numpy(float)
    y_unc = merged["mean_unc_deg"].to_numpy(float)
    y_cal = merged["mean_cal_deg"].to_numpy(float)
    trial_mu_unc = np.radians(y_unc)
    trial_mu_cal = np.radians(y_cal)
    trial_nwin = merged["n_samples"].to_numpy(float)

    make_figure(
        err_unc_rad=phase_err_unc_rad,
        err_cal_rad=phase_err_cal_rad,
        trial_freq_cv=x_cv,
        trial_abs_unc_rad=np.radians(y_unc),
        trial_abs_cal_rad=np.radians(y_cal),
        trial_mu_unc=trial_mu_unc,
        trial_mu_cal=trial_mu_cal,
        trial_nwin=trial_nwin,
        save_base=save_base,
        n_perm=int(1e5),
        perm_seed=0,
        panel_c_xlabel="IAF CV",
        panel_c_title=r"$\mathbf{c}$ Phase error vs. IAF variability",
    )

def main(_NPZ_PATH=_NPZ_PATH, subj_csv_path=SUBJ_CSV, iaf_csv_path=_IAF_CSV):
    data = np.load(_NPZ_PATH)
    if "phase_err_unc_deg_all" not in data.files or "phase_err_cal_deg_all" not in data.files:
        raise KeyError(
            "NPZ file must contain 'phase_err_unc_deg_all' and 'phase_err_cal_deg_all'."
        )

    plot_phase_error(
        data["phase_err_unc_deg_all"],
        data["phase_err_cal_deg_all"],
        subj_csv_path=subj_csv_path,
        iaf_csv_path=iaf_csv_path,
    )


if __name__ == "__main__":
    main()
