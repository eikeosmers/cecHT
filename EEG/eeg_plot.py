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


NPZ_PATH = "phase_error_all.npz"
PHASE_CSV = "phase_error_per_file.csv"
IAF_CSV = "iaf_per_segment.csv"



# IAF helpers
# Rodrigues2017
def compute_iaf_cv_per_file(iaf_csv: str):
    df = pd.read_csv(iaf_csv)

    mask_valid = (
        (df["segment_index"] >= 0)
        & (df["had_alpha"] == 1)
        & np.isfinite(df["paf_hz"])
    )

    df_valid = df.loc[mask_valid].copy()

    if df_valid.empty:
        raise RuntimeError("No valid segments with alpha found")

    # Extract participant ID
    df_valid["participant"] = df_valid["file"].str.extract(r"^s?([a-zA-Z0-9]+)_", expand=False)

    stats = (
        df_valid
        .groupby("participant")["paf_hz"]
        .agg(
            mean_iaf_hz="mean",
            std_iaf_hz="std",
            n_segments="count",
        )
        .reset_index()
        .rename(columns={"participant": "file"})
    )

    stats["cv_iaf"] = stats["std_iaf_hz"] / stats["mean_iaf_hz"]

    return stats


def plot_phase_error(
    phase_err_unc_deg_all,
    phase_err_cal_deg_all,
    phase_csv_path=PHASE_CSV,
    iaf_csv_path=IAF_CSV,
    save_base="phase_error_rod17",
):
    # Data for panels A & B (Raw samples)
    phase_err_unc_rad = np.radians(np.asarray(phase_err_unc_deg_all))
    phase_err_cal_rad = np.radians(np.asarray(phase_err_cal_deg_all))

    # Data for Panel C AND Significance Test (Aggregated to Subject)
    phase_df = pd.read_csv(phase_csv_path)
    iaf_stats = compute_iaf_cv_per_file(iaf_csv_path)

    # Extract Participant ID from segment filenames
    phase_df["participant"] = phase_df["file"].str.extract(r"^s?([a-zA-Z0-9]+)_", expand=False)

    phase_subject = phase_df.groupby("participant").agg({
        "mean_unc_deg": "mean",
        "mean_cal_deg": "mean",
        "n_samples": "sum" 
    }).reset_index()

    # Prepare variables for the Significance Test
    trial_mu_unc = np.radians(phase_subject["mean_unc_deg"].to_numpy(dtype=float))
    trial_mu_cal = np.radians(phase_subject["mean_cal_deg"].to_numpy(dtype=float))
    trial_nwin = phase_subject["n_samples"].to_numpy(dtype=float)

    # merge the subject-level phase data with the subject-level IAF stats
    merged = phase_subject.merge(iaf_stats, left_on="participant", right_on="file", how="left")

    if merged.empty:
        raise RuntimeError("No overlapping recordings after aggregating to subject level.")

    x_cv = merged["cv_iaf"].values
    y_unc = merged["mean_unc_deg"].values
    y_cal = merged["mean_cal_deg"].values

    make_figure(
        err_unc_rad=phase_err_unc_rad,      # All samples (for histograms)
        err_cal_rad=phase_err_cal_rad,      # All samples (for histograms)
        trial_freq_cv=x_cv,                 # Per-subject (for Panel C)
        trial_abs_unc_rad=np.radians(y_unc), # Per-subject (for Panel C)
        trial_abs_cal_rad=np.radians(y_cal), # Per-subject (for Panel C)
        trial_mu_unc=trial_mu_unc,          # Per-subject (for Significance Test)
        trial_mu_cal=trial_mu_cal,          # Per-subject (for Significance Test)
        trial_nwin=trial_nwin,              # Per-subject (for Significance Test)
        save_base=save_base,
        n_perm=int(1e5),
        perm_seed=0,
        panel_c_xlabel="IAF CV",
        panel_c_title=r"$\mathbf{c}$ Phase error vs. IAF variability",
    )




def main(npz_path=NPZ_PATH, phase_csv_path=PHASE_CSV, iaf_csv_path=IAF_CSV):
    data = np.load(npz_path)
    if "phase_err_unc_deg_all" not in data.files or "phase_err_cal_deg_all" not in data.files:
        raise KeyError(
            "NPZ file must contain 'phase_err_unc_deg_all' and 'phase_err_cal_deg_all'."
        )

    phase_err_unc_deg_all = data["phase_err_unc_deg_all"]
    phase_err_cal_deg_all = data["phase_err_cal_deg_all"]

    print(f"Loaded {phase_err_unc_deg_all.size} uncalibrated samples "
          f"and {phase_err_cal_deg_all.size} calibrated samples from {npz_path}")

    plot_phase_error(
        phase_err_unc_deg_all,
        phase_err_cal_deg_all,
        phase_csv_path=phase_csv_path,
        iaf_csv_path=iaf_csv_path,
    )

if __name__ == "__main__":
    main()
