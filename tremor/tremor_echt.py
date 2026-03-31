"""
Tremor ecHT plot w/ tracked f0:
(A) Uncalibrated phase error distribution (polar)
(B) Calibrated phase error distribution (polar)
(C) Mean |phase error| vs tremor-frequency CV per trial (paired points + regressions)
"""

import sys
import argparse
from pathlib import Path
from functools import partial

import numpy as np
from scipy.io import loadmat
from scipy.stats import circmean, circstd
from scipy.signal import hilbert, butter, sosfiltfilt, sosfilt, sosfilt_zi, welch

from joblib import Parallel, delayed

from phase_track import ECHT
from utils import (
    _wrap_phase,
    make_figure
)


# Helpers
def _estimate_f0(seg, fs, f_min, f_max, interpolate=False):
    # Estimate dominant frequency in [f_min, f_max]

    x = seg.ravel()
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1/fs)

    band = (freqs >= f_min) & (freqs <= f_max)

    if not np.any(band):
        return np.nan

    mag = np.abs(X[band])
    if mag.size < 3:
        return np.nan

    k0 = int(np.argmax(mag))
    freqs_b = freqs[band]

    # Parabolic interpolation around the peak (in magnitude domain)
    if interpolate:
        if 0 < k0 < (mag.size - 1):
            y1, y2, y3 = mag[k0 - 1], mag[k0], mag[k0 + 1]
            denom = (y1 - 2*y2 + y3)
            if denom != 0:
                delta = 0.5*(y1-y3)/denom  # in bins
                bin_hz = freqs_b[1] - freqs_b[0]
                return freqs_b[k0] + delta * bin_hz

    return freqs_b[k0]

# Tremor frequency CV
def _calc_cv(x_filt, freq_win_len, freq_stride, L, fs):
    freqs_win = []
    if L >= freq_win_len:
        for start in range(0, L - freq_win_len + 1, freq_stride):
            seg_f = x_filt[start:start + freq_win_len]
            freqs, psd = welch(seg_f, fs=fs, nperseg=min(len(seg_f), freq_win_len))
            mask = (freqs >= 0.5) & (freqs <= 20)
            if not np.any(mask):
                continue
            psd_masked = psd[mask]
            if psd_masked.size == 0:
                continue
            freqs_win.append(freqs[mask][np.argmax(psd_masked)])

    if len(freqs_win) >= 2:
        freqs_win = np.asarray(freqs_win, dtype=float)
        mean_f = np.mean(freqs_win)
        std_f = np.std(freqs_win, ddof=1)
        freq_cv = (std_f / mean_f) if mean_f != 0 else np.nan
    else:
        freq_cv = np.nan

    return freq_cv

# Data loading
def process_one_trial(
    x,
    phi_ds,
    fs,
    f0,
    N=256,
    filt_order=2,
    freq_win_len=2048,
    freq_stride=2048,
    track_f0=True,
):
    """
    track_f0
        If True, f0 is updated online during the window loop using an
        exponentially-weighted estimate (tracking mode).
        If False, f0 is held fixed throughout (static mode).

    Returns:
      err_unc, err_cal             : per-window endpoint errors (paired)
      freq_cv                      : trial tremor frequency CV
      trial_abs_unc, trial_abs_cal : per-trial mean |error| (rad)
      trial_mu_unc, trial_mu_cal   : per-trial circular mean error (rad)
      n_windows                    : number of windows
    """

    if len(x) < N:
        return (np.array([]), np.array([]), np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    l_freq = max(0.1, f0 - f0/2)
    h_freq = min(0.5*fs - 0.1, f0 + f0/2)
    if not (0 < l_freq < h_freq < 0.5 * fs):
        return (np.array([]), np.array([]), np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    sos = butter(filt_order, [l_freq, h_freq], fs=fs, btype="bandpass", output="sos")

    # offline reference: non-causal filter
    x_filt = sosfiltfilt(sos, x)
    z_offline = hilbert(x_filt)
    phi_offline = np.angle(z_offline) # baseline, offline filtered

    L = min(len(x), len(phi_offline), len(phi_ds))
    x = x[:L]
    phi_offline = phi_offline[:L]

    # online causal filter: initialise state from first sample
    zi = sosfilt_zi(sos) * x[0]
    x_filt_online, zi = sosfilt(sos, x, zi=zi)  # filter entire trial causally
    # zi is carried forward sample-by-sample implicitly — sosfilt over
    # the full array is equivalent to stepping sample-by-sample with state

    first_seg = x[:N]

    echt_unc = ECHT(
        l_freq=l_freq, h_freq=h_freq, sfreq=fs, filt_order=filt_order,
        calibrate=False, f0=None,
        bandpass_tracking=False, bandpass_update_mode="threshold"
    )
    echt_unc.fit(first_seg)

    echt_cal = ECHT(
        l_freq=l_freq, h_freq=h_freq, sfreq=fs, filt_order=filt_order,
        calibrate=True, f0=f0,
        bandpass_tracking=track_f0, bandpass_update_mode="threshold"
    )
    echt_cal.fit(first_seg)

    out_len = L - (N - 1)
    if out_len <= 0:
        return (np.array([]), np.array([]), np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    err_unc = np.empty(out_len, dtype=float)
    err_cal = np.empty(out_len, dtype=float)

    f0_track = f0
    alpha = 1
    last_f0_hat = np.nan

    k = 0
    for end_idx in range(N - 1, L):
        # ecHT window
        start_idx = end_idx - N + 1
        seg_echt = x[start_idx:end_idx + 1]  # length N

        # f0 tracking buffer
        if track_f0:
            # Update only every freq_stride samples to reduce compute
            if ((end_idx - (N - 1)) % freq_stride) == 0:
                start_f0 = max(0, end_idx - freq_win_len + 1)
                seg_f0 = x_filt_online[start_f0:end_idx + 1]

                # Track in a band around the baseline
                f_min = max(0.1, f0 - 0.5*f0)
                f_max = min(0.5*fs - 0.1, f0 + 0.5*f0)

                last_f0_hat = _estimate_f0(seg_f0, fs, f_min, f_max)

            if np.isfinite(last_f0_hat):
                f0_track = (1 - alpha)*f0_track + alpha * last_f0_hat

        zu = np.squeeze(echt_unc.transform(seg_echt, f0=f0_track))
        zc = np.squeeze(echt_cal.transform(seg_echt, f0=f0_track))

        phi_unc_end = np.angle(zu[-1])
        phi_cal_end = np.angle(zc[-1])
        phi_true_end = phi_offline[end_idx]

        err_unc[k] = _wrap_phase(phi_unc_end - phi_true_end)
        err_cal[k] = _wrap_phase(phi_cal_end - phi_true_end)
        k += 1

    trial_abs_unc = np.mean(np.abs(err_unc))
    trial_abs_cal = np.mean(np.abs(err_cal))
    trial_mu_unc = circmean(err_unc, high=np.pi, low=-np.pi)
    trial_mu_cal = circmean(err_cal, high=np.pi, low=-np.pi)
    n_windows = int(err_unc.size)

    # Tremor frequency CV
    freq_cv = _calc_cv(x_filt_online, freq_win_len, freq_stride, L, fs)

    return (err_unc, err_cal, freq_cv, trial_abs_unc, trial_abs_cal, trial_mu_unc, trial_mu_cal, n_windows)


def iter_trials(mat_path):
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    data_all_res = mat["data_all_res"]

    for subj in np.ravel(data_all_res):
        for cond in np.ravel(subj):
            if cond is None:
                continue
            for trial in np.ravel(cond):
                if trial is None:
                    continue
                if not hasattr(trial, "ADC_in_raw") or trial.ADC_in_raw is None:
                    continue

                x = np.asarray(trial.ADC_in_raw, dtype=float).ravel()
                if x.size < 10:
                    continue

                dt_mean = np.asarray(trial.dt_mean).squeeze()
                fs = 1/dt_mean
                f0 = np.asarray(trial.cal_freq).squeeze()

                phi_ds = np.asarray(trial.ADC_in_phase, dtype=float).ravel()

                L = min(len(x), len(phi_ds))
                yield x[:L], phi_ds[:L], fs, f0


def collect_trials(mat_paths):
    all_trials = []
    for mat_path in mat_paths:
        mat_path = Path(mat_path)
        if not mat_path.exists():
            print(f"File not found, skipping: {mat_path}")
            continue
        this_trials = list(iter_trials(mat_path))
        print(f"Collected {len(this_trials)} trials from {mat_path.name}")
        all_trials.extend(this_trials)
    return all_trials


def compute_endpoint_errors(
    trials,
    process_fn=process_one_trial,
    filt_order=2,
    N=256,
    n_jobs=-1,
    freq_win_len=2048,
    freq_stride=2048,
):
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_fn)(
            x, phi_ds, fs, f0, N, filt_order, freq_win_len, freq_stride
        )
        for (x, phi_ds, fs, f0) in trials
    )

    err_unc_all = []
    err_cal_all = []

    trial_freq_cv = []
    trial_abs_unc = []
    trial_abs_cal = []
    trial_mu_unc = []
    trial_mu_cal = []
    trial_nwin = []

    n_windows_total = 0

    for eu, ec, cv, absu, absc, muu, muc, nwin in results:
        if eu.size and ec.size:
            if eu.size != ec.size:
                m = min(eu.size, ec.size)
                eu = eu[:m]
                ec = ec[:m]
                nwin = m
            err_unc_all.append(eu)
            err_cal_all.append(ec)
            n_windows_total += int(eu.size)

        trial_freq_cv.append(cv)
        trial_abs_unc.append(absu)
        trial_abs_cal.append(absc)
        trial_mu_unc.append(muu)
        trial_mu_cal.append(muc)
        trial_nwin.append(nwin)

    err_unc_all = np.concatenate(err_unc_all) if err_unc_all else np.array([], dtype=float)
    err_cal_all = np.concatenate(err_cal_all) if err_cal_all else np.array([], dtype=float)

    trial_freq_cv = np.asarray(trial_freq_cv, dtype=float)
    trial_abs_unc = np.asarray(trial_abs_unc, dtype=float)
    trial_abs_cal = np.asarray(trial_abs_cal, dtype=float)
    trial_mu_unc = np.asarray(trial_mu_unc, dtype=float)
    trial_mu_cal = np.asarray(trial_mu_cal, dtype=float)
    trial_nwin = np.asarray(trial_nwin, dtype=int)

    print(f"compute_endpoint_errors: {n_windows_total} ecHT windows total")

    return (
        err_unc_all, err_cal_all,
        trial_freq_cv,
        trial_abs_unc, trial_abs_cal,
        trial_mu_unc, trial_mu_cal,
        trial_nwin
    )

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", action="store_true", default=False,
                        help="Use f0-tracking mode (tremor_echt_track behaviour)")
    args = parser.parse_args()

    track_f0 = args.track

    if len(sys.argv) > 1:
        mat_paths = sys.argv[1:]
    else:
        mat_paths = [
            "data_sbj_original_2016.mat",
            # "data_sbj_repeated_2019.mat",
            # "data_sbj_new_2019.mat",
        ]

    print("Using the following .mat files (non-existing ones will be skipped):")
    for p in mat_paths:
        print("  -", p)

    trials = collect_trials(mat_paths)
    print(f"Total trials collected from all selected files: {len(trials)}")

    if len(trials) == 0:
        print("No trials found. Check that the .mat files exist and have the expected structure.")
        return

    filt_order = 2
    N = 128

    process_fn = process_one_trial
    if not track_f0:
        process_fn = partial(process_one_trial, track_f0=False)

    (
        err_unc, err_cal,
        trial_freq_cv,
        trial_abs_unc, trial_abs_cal,
        trial_mu_unc, trial_mu_cal,
        trial_nwin,
    ) = compute_endpoint_errors(
        trials,
        process_fn=process_fn,
        filt_order=filt_order,
        N=N,
        n_jobs=-1,
        freq_win_len=2048,
        freq_stride=2048,
    )

    if err_unc.size == 0 or err_cal.size == 0:
        print("No windows produced any errors (check N and trial lengths).")
        return

    mean_unc = _wrap_phase(circmean(err_unc, high=np.pi, low=-np.pi))
    std_unc = circstd(err_unc, high=np.pi, low=-np.pi)
    mean_cal = _wrap_phase(circmean(err_cal, high=np.pi, low=-np.pi))
    std_cal = circstd(err_cal, high=np.pi, low=-np.pi)

    print("\nSummary of ecHT vs offline Hilbert reference")
    print(f"Mean uncalibrated error (deg): {np.degrees(mean_unc):.2f} ± {np.degrees(std_unc):.2f}")
    print(f"Mean calibrated error   (deg): {np.degrees(mean_cal):.2f} ± {np.degrees(std_cal):.2f}")

    save_base = "tremor_track" if track_f0 else "tremor_echt"
    make_figure(
        err_unc_rad=err_unc,
        err_cal_rad=err_cal,
        trial_freq_cv=trial_freq_cv,
        trial_abs_unc_rad=trial_abs_unc,
        trial_abs_cal_rad=trial_abs_cal,
        trial_mu_unc=trial_mu_unc,
        trial_mu_cal=trial_mu_cal,
        trial_nwin=trial_nwin,
        save_base=save_base,
        n_perm=int(1e5),
        perm_seed=0,
        panel_c_xlabel="Tremor frequency CV",
        panel_c_title=r"$\mathbf{c}$ Phase error vs. tremor variability",
    )


if __name__ == "__main__":
    main()
