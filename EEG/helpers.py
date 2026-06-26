"""Helper functions for EEG phase-estimation error analysis.

Contains:
- IAF estimation (FOOOF + BIC peak validation)
- ecHT parameter derivation
- Acausal reference analytic signal
- ecHT vs Hilbert phase-error computation
- Dataset-specific loaders (HMC, Rodrigues2017)
"""

import csv
from collections import namedtuple
from pathlib import Path

from fooof import FOOOF

import mne
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import butter, hilbert, savgol_filter, sosfiltfilt

from phase import ECHT
from utils import _circ_stats, run_echt_window_loop

# IAF estimation
IafResult = namedtuple(
    "IafResult",
    ["PeakAlphaFrequency", "CenterOfGravity", "AlphaBand",
     "delta_bic", "pink_r2"],
)

_DEFAULT_FMIN = 7.5
_DEFAULT_FMAX = 14
_EDGE_SEARCH_SG_WINDOW = 11
_EDGE_SEARCH_SG_POLY = 3


def _fit_fooof(freqs, psd, freq_range=(1, 30)):
    """Fit a FOOOF model (fixed aperiodic, no knee)."""
    fm = FOOOF(
        peak_width_limits=(1, 8),
        max_n_peaks=6,
        min_peak_height=0,
        peak_threshold=2,
        aperiodic_mode="fixed",
        verbose=False,
    )
    fm.fit(freqs, psd, freq_range)
    return fm


def _fooof_aperiodic_log10(fm, freqs):
    """Evaluate the FOOOF aperiodic component in log10-power space."""
    offset, exponent = fm.aperiodic_params_
    return offset - exponent * np.log10(freqs)


def _gaussian_peak(freqs, amp, center, width):
    """Single Gaussian peak (in log10-PSD space)."""
    return amp * np.exp(-0.5 * ((freqs - center) / width) ** 2)


def _bic_peak_test(residual, freqs, fmin, fmax):
    """BIC (Bayesian Informaion Criterion) comparison: no-peak (H0) vs Gaussian peak (H1) on the alpha region.

    Parameters
    ----------
    residual        log10(PSD) - aperiodic fit (the full broadband flattened spectrum).
    freqs:          Corresponding frequency vector (Hz).
    fmin, fmax:     Alpha-band edges (Hz).

    Returns
    -------
    significant:     True if the peak model is preferred over flat by BIC.
    delta_bic        BIC(H0) - BIC(H1).  Positive -> peak model wins.
    """
    fit_mask = (freqs >= fmin) & (freqs <= fmax)
    fit_freqs = freqs[fit_mask]
    fit_resid = residual[fit_mask]
    n = len(fit_resid)

    if n < 4:
        return False, 0

    ss_h0 = np.sum(fit_resid ** 2)
    peak_idx = np.argmax(fit_resid)
    amp_guess = max(fit_resid[peak_idx], 0.01)
    center_guess = fit_freqs[peak_idx]
    width_guess = (fmax - fmin) / 4

    popt, _ = curve_fit(
        _gaussian_peak, fit_freqs, fit_resid,
        p0=[amp_guess, center_guess, width_guess],
        bounds=([0, fmin, 0.25], [np.inf, fmax, fmax - fmin]),
        maxfev=10_000,
    )

    ss_h1 = np.sum((fit_resid - _gaussian_peak(fit_freqs, *popt)) ** 2)
    bic_h0 = n * np.log(ss_h0 / n)
    bic_h1 = n * np.log(max(ss_h1, 1e-30) / n) + 3 * np.log(n)
    return (bic_h0 - bic_h1) > 0, bic_h0 - bic_h1


def iaf(
    raw,
    picks=None,
    fmin=None,
    fmax=None,
    resolution=0.25,
    pink_max_r2=0.9,
):
    """Estimate IAF with FOOOF aperiodic removal and BIC peak validation.

    Parameters
    ----------
    raw
    picks
    fmin, fmax:                Alpha-band edges (Hz).  Auto-detected when None.
    resolution:                Welch frequency resolution (Hz).
    pink_max_r2:               Spectra with log-log R² above this are treated as pure 1/f.

    """
    freq_range = [0.1, 30]
    n_fft = int(raw.info["sfreq"] / resolution)
    w_spec = raw.compute_psd(method="welch", picks=picks, n_fft=n_fft,
                             fmin=freq_range[0], fmax=freq_range[1])
    psd = w_spec.get_data().mean(axis=0)
    freqs = w_spec.freqs

    fm = _fit_fooof(freqs, psd, freq_range)
    residual = np.log10(psd) - _fooof_aperiodic_log10(fm, freqs)
    psd_flat = np.power(10, residual)

    alpha_mask = (freqs >= fmin) & (freqs <= fmax)

    _, _, r, _, _ = stats.linregress(np.log10(freqs), np.log10(psd))
    pink_r2 = r ** 2

    peak_sig, delta_bic = _bic_peak_test(residual, freqs, fmin, fmax)

    if pink_r2 > pink_max_r2 or not peak_sig:
        paf, cog = None, None
    else:
        paf = freqs[alpha_mask][np.argmax(psd_flat[alpha_mask])]
        alpha_weights = psd_flat[alpha_mask]
        cog = (float(np.average(freqs[alpha_mask], weights=alpha_weights))
               if np.any(alpha_weights > 0) else None)
        if cog is None:
            paf = None

    return IafResult(paf, cog, (fmin, fmax), delta_bic, pink_r2)


def estimate_paf(data, info, fmin=7.5, fmax=14,
                 segment_duration=6, polyorder=5, step=0.15):
    """Estimate PAF as the median across sliding Savgol-IAF windows.

    Parameters
    ----------
    data
    info
    fmin, fmax:         Alpha-band search edges (Hz).
    segment_duration:   Duration of each sub-window (s).
    polyorder:          Savgol polynomial order.
    step:               Sliding step size (s).
    """
    raw_tmp = mne.io.RawArray(data[np.newaxis, :], info)
    duration = raw_tmp.times[-1]
    segment_duration = min(duration, segment_duration)

    pafs = []
    for tmin in np.arange(0, duration - segment_duration, step):
        seg = raw_tmp.copy().crop(tmin=tmin, tmax=tmin + segment_duration)
        res = iaf(seg, fmin=fmin, fmax=fmax,
                         pink_max_r2=0.9, resolution=0.17)
        if res.PeakAlphaFrequency is not None and res.PeakAlphaFrequency > 0:
            pafs.append(res.PeakAlphaFrequency)

    return float(np.median(pafs)) if pafs else None

def params_from_f0(fs, f0, bw_factor=0.5):
    """Derive ecHT window length and band-pass edges from centre frequency.
    """
    win_len = int(round(0.512 * fs))   # Bressler et al. (2023): 128 samples @ 250 Hz
    bw = bw_factor * f0
    l_freq = max(f0 - bw/2, 0.1)
    h_freq = min(f0 + bw/2, fs/2 - 0.1)
    return win_len, l_freq, h_freq


def echt_vs_hilbert(data, fs, filt_order, f0, l_freq, h_freq, win_len,
                    ref_analytic_signal):
    """Run ecHT online (uncalibrated + calibrated) and return phase errors (deg).
    """
    n = data.size
    if win_len < 3 or win_len >= n:
        raise ValueError(f"invalid window length (win_len={win_len}, N={n})")

    echt_unc = ECHT(l_freq=l_freq, h_freq=h_freq, sfreq=fs,
                    filt_order=filt_order, calibrate=False)
    echt_cal = ECHT(l_freq=l_freq, h_freq=h_freq, sfreq=fs,
                    filt_order=filt_order, f0=f0, calibrate=True)
    echt_unc.fit(data[:win_len])
    echt_cal.fit(data[:win_len])

    ref_phase = np.angle(ref_analytic_signal)
    err_unc, err_cal = run_echt_window_loop(
        data, ref_phase, echt_unc, echt_cal, win_len
    )
    return np.degrees(err_unc), np.degrees(err_cal)


# Dataset loaders
HMC_EXCLUDE_SUBJECTS = {"SN049"}  # bad recording

def get_first_stage_change_end(annot):
    """Return (t_start, t_end) for the first Wake segment in HMC annotations."""
    sleep_annots = [
        (a["onset"], a["description"])
        for a in annot
        if isinstance(a["description"], str) and a["description"].startswith("Sleep stage")
    ]
    if not sleep_annots:
        raise RuntimeError("No sleep stage annotations found.")

    wake_annots = [(o, d) for o, d in sleep_annots if d == "Sleep stage W"]
    if not wake_annots:
        raise RuntimeError("No Wake (Sleep stage W) annotation found.")

    t_start = wake_annots[0][0]
    later_nonwake = [o for o, d in sleep_annots if o > t_start and d != "Sleep stage W"]
    t_end = (later_nonwake[0] if later_nonwake
             else annot[-1]["onset"] + annot[-1].get("duration", 0))
    return t_start, t_end

def load_hmc(edf_dir, channel_name="EEG O2-M1", window_dur=30, step_dur=15):
    """Load HMC dataset: Split the first Wake segment into multiple windows.

    Parameters
    ----------
    window_dur : float
        Duration of each sub-segment in seconds.
    step_dur : float
        Step size between windows in seconds.
    """
    edf_dir = Path(edf_dir)
    edf_files = sorted(
        p for p in edf_dir.glob("*.edf")
        if not p.name.endswith("_sleepscoring.edf")
        and not any(subj in p.name for subj in HMC_EXCLUDE_SUBJECTS)
    )

    segments = []
    for edf_path in edf_files:
        scoring_path = edf_path.with_name(edf_path.stem + "_sleepscoring.edf")
        if not scoring_path.exists():
            print(f"  Skipping {edf_path.name}: no scoring file")
            continue
        try:
            ann = mne.read_annotations(scoring_path)
            if not ann: continue
            t_start, t_end = get_first_stage_change_end(ann)

            raw = mne.io.read_raw_edf(str(edf_path), preload=True)
            if channel_name not in raw.ch_names: continue

            t_start = max(0, t_start)
            t_end = min(t_end, raw.times[-1])

            # Crop the continuous data once to the Wake period
            raw_wake = raw.copy().pick([channel_name]).crop(tmin=t_start, tmax=t_end)
            full_data = raw_wake.get_data().squeeze()
            fs = float(raw_wake.info["sfreq"])
            full_info = raw_wake.info.copy()

            # Break the Wake period into windows
            total_dur = full_data.size / fs
            current_t = 0
            block_idx = 0

            while current_t + window_dur <= total_dur:
                s0 = int(round(current_t * fs))
                s1 = int(round((current_t + window_dur) * fs))

                # Ensure we don't exceed array bounds
                s1 = min(s1, full_data.size)
                if (s1 - s0) < 10: break  # Skip tiny fragments

                segments.append(dict(
                    subject=edf_path.stem,
                    condition="Wake",
                    block_idx=block_idx,
                    channel=channel_name,
                    duration=window_dur,
                    full_data=full_data,
                    full_info=full_info,
                    fs=fs,
                    sample_start=s0,
                    sample_end=s1,
                ))

                current_t += step_dur
                block_idx += 1

        except Exception as e:
            print(f"  Skipping {edf_path.name}: {e}")

    print(f"Loaded {len(segments)} windows from HMC ({edf_dir})")
    return segments


def load_rodrigues2017(conditions=None, channel_name=None):
    """Load Rodrigues2017 and extract individual EC / EO blocks.

    Each dict carries the full continuous recording (full_data) and
    sample-level indices so the acausal reference can be computed on the
    entire ~100 s recording and then sliced.

    Parameters
    ----------
    conditions:          Filter by "EC" / "EO".
    channel_name:        Preferred channels, in priority order.  Defaults to ["Oz"].

    """
    if channel_name is None:
        channel_name = ["Oz"]

    from moabb.datasets import Rodrigues2017

    dataset = Rodrigues2017()
    subjects = dataset.subject_list

    segments = []
    for subj in subjects:
        data = dataset.get_data(subjects=[subj])
        for sess in data[subj].values():
            for raw in sess.values():
                picks = [ch for ch in channel_name if ch in raw.ch_names]
                if picks:
                    raw_pick = raw.copy().pick(picks)
                    pick_label = picks[0]
                else:
                    raw_pick = raw.copy().pick(eeg=True)
                    pick_label = "all_eeg"

                raw_pick.load_data()
                full_data = raw_pick.get_data().squeeze()
                full_info = raw_pick.info.copy()
                fs = float(raw_pick.info["sfreq"])

                annot = raw_pick.annotations
                if not annot:
                    continue

                for i, (onset, dur, desc) in enumerate(
                        zip(annot.onset, annot.duration, annot.description)):
                    desc_l = str(desc).lower().strip()
                    if "closed" in desc_l or desc == "1":
                        cond = "EC"
                    elif "open" in desc_l or desc == "2":
                        cond = "EO"
                    else:
                        continue
                    if conditions is not None and cond not in conditions:
                        continue

                    tmax = min(onset + dur, raw_pick.times[-1])
                    if tmax - onset < 2:
                        continue

                    s0 = int(round(onset * fs))
                    s1 = min(int(round(tmax * fs)), full_data.size)
                    segments.append(dict(
                        subject=subj, condition=cond, block_idx=i,
                        channel=pick_label,
                        duration=tmax - onset,
                        full_data=full_data,
                        full_info=full_info,
                        fs=fs,
                        sample_start=s0,
                        sample_end=s1,
                    ))

    n_ec = sum(s["condition"] == "EC" for s in segments)
    n_eo = sum(s["condition"] == "EO" for s in segments)
    print(f"Loaded {len(segments)} segments from "
          f"{len({s['subject'] for s in segments})} subjects  "
          f"(EC={n_ec}, EO={n_eo})")
    return segments


# Per-segment processing
def _fail(seg_id, reason):
    return dict(seg_id=str(seg_id), ok=False, reason=reason,
                phase_err_unc=None, phase_err_cal=None, iaf_segments=[])


def process_segment(seg, iaf_window=10, bw_factor=0.5, filt_order=1):
    """Estimate IAF (offline) and compute phase errors for one segment.

    Parameters
    ----------
    seg:               As returned by load_hmc / load_rodrigues2017.
    iaf_window:        Duration (s) of the initial IAF estimation window.  <=0 -> full segment.
    bw_factor:         Bandwidth factor: band = f0 ± bw_factor*f0/2.
    filt_order:        Butterworth order for ecHT and the acausal reference.

    """
    mne.set_log_level("ERROR")
    sid = f"s{seg['subject']}_{seg.get('condition', '')}_b{seg.get('block_idx', 0)}"

    try:
        x = seg["full_data"][seg["sample_start"]:seg["sample_end"]].copy()
        fs = seg["fs"]
        info = seg["full_info"].copy()

        if x.size < 10:
            return _fail(sid, f"segment too short (N={x.size})")

        # Pre-filter: 1–40 Hz bandpass + 46–54 Hz notch
        for sos in [butter(4, [1, 40], fs=fs, btype="band", output="sos"),
                    butter(4, [46, 54], fs=fs, btype="stop", output="sos")]:
            x = sosfiltfilt(sos, x)

        n_iaf = int(min(x.size, round(fs * iaf_window))) if iaf_window > 0 else x.size
        if n_iaf < 10:
            return _fail(sid, f"IAF window too short (N={n_iaf})")

        paf = estimate_paf(x[:n_iaf], info)
        if paf is None:
            return _fail(sid, "no_alpha: no valid PAF")

        win_len, l_freq, h_freq = params_from_f0(fs, paf, bw_factor)
        if win_len < 3 or win_len >= x.size:
            return _fail(sid, f"invalid window length ({win_len=}, N={x.size})")

        # Acausal reference on full recording, then slice
        sos = butter(filt_order, [l_freq, h_freq], fs=fs, btype="band", output="sos")
        ref = hilbert(sosfiltfilt(sos, seg["full_data"]))
        ref_seg = ref[seg["sample_start"]:seg["sample_end"]]

        pe_unc, pe_cal = echt_vs_hilbert(
            x, fs, filt_order, paf, l_freq, h_freq, win_len,
            ref_analytic_signal=ref_seg,
        )

        return dict(
            seg_id=sid, ok=True, reason="",
            phase_err_unc=pe_unc, phase_err_cal=pe_cal,
            iaf_segments=[dict(segment_index=0, had_alpha=1, paf_hz=paf)],
        )

    except Exception as e:
        return _fail(sid, f"error: {e}")


def aggregate_and_save(results, csv_path, npz_path, iaf_csv_path):
    """Aggregate per-segment results, print summary, and write output files.
    """
    all_unc, all_cal = [], []
    per_file_rows, iaf_rows = [], []
    no_alpha, errors = [], []

    for r in results:
        if not r["ok"]:
            bucket = no_alpha if "no_alpha" in r["reason"] else errors
            bucket.append((r["seg_id"], r["reason"]))
            continue

        iaf_rows.extend(
            dict(file=r["seg_id"], segment_index=s["segment_index"],
                 had_alpha=s["had_alpha"], paf_hz=s["paf_hz"])
            for s in r.get("iaf_segments", [])
        )

        pe_unc, pe_cal = r["phase_err_unc"], r["phase_err_cal"]
        all_unc.append(pe_unc)
        all_cal.append(pe_cal)

        m_u, s_u, plv_u, pli_u = _circ_stats(np.radians(pe_unc))
        m_c, s_c, plv_c, pli_c = _circ_stats(np.radians(pe_cal))
        per_file_rows.append(dict(
            file=r["seg_id"], n_samples=pe_unc.size,
            mean_unc_deg=np.degrees(m_u), std_unc_deg=np.degrees(s_u),
            plv_unc=plv_u, pli_unc=pli_u,
            mean_cal_deg=np.degrees(m_c), std_cal_deg=np.degrees(s_c),
            plv_cal=plv_c, pli_cal=pli_c,
        ))

    print(f"\n=== Summary ===\n"
          f"  Total: {len(results)}   Valid: {len(all_unc)}   "
          f"No alpha: {len(no_alpha)}   Errors: {len(errors)}")
    for label, items in [("No alpha", no_alpha), ("Errors", errors)]:
        for path, reason in items:
            print(f"    [{label}] {path}: {reason}")

    # IAF CSV
    with open(iaf_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "segment_index", "had_alpha", "paf_hz"])
        w.writeheader()
        w.writerows(iaf_rows)
    print(f"IAF estimates -> {iaf_csv_path}")

    if not all_unc:
        print("No valid phase-error data.")
        return

    # Per-file CSV
    fields = ["file", "n_samples",
              "mean_unc_deg", "std_unc_deg", "plv_unc", "pli_unc",
              "mean_cal_deg", "std_cal_deg", "plv_cal", "pli_cal"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_file_rows)
    print(f"Per-file stats  -> {csv_path}")

    # Concatenated NPZ
    np.savez(npz_path,
             phase_err_unc_deg_all=np.concatenate(all_unc),
             phase_err_cal_deg_all=np.concatenate(all_cal))
    print(f"Phase errors    -> {npz_path}")
