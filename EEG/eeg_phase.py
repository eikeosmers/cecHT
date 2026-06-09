"""Compute EEG phase-estimation errors for ecHT across datasets.

Evaluates the online phase output of the Endpoint-Corrected Hilbert Transform
(ecHT / ECHT) against an acausal reference (band-pass filtfilt + Hilbert).

Supported datasets
------------------
- **hmc**           : HMC sleep dataset (local EDF files + sleep scoring).
                      Crops each recording to the first Wake segment.
- **rodrigues2017** : Rodrigues2017 (Alpha Waves) resting-state via MOABB.
                      10 × 10 s blocks (5 EC, 5 EO) per subject.

Workflow
--------
1. Load dataset segments via the appropriate loader.
2. For each segment, estimate the individual alpha frequency (IAF) once on
   an initial window using FOOOF-based BIC peak validation.
3. Run ecHT online (uncalibrated + calibrated) and compare to the acausal
   reference.  Phase error = angle(z_echt · conj(z_ref)) in degrees.

Outputs
-------
- phase_error_per_file.csv   per-segment circular statistics
- phase_error_all.npz        concatenated phase-error samples (deg)
- iaf_per_segment.csv        per-segment IAF estimates
"""

import argparse

from joblib import Parallel, delayed

from helpers import (
    load_hmc,
    load_rodrigues2017,
    process_segment,
    aggregate_and_save,
)

_CSV_PATH = "phase_error_per_file.csv"
_NPZ_PATH = "phase_error_all.npz"
_IAF_CSV  = "iaf_per_segment.csv"
_N_JOBS   = -1


def main(
    dataset,
    iaf_window=10.0,
    conditions=None,
    bw_factor=0.5,
    filt_order=1,
    edf_dir=None,
    channel_name=None,
    csv_path=_CSV_PATH,
    npz_path=_NPZ_PATH,
    iaf_csv_path=_IAF_CSV,
):
    if dataset == "hmc":
        if edf_dir is None:
            raise ValueError("--edf-dir is required for the HMC dataset")
        kwargs = dict(edf_dir=edf_dir)
        if channel_name is not None:
            kwargs["channel_name"] = channel_name
        segments = load_hmc(**kwargs)

    elif dataset == "rodrigues2017":
        segments = load_rodrigues2017(conditions=conditions)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    if not segments:
        print("No segments loaded. Exiting.")
        return

    print(f"\n{len(segments)} segments  |  "
          f"IAF window: {iaf_window:.1f} s  |  "
          f"bw_factor: {bw_factor}  |  filt_order: {filt_order}")

    results = Parallel(n_jobs=_N_JOBS, backend="loky")(
        delayed(process_segment)(seg, iaf_window, bw_factor, filt_order)
        for seg in segments
    )
    for r in results:
        print(f"  {r['seg_id']}: {r['reason'] or 'OK'}")

    aggregate_and_save(results, csv_path, npz_path, iaf_csv_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compute ecHT phase-estimation errors for EEG datasets.")

    p.add_argument("--dataset", choices=["hmc", "rodrigues2017"],
                   default="rodrigues2017")
                   # default = "hmc")
    p.add_argument("--iaf-window", type=float, default=10,
                   help="IAF estimation window (s). <=0 -> full segment. (default: 10)")
    p.add_argument("--subjects", type=int, default=None,
                   help="Max number of subjects (default: all).")
    p.add_argument("--conditions", nargs="*", default="EC",
                   help="Conditions to include, e.g. EC EO (Rodrigues2017 only).")
    p.add_argument("--bw", type=float, default=0.5,
                   help="Bandwidth factor: f0 ± bw*f0/2 (default: 0.5).")
    p.add_argument("--filt-order", type=int, default=1,
                   help="Butterworth filter order (default: 1).")
    p.add_argument("--edf-dir", type=str, default="HMC/1.1/recordings",
                   help="Directory with EDF files (required for HMC).")
    p.add_argument("--channel", type=str, default=None,
                   help="EEG channel name (HMC only; default: 'EEG O2-M1').")
    p.add_argument("--csv", type=str, default=_CSV_PATH,
                   help=f"Output CSV path (default: {_CSV_PATH}).")
    p.add_argument("--npz", type=str, default=_NPZ_PATH,
                   help=f"Output NPZ path (default: {_NPZ_PATH}).")
    p.add_argument("--iaf-csv", type=str, default=_IAF_CSV,
                   help=f"IAF estimates CSV path (default: {_IAF_CSV}).")

    args = p.parse_args()
    main(
        dataset=args.dataset,
        iaf_window=args.iaf_window,
        conditions=args.conditions,
        bw_factor=args.bw,
        filt_order=args.filt_order,
        edf_dir=args.edf_dir,
        channel_name=args.channel,
        csv_path=args.csv,
        npz_path=args.npz,
        iaf_csv_path=args.iaf_csv,
    )
