"""
ecHT Endpoint Error Simulation

Reproduce the endpoint analytic-phase and amplitude error for the
Endpoint-Corrected Hilbert Transform (ecHT), following the setup
of Schreglmann et al. (2021), Fig. 1c–e.

- finite-length cosine (1 s)
- 256 Hz  -> N = 256 samples
- Frequency sweep: f ∈ [2, 3] Hz
- ecHT filter: 2nd-order Butterworth band-pass
      - center frequency f3 = f
      - bandwidth BW = f/2
      - l_freq = 0.75 f
        h_freq = 1.25 f

We compute endpoint analytic-signal errors for:
1) Uncalibrated ecHT
2) Calibrated ecHT (using theoretical C_opt gain)

Typical outcomes (Schreglmann et al. 2021):
    phase error      = (7 pm 2)°
    amplitude error  = (4 pm 2)%
"""

import numpy as np
from phase import ECHT
from utils import _wrap_phase


def schreglmann(
    sfreq = 373,
    duration = 1,
    f_min = 2,
    f_max = 3,
    n_freqs = 1000,
    filt_order = 2,
):
    """

    Returns
    -------
    freqs:                  Frequencies simulated
    phase_err_uncal:        Absolute endpoint phase errors (deg) for uncalibrated ecHT
    amp_err_uncal:          Endpoint amplitude errors (% of true amplitude) for uncalibrated ecHT
    phase_err_cal:          Absolute endpoint phase errors (deg) for calibrated ecHT
    amp_err_cal:            Endpoint amplitude errors (% of true amplitude) for calibrated ecHT
    """
    n_samples = int(round(sfreq * duration))
    t = np.arange(n_samples) / sfreq
    t_end = t[-1]

    freqs = np.linspace(f_min, f_max, n_freqs)

    phase_err_uncal = []
    amp_err_uncal = []
    phase_err_cal = []
    amp_err_cal = []

    for f in freqs:
        num_phases = 180
        for n in range(num_phases):
            phi = 2 * np.pi * n / num_phases
            x = np.cos(2 * np.pi * f * t + phi)

            # True analytic endpoint for a unit-amplitude cosine:
            phi_true_end = _wrap_phase(2 * np.pi * f * t_end + phi)

            # ecHT band-pass parameters according to Schreglmann et al.
            BW = f / 2
            l_freq = f - BW / 2
            h_freq = f + BW / 2

            # Uncalibrated ecHT
            echt_uncal = ECHT(
                l_freq=l_freq,
                h_freq=h_freq,
                sfreq=sfreq,
                n_fft=n_samples,
                filt_order=filt_order,
                filter_type="butter",
                calibrate=False,
                f0=None,
            )
            z_uncal = echt_uncal.fit_transform(x).ravel()
            z_end_uncal = z_uncal[-1]
            phi_hat_uncal = np.angle(z_end_uncal)
            amp_hat_uncal = np.abs(z_end_uncal)

            # Endpoint phase error (deg)
            err_phase_uncal = np.degrees(
                abs(_wrap_phase(phi_hat_uncal - phi_true_end))
            )
            # Amplitude percentage error (%)
            err_amp_uncal = 100 * abs(amp_hat_uncal - 1)

            phase_err_uncal.append(err_phase_uncal)
            amp_err_uncal.append(err_amp_uncal)

            # Calibrated ecHT
            echt_cal = ECHT(
                l_freq=l_freq,
                h_freq=h_freq,
                sfreq=sfreq,
                n_fft=n_samples,
                filt_order=filt_order,
                filter_type="butter",
                calibrate=True,
                f0=f,
            )
            z_cal = echt_cal.fit_transform(x).ravel()
            z_end_cal = z_cal[-1]
            phi_hat_cal = np.angle(z_end_cal)
            amp_hat_cal = np.abs(z_end_cal)

            err_phase_cal = np.degrees(
                abs(_wrap_phase(phi_hat_cal - phi_true_end))
            )
            err_amp_cal = 100 * abs(amp_hat_cal - 1)

            phase_err_cal.append(err_phase_cal)
            amp_err_cal.append(err_amp_cal)

    return (
        freqs,
        np.asarray(phase_err_uncal),
        np.asarray(amp_err_uncal),
        np.asarray(phase_err_cal),
        np.asarray(amp_err_cal),
    )


def main():
    """
    Run simulation with parameters matching Schreglmann et al. (2021).
    Prints summary statistics for uncalibrated and calibrated ecHT.
    """
    (freqs, phase_unc, amp_unc, phase_cal, amp_cal) = schreglmann()

    def summarize(label, phase_err, amp_err):
        print("\n=== {} ===".format(label))
        print(
            "Phase error (deg): mean = {:.2f}, std = {:.2f}, max = {:.2f}".format(
                phase_err.mean(), phase_err.std(ddof=0), phase_err.max()
            )
        )
        print(
            "Amplitude %error: mean = {:.2f}%, std = {:.2f}%, max = {:.2f}%".format(
                amp_err.mean(), amp_err.std(ddof=0), amp_err.max()
            )
        )

    summarize("Uncalibrated ecHT (original)", phase_unc, amp_unc)
    summarize("Calibrated ecHT", phase_cal, amp_cal)

if __name__ == "__main__":
    main()
