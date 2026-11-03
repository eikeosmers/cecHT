"""
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
    phase error      = (9 pm 2)°
    amplitude error  = (4 pm 2)%
"""

import numpy as np
from phase import ECHT
from utils import _wrap_phase


def schreglmann(
    sfreq = 256,
    duration = 1,
    f_min = 2,
    f_max = 3,
    n_freqs = 10000,
    filt_order = 2,
    num_phases = 3600,
):

    n_samples = int(round(sfreq * duration))
    t = np.arange(n_samples) / sfreq
    t_end = t[-1]

    freqs = np.linspace(f_min, f_max, n_freqs)

    phase_err_uncal = []
    amp_err_uncal = []
    phase_err_cal = []
    amp_err_cal = []

    r_leak = np.empty(n_freqs)
    arcsin_r = np.empty(n_freqs)
    th_phase_uncal = np.empty((n_freqs, num_phases))
    th_amp_uncal = np.empty((n_freqs, num_phases))
    th_phase_cal = np.empty((n_freqs, num_phases))
    th_amp_cal = np.empty((n_freqs, num_phases))

    # e^{-j2 phi_0}, phi_0 uniform  <=>  leakage phase uniform on the circle
    u = np.exp(-1j * 2 * np.pi * np.arange(num_phases) / num_phases)
    dummy = np.zeros(n_samples)  # to fit (c)ecHT

    for i, f in enumerate(freqs):
        # ecHT band-pass parameters according to Schreglmann et al.
        BW = f / 2
        l_freq = f - BW / 2
        h_freq = f + BW / 2

        # Uncalibrated ecHT
        echt_uncal = ECHT(
            l_freq=l_freq, h_freq=h_freq, sfreq=sfreq,
            n_fft=n_samples, filt_order=filt_order,
            filter_type="butter", calibrate=False, f0=None,
        )
        # Calibrated ecHT
        echt_cal = ECHT(
            l_freq=l_freq, h_freq=h_freq, sfreq=sfreq,
            n_fft=n_samples, filt_order=filt_order,
            filter_type="butter", calibrate=True, f0=f,
        )
        echt_uncal.fit(dummy)
        echt_cal.fit(dummy)

        Gp = echt_cal.calib_err_["Gplus"]
        Gm = echt_cal.calib_err_["Gminus"]
        C_opt = echt_cal.calib_gain_

        r = np.abs(Gm) / np.abs(Gp)
        r_leak[i] = r
        arcsin_r[i] = np.degrees(np.arcsin(r))

        F = Gp + Gm * u  # unc endpoint gain
        Fc = C_opt * F  # cal endpoint gain
        th_phase_uncal[i] = np.degrees(np.abs(np.angle(F)))
        th_amp_uncal[i] = 100 * np.abs(np.abs(F) - 1)
        th_phase_cal[i] = np.degrees(np.abs(np.angle(Fc)))
        th_amp_cal[i] = 100 * np.abs(np.abs(Fc) - 1)

        for n in range(num_phases):
            phi = 2 * np.pi * n / num_phases
            x = np.cos(2 * np.pi * f * t + phi)

            # True analytic endpoint
            phi_true_end = _wrap_phase(2 * np.pi * f * t_end + phi)

            # unc
            z_end_uncal = echt_uncal.fit_transform(x).ravel()[-1]
            phi_hat_uncal = np.angle(z_end_uncal)
            amp_hat_uncal = np.abs(z_end_uncal)

            phase_err_uncal.append(np.degrees(np.abs(_wrap_phase(phi_hat_uncal - phi_true_end))))
            amp_err_uncal.append(100 * abs(amp_hat_uncal - 1))

            # cal
            z_end_cal = echt_cal.fit_transform(x).ravel()[-1]
            phi_hat_cal = np.angle(z_end_cal)
            amp_hat_cal = np.abs(z_end_cal)

            phase_err_cal.append(np.degrees(np.abs(_wrap_phase(phi_hat_cal - phi_true_end))))
            amp_err_cal.append(100 * abs(amp_hat_cal - 1))

    theory = {
        "r": r_leak,
        "arcsin_r": arcsin_r,
        "phase_uncal": th_phase_uncal,
        "amp_uncal": th_amp_uncal,
        "phase_cal": th_phase_cal,
        "amp_cal": th_amp_cal,
    }

    return (
        freqs,
        np.asarray(phase_err_uncal),
        np.asarray(amp_err_uncal),
        np.asarray(phase_err_cal),
        np.asarray(amp_err_cal),
        theory
    )


def main():

    freqs, phase_unc, amp_unc, phase_cal, amp_cal, theory = schreglmann()

    def compare(label, emp, pred, unit):
        print("{}: empirical {:.2f} / {:.2f} / {:.2f} {}"
              "  vs  theory {:.2f} / {:.2f} / {:.2f} {}".format(
            label, emp.mean(), emp.std(ddof=0), emp.max(), unit,
            pred.mean(), pred.std(ddof=0), pred.max(), unit))

    print("\n=== empirical vs theory (mean / std / max) ===")
    compare("Phase err, uncalibrated", phase_unc, theory["phase_uncal"], "deg")
    compare("Phase err, calibrated  ", phase_cal, theory["phase_cal"], "deg")
    compare("Amp. err,  uncalibrated", amp_unc, theory["amp_uncal"], "%")
    compare("Amp. err,  calibrated  ", amp_cal, theory["amp_cal"], "%")

if __name__ == "__main__":
    main()
