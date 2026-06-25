# cecHT
### endpoint corrected Hilbert Transform (ecHT) with optional calibration

[![arXiv](https://img.shields.io/badge/arXiv-2601.13962-b31b1b.svg)](https://arxiv.org/abs/2601.13962)

[Eike Osmers](https://www.tu.berlin/en/mtec/about/teamfotos/eike-osmers), [Dorothea Kolossa](https://www.tu.berlin/en/mtec/about/management-and-administration/dorothea-kolossa)
TU Berlin


---

## 🚀 Getting Started

### Prerequisites

```bash
- Python 3.12+ (but other versions probably work)
 fooof, joblib, matplotlib, mne, numpy, pandas, scipy, tqdm
```

### Installation

```bash
# Clone the repository
git clone https://github.com/eosmers/cecHT.git
cd repo-name

# Install dependencies
pip install -r requirements.txt
```

### Using (c)ecHT in your own script
Minimal working example

```python
import numpy as np
from phase import ECHT

# Parameters
duration = 1           # Duration of recording (s)
N = 128                # number of samples in a window
fs = 256               # sampling rate (Hz)
l_freq, h_freq = 8, 13 # low- & highpass filter cutoff (Hz)
filt_order = 1         # order of bandpass filter
cal = True             # optional calibration
f0 = 10                # Frequency of signal (Hz), required for calibration

# Signal
t = np.arange(int(fs * duration)) / fs
x = np.cos(2 * np.pi * f0 * t)

echt = ECHT(
    l_freq=l_freq, h_freq=h_freq, sfreq=fs, filt_order=filt_order,
    calibrate=cal, f0=f0
)
# Initialize
n_out = len(x) - (N - 1)
z = np.empty(n_out, dtype=complex)

# Fit once
echt.fit(x[:N])

# Online transform
for k, end_idx in enumerate(range(N - 1, len(x))):
    seg = x[end_idx - N + 1:end_idx + 1]
    z[k] = echt.transform(seg).ravel()[0]
```

---

## 🔬 Experiments

### Experiment 1: Simulations
Investigate the performance of ecHT and cecHT on ideal data.

```python
# Static performance
python simulations/simple.py
# Deeper simulations
python simulations/harmonic_experiments.py
# Latency analysis
python simulations/latency.py
```

### Experiment 2: EEG data
How does ecHT and cecHT perform on real EEG data based on the HMC dataset.

```python
python EEG/eeg_phase.py
python EEG/eeg_plot.py
```

### Experiment 3: Tremor data
ecHT and cecHT performance on tremor data from Schreglmann et al.

```python
# w/o frequency tracking
python tremor/tremor_echt.py
# w/ frequency tracking
python tremor/tremor_echt.py --track
```

---

## 📈 Datasets

We investigated the performance of ecHT and c-ecHT on multiple types of data. EEG data based on the EEG Alpha Waves and HMC dataset and tremor
data based on Schregelmann et al. (2021). Download the datasets at the link below and acknowledge the original authors in your work.

- [EEG Alpha Waves Dataset](https://doi.org/10.5281/zenodo.2348891)
- [Haaglanden Medisch Centrum sleep staging database](https://doi.org/10.13026/t4w7-3k21)
- [Replication Data for: Non-invasive Suppression of Essential Tremor via Phase-Locked Disruption of its Temporal Coherence](https://doi.org/10.7910/DVN/Z6EN2I)

---

## 📁 Repository Structure

```
├── EEG/                            # EEG experiments
│   ├── eeg_phase.py                # Phase estimation on EEG data
│   ├── eeg_plot.py/                # Plot of eeg_phase.py
│   └── helpers.py/                 # Helper scripts for eeg_phase.py
├── simulations/                    # Ideal experiments
│   ├── harmonic_experiments.py     # Parameter sweeps 
│   ├── intro_diagram.py            # Fig. 1 of paper
│   ├── latency.py                  # Latency analysis on your machine
│   └── simple.py/                  # Reproduction of Schreglmann et al.'s first experiment
├── tremor/                         # Tremor experiments
│   └── tremor_echt.py              # ECHT performance on tremor data, w/ and w/o frequency tracking
├── phase.py                        # Main function, contains ECHT class
├── phase_track.py                  # frequency tracking variant of phase.py
├── README.md                       # This file
├── requirements.txt                # Python dependencies
└── utils.py                        # Helper functions
```

---

## 📄 Abstract

Accurate, low-latency estimates of the instantaneous phase of a narrow- band oscillation are central to closed-loop
sensing and actuation, including (but not limited to) phase-locked neurostimulation and other real-time applications.
The endpoint-corrected Hilbert transform (ecHT) reduces boundary artefacts of Hilbert approaches by applying a causal
narrow-band filter to the analytic spectrum, thereby improving the phase estimate at the most recent sample. Despite
broad empirical use, ecHT’s systematic endpoint distortions have lacked a principled, closed-form error analysis.
Here we derive the ecHT endpoint operator analytically and show that its output can be decomposed into a desired
positive-frequency term (a deterministic complex gain that induces a calibratable amplitude/phase bias) and a residual
leakage term that sets an irreducible variance floor. The resulting calibrated ecHT achieves near-zero mean phase error
and remains computationally compatible with real-time pipelines.

---

## 🎯 Key Contributions

- **Contribution 1:** Explicit characterisation and bounds for endpoint phase/amplitude error
- **Contribution 2:** Mean-squared-error-optimal scalar calibration
- **Contribution 3:** Practical design rules relating window length, bandwidth/order, and centre-frequency mismatch to residual bias via an endpoint group delay


---

## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{osmersOptimalCalibration2026,
  title = {Optimal Calibration of the endpoint-corrected Hilbert Transform},
  author={Osmers, Eike AND Kolossa, Dorothea},
  journal={arXiv preprint arXiv:2601.13962},
  year={2026}
  doi = {10.48550/ARXIV.2601.13962},
}
```

---

## 📅 Updates

- **21 Jan 2026** Initial preprint release
- **26 Jun 2026** Camera-ready version release

