# Potentiodynamic Polarization Handler

A Python toolkit for interactive Butler-Volmer curve fitting on potentiodynamic scan data.

DOI: [10.5281/zenodo.1342163](http://doi.org/10.5281/zenodo.1343975)
License: GNU General Public License 3.0

---

## Features

- Fit Butler-Volmer (BV) and BV + Film growth/dissolution (BVFeq) equations
- Linear Polarization Resistance (LPR) cross-check
- Post-hoc IR compensation
- Two interfaces: **Jupyter Notebook** (ipywidgets) and **Streamlit web app**
- Output: Ecorr, Icorr, icorr density, Ba, Bc, B, Rp, R², Chi²

---

## Files

| File | Description |
|---|---|
| `Tafel_LPR_fit_notebook_IRcomp_IFilm.py` | Core module — `Info`, `Tafit` classes and equation functions |
| `dta_parser.py` | Gamry Instruments `.DTA` file parser |
| `polarization_fitter.py` | Streamlit web app |
| `test_driver.ipynb` | Tutorial notebook (interactive, ipywidgets) |
| `tests/test_tafel_lpr.py` | Formal unit test suite (65 tests) |
| `test_file_Icorr10uA.DTA` | Sample Gamry DTA file |
| `validate_one_step_scan.xlsx` | Sample one-step scan (absolute current) |
| `validate_two_step_scan.xlsx` | Sample two-step scan (absolute current) |

---

## Input Formats

| Format | Description |
|---|---|
| `.DTA` | Gamry Instruments potentiodynamic scan export. Signed current parsed automatically. Sample area read from file metadata. |
| `.xlsx` / `.csv` — signed | Two columns: `I` (signed A, cathodic < 0) and `E` (V). One header row. |
| `.xlsx` — absolute, one-step | Two columns: `\|I\|` and `E`. Cathodic side auto-detected from current minimum. |
| `.xlsx` — absolute, two-step | Four columns: `Ic`, `Ec`, `Ia`, `Ea`. OCP drift correction applied automatically. |

---

## Installation

Python ≥ 3.8 (Anaconda recommended).

```bash
pip install numpy pandas matplotlib scipy scikit-learn ipywidgets streamlit openpyxl
```

---

## Streamlit App

```bash
streamlit run polarization_fitter.py
```

**Workflow:**

1. Select input format in the sidebar
2. Upload a file or click **Load sample data** to use the bundled example
3. Set sample area (cm²) — auto-filled from DTA metadata when available
4. Adjust data range, Tafel init points, and IR compensation with sliders
5. Optionally enable **BV + Film model** to fit BVFeq alongside BVeq
6. Results table shows Ecorr, Icorr, icorr density, Ba, Bc, B, Rp, R², and LPR cross-check values

---

## Jupyter Notebook

Open `test_driver.ipynb` for a step-by-step tutorial covering:

- One-step and two-step scan loading
- Interactive BV fitting with `BV_LPR_interact()`
- BV + Film fitting with `BVF_LPR_interact()`
- Validation against synthetic data with known parameters
- Multi-scan comparison with `plot_compare()`

---

## Python API

```python
import Tafel_LPR_fit_notebook_IRcomp_IFilm as tf

# Load a Gamry DTA file
info = tf.Info("test_file_Icorr10uA.DTA", area=2.929)

# Load an Excel file (signed current)
info = tf.Info("data.xlsx", pd_dfIE=df, use_pd_df=True, area=1.0)

# Fit
tafit = tf.Tafit(info)
tafit.BV_LPR_interact(anodic_range=0.15, cathodic_range=0.15)

# Access results
print(tafit.result)   # Ecorr, Icorr, Ba, Bc, B, Rp, Icorr_LPR
```

### DTA parser (standalone)

```python
from dta_parser import parse_dta

df, meta = parse_dta("experiment.DTA")
# df: columns I (A, signed) and E (V)
# meta: dict with AREA, EOC, SCANRATE, DATE, ...
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

65 tests covering `BVeq`, `Feq`, `BVFeq`, `Info` (DataFrame / Excel / DTA loading), and `Tafit` fitting accuracy against synthetic ground-truth data.

---

## Citation

Li, Gang, Evitts, Richard, Boulfiza, Moh, & Li, Alice D.S. (2018, August 11).
*A customized Python module for interactive curve fitting on potentiodynamic scan data* (Version v1.0.1).
Zenodo. http://doi.org/10.5281/zenodo.1343975
