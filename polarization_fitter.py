# DOI 10.5281/zenodo.1342163
# GNU General Public License 3.0
# cite as:
# Li, Gang, Evitts, Richard, Boulfiza, Moh, & Li, Alice D.S. (2018, August 11).
# A customized Python module for interactive curve fitting on potentiodynamic scan
# data (Version v1.0.1). Zenodo. http://doi.org/10.5281/zenodo.1343975

import io
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dta_parser import parse_dta

# ── Butler-Volmer math ───────────────────────────────────────────────────────

def BVeq(E, Eeq, i0, Ba, Bc):
    ia = i0 * 10 ** ((E - Eeq) / Ba)
    ic = -i0 * 10 ** ((E - Eeq) / Bc)
    return ia + ic


def Feq(E, Eeq, i0, Ba, Bc, Va, Vc):
    iF = (
        (E >= Eeq) * i0 * (np.abs(E - Eeq) / Va * 10 ** (-(E - Eeq) / Ba)) ** 0.5
        + (E < Eeq) * -i0 * (-np.abs(E - Eeq) / Vc * 10 ** (-(E - Eeq) / Bc)) ** 0.5
    )
    return iF


def BVFeq(E, Eeq, i0, Ba, Bc, Va, Vc):
    return BVeq(E, Eeq, i0, Ba, Bc) + Feq(E, Eeq, i0, Ba, Bc, Va, Vc)


# ── Data loading (cached per uploaded file) ──────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Sample file paths ─────────────────────────────────────────────────────────
_SAMPLE_FILES = {
    "DTA — Gamry Instruments":                             os.path.join(_HERE, "test_file_Icorr10uA.DTA"),
    "Excel — absolute current, one-step scan  (legacy)":  os.path.join(_HERE, "validate_one_step_scan.xlsx"),
    "Excel — absolute current, two-step scan  (legacy)":  os.path.join(_HERE, "validate_two_step_scan.xlsx"),
    # signed format: generated synthetically (no file on disk)
    "Excel / CSV — signed current  (cathodic < 0)":       None,
}


@st.cache_data(show_spinner="Parsing DTA file…")
def _load_dta(file_bytes: bytes, filename: str):
    """Parse Gamry .DTA bytes → (df with signed I/E, metadata dict)."""
    with tempfile.NamedTemporaryFile(suffix=".DTA", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        df, meta = parse_dta(tmp_path)
    finally:
        os.unlink(tmp_path)
    return df.reset_index(drop=True), meta


@st.cache_data(show_spinner="Loading sample DTA…")
def _load_sample_dta(path: str):
    df, meta = parse_dta(path)
    return df.reset_index(drop=True), meta


@st.cache_data(show_spinner="Loading sample Excel…")
def _load_sample_excel_one_step(path: str):
    df = pd.read_excel(path, skiprows=1, header=None, names=["I", "E"]).dropna().copy()
    Ecorr_est = df.loc[df.I == df.I.min(), "E"].values[0]
    df.loc[df.E < Ecorr_est, "I"] *= -1
    return df.reset_index(drop=True)


@st.cache_data(show_spinner="Loading sample Excel…")
def _load_sample_excel_two_step(path: str):
    df = pd.read_excel(path, skiprows=2, header=None,
                       names=["Ic", "Ec", "Ia", "Ea"]).dropna().copy()
    OCP_c = df.loc[df.Ic == df.Ic.min(), "Ec"].values[0]
    OCP_a = df.loc[df.Ia == df.Ia.min(), "Ea"].values[0]
    drift = OCP_a - OCP_c
    df["Ea"] -= drift
    df.loc[df.Ec < OCP_c, "Ic"] *= -1
    df.loc[df.Ea < OCP_a, "Ia"] *= -1
    df_rev = df[["Ec", "Ic"]].sort_index(ascending=False).reset_index(drop=True)
    df_IE = pd.DataFrame({
        "I": pd.concat([df_rev["Ic"], df["Ia"]], ignore_index=True),
        "E": pd.concat([df_rev["Ec"], df["Ea"]], ignore_index=True),
    })
    return df_IE.dropna().reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _generate_signed_sample():
    """Synthetic BV data with known parameters (Ecorr=-0.5 V, Icorr=10 µA)."""
    np.random.seed(42)
    Ecorr, Icorr, Ba, Bc = -0.5, 1e-5, 0.2, -0.3
    E = np.linspace(Ecorr - 0.2, Ecorr + 0.2, 2400)
    I = BVeq(E=E, Eeq=Ecorr, i0=Icorr, Ba=Ba, Bc=Bc)
    E_noisy = E + 2e-3 * np.random.randn(len(E))
    df = pd.DataFrame({"I": I, "E": E_noisy})
    meta = {"note": "Synthetic BV data — Ecorr=−0.5 V, Icorr=10 µA, Ba=0.2, Bc=−0.3"}
    return df.reset_index(drop=True), meta


@st.cache_data(show_spinner="Reading file…")
def _load_excel_signed(file_bytes: bytes):
    """Two-column Excel/CSV: col0 = signed I (A), col1 = E (V), row0 = header."""
    df = pd.read_excel(io.BytesIO(file_bytes), skiprows=1,
                       header=None, names=["I", "E"])
    return df.dropna().reset_index(drop=True)


@st.cache_data(show_spinner="Reading file…")
def _load_excel_absolute_one_step(file_bytes: bytes):
    """Two-column Excel: all-positive current.  Cathodic side auto-detected."""
    df = pd.read_excel(io.BytesIO(file_bytes), skiprows=1,
                       header=None, names=["I", "E"]).dropna().copy()
    Ecorr_est = df.loc[df.I == df.I.min(), "E"].values[0]
    df.loc[df.E < Ecorr_est, "I"] *= -1
    return df.reset_index(drop=True)


@st.cache_data(show_spinner="Reading file…")
def _load_excel_absolute_two_step(file_bytes: bytes):
    """Four-column Excel (Ic, Ec, Ia, Ea): all-positive current, two-step scan."""
    df = pd.read_excel(io.BytesIO(file_bytes), skiprows=2,
                       header=None, names=["Ic", "Ec", "Ia", "Ea"]).dropna().copy()
    OCP_c = df.loc[df.Ic == df.Ic.min(), "Ec"].values[0]
    OCP_a = df.loc[df.Ia == df.Ia.min(), "Ea"].values[0]
    drift = OCP_a - OCP_c
    if abs(drift) > 0.02:
        st.warning(f"OCP drift between half-scans: {drift*1000:.1f} mV — drift correction applied.")
    df["Ea"] = df["Ea"] - drift
    df.loc[df.Ec < OCP_c, "Ic"] *= -1
    df.loc[df.Ea < OCP_a, "Ia"] *= -1
    df_rev = df[["Ec", "Ic"]].sort_index(ascending=False).reset_index(drop=True)
    df_IE = pd.DataFrame({
        "I": pd.concat([df_rev["Ic"], df["Ia"]], ignore_index=True),
        "E": pd.concat([df_rev["Ec"], df["Ea"]], ignore_index=True),
    })
    return df_IE.dropna().reset_index(drop=True)


def _add_density_columns(df_IE: pd.DataFrame, area: float) -> pd.DataFrame:
    df = df_IE.copy()
    df["i_density"] = df["I"] / area
    df["i_density_abs"] = df["I"].abs() / area
    return df


def _quick_ecorr(data: pd.DataFrame) -> float:
    return data.loc[data.i_density_abs == data.i_density_abs.min(), "E"].values[0]


# ── Core fitting + plotting ───────────────────────────────────────────────────

def fit_and_plot(
    data: pd.DataFrame,
    data_range: tuple,
    taf_init: int,
    R: float,
    area: float,
    logx: bool,
    auto_zoom: bool,
    grid_on: bool,
    use_bvf: bool = False,
    title: str = "",
):
    """
    Fit Butler-Volmer (and optionally BV+Film) equation to the selected data
    range and return the figure and a results dict.  Raises RuntimeError with
    a user-readable message on failure.
    """
    start, stop = data_range
    I_all = data["I"].values
    E_all = data.E.values - I_all * R   # IR compensation (E - I·R)

    I_sel = I_all[start:stop]
    E_sel = E_all[start:stop]

    if len(I_sel) < 10:
        raise RuntimeError("Selected range has fewer than 10 points — widen the range.")

    I_a = I_sel[I_sel > 0];        E_a = E_sel[I_sel > 0]
    I_c = np.abs(I_sel[I_sel < 0]); E_c = E_sel[I_sel < 0]

    if len(I_a) < 3 or len(I_c) < 3:
        raise RuntimeError("Not enough anodic or cathodic points — widen the data range.")

    n = max(2, min(taf_init, len(I_a) - 1, len(I_c) - 1))

    Ba_scan, int_a = np.polyfit(np.log10(I_a)[-n:], E_a[-n:], 1)
    Bc_scan, int_c = np.polyfit(np.log10(I_c)[:n],  E_c[:n],  1)
    Icorr_g = 10 ** (-(int_a - int_c) / (Ba_scan - Bc_scan))

    ind_min = np.argmin(np.abs(I_sel))
    OCP = E_sel[ind_min]

    bound = (
        [OCP - 0.001, Icorr_g * 0.01, Ba_scan - 0.20, Bc_scan - 0.20],
        [OCP + 0.001, Icorr_g * 100.0, Ba_scan + 0.20, Bc_scan + 0.20],
    )
    p0 = [OCP, Icorr_g, Ba_scan, Bc_scan]

    try:
        popt, _ = curve_fit(BVeq, E_sel, I_sel, p0, bounds=bound, maxfev=10000)
    except RuntimeError as exc:
        raise RuntimeError(f"curve_fit did not converge: {exc}") from exc

    Ecorr, Icorr, Ba, Bc = popt
    B = Ba * abs(Bc) / (2.303 * (Ba + abs(Bc)))

    # ── BVFeq fit (optional) ──────────────────────────────────────────────
    popt_bvf = None
    r2_bvf = None
    if use_bvf:
        Va_g, Vc_g = 1000.0, -1000.0
        p0_bvf = [Ecorr, Icorr, Ba, Bc, Va_g, Vc_g]
        bound_bvf = (
            [OCP - 0.001, Icorr * 0.01, Ba - 0.20, Bc - 0.20, 1e-4, -1e6],
            [OCP + 0.001, Icorr * 100.0, Ba + 0.20, Bc + 0.20, 1e6, -1e-4],
        )
        try:
            popt_bvf, _ = curve_fit(BVFeq, E_sel, I_sel, p0_bvf,
                                    bounds=bound_bvf, maxfev=20000)
            r2_bvf = r2_score(I_sel, BVFeq(E_sel, *popt_bvf))
        except RuntimeError:
            popt_bvf = None  # BVFeq did not converge; fall back silently

    # LPR
    Rp = Icorr_LPR = None
    df_a = data[(data.E > OCP + 0.005) & (data.E < OCP + 0.020)]
    df_c = data[(data.E > OCP - 0.020) & (data.E < OCP - 0.005)]
    if len(df_a) >= 2 and len(df_c) >= 2:
        Rp1, _ = np.polyfit(df_a["I"], df_a["E"], 1)
        Rp2, _ = np.polyfit(df_c["I"], df_c["E"], 1)
        Rp = float(np.mean([Rp1, Rp2]))
        Icorr_LPR = B / Rp

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    E_line = np.linspace(E_all.min(), E_all.max(), 1000)
    I_sel_abs = np.abs(I_sel)

    def _plot_curve(x, y, fmt=None, **kw):
        if fmt is not None:
            ax.plot(x, y, fmt, **kw)
        else:
            ax.plot(x, y, **kw)

    if logx:
        ax.set_xscale("log")
        _plot_curve(np.abs(I_all[:start]),  E_all[:start],  color="C1", marker=".", ls="none", ms=2, label="discarded")
        _plot_curve(np.abs(I_all[stop:]),   E_all[stop:],   color="C1", marker=".", ls="none", ms=2)
        _plot_curve(I_sel_abs,              E_sel,          color="C0", marker=".", ls="none", ms=2, label="selected")
        _plot_curve(np.abs(BVeq(E_line, *p0)),   E_line, "--", color="green", alpha=0.6, lw=1, label="initial guess")
        _plot_curve(np.abs(BVeq(E_line, *popt)), E_line, "-",  color="red",   lw=1.8,   label="BV fit")
        _plot_curve(Icorr * 10 ** ((E_line - Ecorr) / Ba), E_line, "--", color="red", alpha=0.4)
        _plot_curve(Icorr * 10 ** ((E_line - Ecorr) / Bc), E_line, "--", color="red", alpha=0.4)
        if popt_bvf is not None:
            _plot_curve(np.abs(BVFeq(E_line, *popt_bvf)), E_line, "-", color="C2", lw=1.8, label="BV+Film fit")
        if auto_zoom:
            ax.set_xlim(0.1 * I_sel_abs.min(), 10 * I_sel_abs.max())
    else:
        ax.set_xscale("linear")
        _plot_curve(I_all[:start], E_all[:start], color="C1", marker=".", ls="none", ms=2, label="discarded")
        _plot_curve(I_all[stop:],  E_all[stop:],  color="C1", marker=".", ls="none", ms=2)
        _plot_curve(I_sel,         E_sel,          color="C0", marker=".", ls="none", ms=2, label="selected")
        _plot_curve(BVeq(E_line, *p0),   E_line, "--", color="green", alpha=0.6, lw=1, label="initial guess")
        _plot_curve(BVeq(E_line, *popt), E_line, "-",  color="red",   lw=1.8,   label="BV fit")
        if popt_bvf is not None:
            _plot_curve(BVFeq(E_line, *popt_bvf), E_line, "-", color="C2", lw=1.8, label="BV+Film fit")
        if auto_zoom:
            ax.set_xlim(1.1 * I_sel.min(), 1.1 * I_sel.max())

    ax.set_xlabel("Current [A]")
    ax.set_ylabel("Potential, E [V]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    if grid_on:
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    else:
        ax.grid(False)
    fig.tight_layout()

    r2 = r2_score(I_sel, BVeq(E_sel, *popt))

    result = {
        "Ecorr":      Ecorr,
        "Icorr":      Icorr,
        "Ba":         Ba,
        "Bc":         Bc,
        "B":          B,
        "Rp":         Rp,
        "Icorr_LPR":  Icorr_LPR,
        "R²":         r2,
        "Va":         popt_bvf[4] if popt_bvf is not None else None,
        "Vc":         popt_bvf[5] if popt_bvf is not None else None,
        "R²_bvf":     r2_bvf,
        "range_low":  E_sel[0] - Ecorr,
        "range_high": E_sel[-1] - Ecorr,
    }
    return fig, result


# ── Streamlit app ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Potentiodynamic Scan Fitter",
        page_icon="⚗️",
        layout="wide",
    )
    st.title("Potentiodynamic Scan — Butler-Volmer Curve Fitter")
    st.caption(
        "Interactive BV fitting with LPR cross-check.  "
        "Supports Gamry .DTA, Excel (signed current), and Excel (legacy absolute current)."
    )

    # ── Sidebar: load data ──────────────────────────────────────────────────
    with st.sidebar:
        st.header("1 · Load Data")

        fmt = st.radio(
            "Input format",
            options=[
                "DTA — Gamry Instruments",
                "Excel / CSV — signed current  (cathodic < 0)",
                "Excel — absolute current, one-step scan  (legacy)",
                "Excel — absolute current, two-step scan  (legacy)",
            ],
            index=0,
        )

        accept = (
            ["dta"]          if fmt.startswith("DTA")
            else ["xlsx", "csv"] if "signed" in fmt
            else ["xlsx"]
        )
        uploaded = st.file_uploader(
            "Upload data file",
            type=accept,
            help="One file at a time.  Use the format selector above to match your file.",
        )

        sample_path = _SAMPLE_FILES.get(fmt)
        sample_available = (sample_path is None) or os.path.exists(sample_path)

        btn_col, dl_col = st.columns([1, 1])
        use_sample = btn_col.button(
            "Load sample data",
            help="Load the bundled example file for the selected format.",
            disabled=not sample_available,
        )

        # Download button for the sample file
        if sample_path is not None and os.path.exists(sample_path):
            with open(sample_path, "rb") as _f:
                _sample_bytes = _f.read()
            dl_col.download_button(
                "Download sample",
                data=_sample_bytes,
                file_name=os.path.basename(sample_path),
                mime="application/octet-stream",
                help="Download the sample file to inspect its format.",
            )
        elif sample_path is None:
            # Signed format: offer a CSV download of the synthetic data
            _syn_df, _ = _generate_signed_sample()
            dl_col.download_button(
                "Download sample",
                data=_syn_df.to_csv(index=False).encode(),
                file_name="sample_synthetic_BV.csv",
                mime="text/csv",
                help="Download a synthetic BV scan in signed-current CSV format.",
            )
        # Persist sample-data choice across reruns via session_state.
        # Clear it when the format changes or a real file is uploaded.
        if use_sample:
            st.session_state["sample_fmt"] = fmt
        if uploaded is not None or st.session_state.get("sample_fmt") != fmt:
            st.session_state.pop("sample_fmt", None)

    # ── Parse uploaded file or load sample → df_raw (no area yet) ──────────
    df_raw = None
    meta = {}
    file_title = ""
    using_sample = "sample_fmt" in st.session_state and uploaded is None

    if uploaded is not None:
        file_bytes = uploaded.getvalue()  # getvalue() always returns full bytes; read() empties after first rerun
        file_title = uploaded.name
        try:
            if fmt.startswith("DTA"):
                df_raw, meta = _load_dta(file_bytes, uploaded.name)
            elif "signed" in fmt:
                df_raw = _load_excel_signed(file_bytes)
            elif "one-step" in fmt:
                df_raw = _load_excel_absolute_one_step(file_bytes)
            else:
                df_raw = _load_excel_absolute_two_step(file_bytes)
        except Exception as exc:
            st.error(f"Could not parse file: {exc}")
        if "AREA" in meta:
            st.session_state["suggested_area"] = float(meta["AREA"])
        else:
            st.session_state.pop("suggested_area", None)

    elif using_sample:
        try:
            if fmt.startswith("DTA"):
                df_raw, meta = _load_sample_dta(sample_path)
                file_title = os.path.basename(sample_path)
            elif "signed" in fmt:
                df_raw, meta = _generate_signed_sample()
                file_title = "sample_synthetic_BV.csv"
            elif "one-step" in fmt:
                df_raw = _load_sample_excel_one_step(sample_path)
                file_title = os.path.basename(sample_path)
            else:
                df_raw = _load_sample_excel_two_step(sample_path)
                file_title = os.path.basename(sample_path)
        except Exception as exc:
            st.error(f"Could not load sample: {exc}")
        if "AREA" in meta:
            st.session_state["suggested_area"] = float(meta["AREA"])
        else:
            st.session_state.pop("suggested_area", None)
    else:
        st.session_state.pop("suggested_area", None)

    # ── Sidebar: area input + fit controls (only when data is loaded) ───────
    with st.sidebar:
        st.header("2 · Fit Controls")

        if df_raw is None:
            st.info("Upload a file or click **Load sample data** above to enable controls.")
            area = 1.0
            data = None
            # Provide dummy defaults so variables exist
            ecorr_est = None
            data_range = (0, 1)
            taf_init = 200
            R = 0.0
            use_bvf = False
            auto_zoom = True
            grid_on = True
            logx = True
        else:
            area = st.number_input(
                "Sample area (cm²)",
                value=float(st.session_state.get("suggested_area", 1.0)),
                min_value=1e-6, step=0.1, format="%.4f",
                help="Set to 1.0 to work with raw current (A) instead of current density.",
            )
            data = _add_density_columns(df_raw, area)
            n = len(data)
            ecorr_est = _quick_ecorr(data)

            # Default range: Ecorr ± 0.15 V
            def _idx(target):
                return int(np.abs(data["E"].values - target).argmin())

            def_start = _idx(ecorr_est - 0.15)
            def_end   = _idx(ecorr_est + 0.15)

            data_range = st.slider(
                "Data range  (point indices)",
                min_value=0, max_value=n,
                value=(def_start, def_end), step=1,
            )
            taf_init = st.slider(
                "Tafel init points  (for initial guess)",
                min_value=2, max_value=min(500, n // 4),
                value=min(200, n // 8), step=1,
                help="Number of end-points used for the quick Tafel slope estimate.",
            )
            R = st.slider(
                "R — IR compensation (Ω)",
                min_value=-20000.0, max_value=20000.0,
                value=0.0, step=1.0,
                help="Post-hoc IR compensation: E_corrected = E − I·R.",
            )
            st.divider()
            use_bvf   = st.checkbox("BV + Film model (BVFeq)", value=False,
                                    help="Also fit the Butler-Volmer + Film growth/dissolution model.")
            auto_zoom = st.checkbox("Auto-zoom", value=True)
            grid_on   = st.checkbox("Grid",      value=True)
            logx      = st.checkbox("Log x-axis", value=True)

    # ── Main area ───────────────────────────────────────────────────────────
    if data is None:
        st.info(
            "👈 Upload a file in the sidebar to get started.\n\n"
            "**Supported formats**\n"
            "- **DTA** — Gamry Instruments potentiodynamic scan export\n"
            "- **Excel signed** — two columns: I (signed, A) and E (V), one header row\n"
            "- **Excel absolute one-step** — two columns: |I| (A) and E (V), "
            "cathodic side auto-detected from minimum current\n"
            "- **Excel absolute two-step** — four columns: Ic, Ec, Ia, Ea"
        )
        return

    # ── File info panel ─────────────────────────────────────────────────────
    title_badge = f"**{file_title}**" + ("  *(sample)*" if using_sample else "")
    st.markdown(title_badge)
    info_cols = st.columns(4)
    info_cols[0].metric("Points loaded", len(data))
    info_cols[1].metric("Ecorr estimate", f"{ecorr_est:.4f} V")
    info_cols[2].metric("E range", f"{data.E.min():.3f} → {data.E.max():.3f} V")
    info_cols[3].metric("Area", f"{area} cm²")

    if meta:
        with st.expander("File metadata (DTA header)"):
            st.json(meta)

    st.divider()

    # ── Run fit ─────────────────────────────────────────────────────────────
    try:
        fig, result = fit_and_plot(
            data=data,
            data_range=data_range,
            taf_init=taf_init,
            R=R,
            area=area,
            logx=logx,
            auto_zoom=auto_zoom,
            grid_on=grid_on,
            use_bvf=use_bvf,
            title=file_title,
        )
        fit_ok = True
    except Exception as exc:
        st.error(f"Fitting failed: {exc}")
        fig = None
        result = None
        fit_ok = False

    # ── Plot ────────────────────────────────────────────────────────────────
    plot_col, result_col = st.columns([2, 1])

    with plot_col:
        if fit_ok:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            # Show raw data even when fitting fails
            fig_raw, ax_raw = plt.subplots(figsize=(8, 6))
            if logx:
                ax_raw.semilogx()
                ax_raw.plot(data.i_density_abs, data.E, "C0.", ms=1)
            else:
                ax_raw.plot(data.i_density, data.E, "C0.", ms=1)
            ax_raw.set_xlabel("Current [A]")
            ax_raw.set_ylabel("Potential, E [V]")
            ax_raw.set_title(file_title + "  (raw data — fit failed)")
            ax_raw.grid(True, which="both", linestyle=":", linewidth=0.5)
            fig_raw.tight_layout()
            st.pyplot(fig_raw, use_container_width=True)
            plt.close(fig_raw)

    # ── Results table ────────────────────────────────────────────────────────
    with result_col:
        st.subheader("Results")
        if fit_ok and result is not None:
            icorr_density = result["Icorr"] / area
            icorr_lpr_density = result["Icorr_LPR"] / area if result["Icorr_LPR"] else None
            rows = {
                "Ecorr":                f"{result['Ecorr']:.10g} V",
                "Icorr":               f"{result['Icorr']:.10g} A  ({result['Icorr']*1e6:.6g} µA)",
                "icorr density":       f"{icorr_density:.10g} A/cm²  ({icorr_density*1e6:.6g} µA/cm²)",
                "Ba":                  f"{result['Ba']:.10g} V/dec",
                "Bc":                  f"{result['Bc']:.10g} V/dec",
                "B":                   f"{result['B']:.10g} V",
                "Rp":                  f"{result['Rp']:.10g} Ω" if result["Rp"] else "—",
                "Icorr (LPR)":        (f"{result['Icorr_LPR']:.10g} A  ({result['Icorr_LPR']*1e6:.6g} µA)") if result["Icorr_LPR"] else "—",
                "icorr density (LPR)": f"{icorr_lpr_density:.10g} A/cm²  ({icorr_lpr_density*1e6:.6g} µA/cm²)" if icorr_lpr_density else "—",
                "R²  (BV)":            f"{result['R²']:.10g}",
                "Fit range":           f"{result['range_low']:+.6g} → {result['range_high']:+.6g} V vs Ecorr",
            }
            if use_bvf:
                rows["Va"]           = f"{result['Va']:.10g}" if result["Va"] is not None else "—"
                rows["Vc"]           = f"{result['Vc']:.10g}" if result["Vc"] is not None else "—"
                rows["R²  (BV+Film)"]= f"{result['R²_bvf']:.10g}" if result["R²_bvf"] is not None else "—"
            st.table(pd.DataFrame.from_dict(rows, orient="index", columns=["Value"]))

            st.divider()
            st.caption("**Icorr (LPR)** cross-check uses ±5–20 mV window around OCP.")

    st.divider()
    st.caption(
        "Web app based on: Li, Gang, Evitts, Richard, Boulfiza, Moh, & Li, Alice D.S. (2018). "
        "*A customized Python module for interactive curve fitting on potentiodynamic scan data* "
        "(Version v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.1343975"
    )


if __name__ == "__main__":
    main()
