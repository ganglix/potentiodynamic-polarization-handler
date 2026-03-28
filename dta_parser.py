"""
dta_parser.py — Gamry Instruments .DTA file parser

Parses potentiodynamic scan data exported by Gamry Framework software and
returns a signed-current DataFrame ready for use with the Info class in
Tafel_LPR_fit_notebook_IRcomp_IFilm.py.

Usage
-----
    from dta_parser import parse_dta

    df, meta = parse_dta("experiment.DTA")
    # df has columns: 'I' (current, A, signed) and 'E' (potential, V vs. Ref.)
    # meta is a dict with keys like 'AREA', 'EOC', 'SCANRATE', 'DATE', ...

    # Pass to Info using the signed-current path:
    info = tf.Info(
        "experiment.DTA",
        pd_dfIE=df,
        use_pd_df=True,
        area=meta.get("AREA", 1.0),
    )
"""

import pandas as pd


# Scalar metadata keys to extract and their value type
_FLOAT_KEYS = {"AREA", "SCANRATE", "EOC", "BETAA", "BETAC",
               "DENSITY", "EQUIV", "VINIT", "VFINAL"}
_STR_KEYS   = {"DATE", "TIME", "TAG", "TITLE", "PSTAT", "SYSTEM"}


def parse_dta(filepath):
    """
    Parse a Gamry Instruments .DTA file.

    The parser handles:
    - A single CURVE TABLE section (standard for potentiodynamic scans).
    - Signed current (Im column): negative = cathodic, positive = anodic.
    - Arbitrary column ordering — column positions are read from the header row.

    Parameters
    ----------
    filepath : str
        Path to the .DTA file.

    Returns
    -------
    df : pd.DataFrame
        Two columns:
          - ``'I'``: current in Amperes (signed: negative = cathodic)
          - ``'E'``: potential in V vs. reference electrode
    metadata : dict
        Scalar values extracted from the file header.  Common keys:

        =========  ============================================================
        ``AREA``   Sample area (cm²)
        ``EOC``    Open-circuit potential (V)
        ``SCANRATE`` Scan rate (mV/s)
        ``DATE``   Date string
        ``TIME``   Time string
        ``TITLE``  Experiment title
        ``TAG``    Experiment type (e.g. ``'POTENTIODYNAMIC'``)
        =========  ============================================================

    Raises
    ------
    ValueError
        If no CURVE TABLE section is found in the file.
    """
    metadata = {}
    data_rows = []

    # State machine: "header" → "col_names" → "units" → "data"
    state = "header"
    col_e = None   # index of potential column (Vf)
    col_i = None   # index of current column (Im)

    # DTA files are written by Windows software; fall back to latin-1 for
    # non-ASCII characters in notes or labels.
    try:
        fh = open(filepath, "r", encoding="utf-8")
        fh.read(1)
        fh.seek(0)
    except UnicodeDecodeError:
        fh = open(filepath, "r", encoding="latin-1")

    try:
        for raw_line in fh:
            line = raw_line.rstrip("\n").rstrip("\r")
            parts = line.split("\t")

            # ── header section ────────────────────────────────────────────
            if state == "header":
                key = parts[0].strip()

                if key == "CURVE" and len(parts) >= 2 and parts[1] == "TABLE":
                    state = "col_names"
                    continue

                if len(parts) >= 3:
                    if key in _FLOAT_KEYS:
                        try:
                            metadata[key] = float(parts[2])
                        except ValueError:
                            metadata[key] = parts[2]
                    elif key in _STR_KEYS:
                        metadata[key] = parts[2]

            # ── column name row (first row after CURVE TABLE) ─────────────
            elif state == "col_names":
                col_names = [p.strip() for p in parts]
                if "Vf" not in col_names or "Im" not in col_names:
                    raise ValueError(
                        f"Expected 'Vf' and 'Im' columns in CURVE section, "
                        f"found: {col_names}"
                    )
                col_e = col_names.index("Vf")
                col_i = col_names.index("Im")
                state = "units"

            # ── units row — skip ──────────────────────────────────────────
            elif state == "units":
                state = "data"

            # ── data rows ─────────────────────────────────────────────────
            elif state == "data":
                # Data rows start with a leading tab → parts[0] == ""
                if not parts[0] == "":
                    break  # reached the next section header
                try:
                    e_val = float(parts[col_e])
                    i_val = float(parts[col_i])
                    data_rows.append((i_val, e_val))
                except (ValueError, IndexError):
                    break
    finally:
        fh.close()

    if state == "header":
        raise ValueError(
            f"No 'CURVE TABLE' section found in {filepath!r}. "
            "Is this a valid Gamry .DTA file?"
        )

    df = pd.DataFrame(data_rows, columns=["I", "E"])
    return df, metadata
