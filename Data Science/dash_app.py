#!/usr/bin/env python3
import argparse
import base64
import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

H_DIGUE = 3.6
DEFAULT_THRESHOLD = 10000
COLORWAY = ["#0f8b8d", "#ff6b35", "#1d3557", "#2a9d8f", "#e9c46a", "#264653"]

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --bg: #f6efe4;
  --ink: #1b1b1b;
  --muted: #5c5c5c;
  --card: rgba(255, 255, 255, 0.9);
  --accent: #0f8b8d;
  --accent-2: #ff6b35;
  --accent-3: #1d3557;
  --ring: rgba(15, 139, 141, 0.35);
  --shadow: 0 18px 45px rgba(25, 25, 25, 0.15);
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: 'IBM Plex Sans', sans-serif;
  color: var(--ink);
  background:
    __BG_LAYER__,
    radial-gradient(1200px 600px at 10% -10%, rgba(255, 214, 170, 0.5), transparent 70%),
    radial-gradient(1000px 800px at 110% 10%, rgba(119, 200, 190, 0.35), transparent 60%),
    linear-gradient(130deg, #f8efe5 0%, #f1f6f6 45%, #f7efe3 100%);
  background-attachment: fixed;
}

.app-shell {
  position: relative;
  min-height: 100vh;
  padding: 36px 22px 48px;
  max-width: 1200px;
  margin: 0 auto;
}

.app-shell::before,
.app-shell::after {
  content: "";
  position: absolute;
  pointer-events: none;
  z-index: 0;
}

.app-shell::before {
  width: 380px;
  height: 380px;
  border-radius: 999px;
  background: radial-gradient(circle at 30% 30%, rgba(15, 139, 141, 0.35), transparent 65%);
  top: 40px;
  right: -80px;
}

.app-shell::after {
  width: 520px;
  height: 320px;
  border-radius: 80px;
  background: radial-gradient(circle at 70% 40%, rgba(255, 107, 53, 0.35), transparent 70%);
  bottom: 80px;
  left: -120px;
  transform: rotate(-8deg);
}

.app-shell > * {
  position: relative;
  z-index: 1;
}

.hero {
  display: grid;
  grid-template-columns: 1.3fr 0.7fr;
  gap: 24px;
  align-items: center;
  margin-bottom: 24px;
  position: relative;
  z-index: 5;
}

.eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.18em;
  font-size: 12px;
  font-weight: 600;
  color: var(--accent-3);
}

.hero-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 42px;
  line-height: 1.05;
  margin: 8px 0 12px;
}

.hero-subtitle {
  font-size: 16px;
  color: var(--muted);
  max-width: 560px;
}

.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 14px;
}

.chip {
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(29, 53, 87, 0.1);
  color: #1d3557;
  font-size: 12px;
  font-weight: 600;
}

.control-card,
.kpi-card,
.graph-card,
.about-card {
  background: var(--card);
  border: 1px solid rgba(25, 25, 25, 0.08);
  border-radius: 18px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(6px);
}

.control-card {
  padding: 18px;
  position: relative;
  z-index: 6;
  overflow: visible;
}

.card-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 18px;
  margin: 0 0 10px;
}

.card-hint {
  font-size: 13px;
  color: var(--muted);
  margin-top: 10px;
}

.scenario-label {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--muted);
  margin-bottom: 6px;
  display: block;
}

.scenario-dropdown {
  position: relative;
  z-index: 9999;
}

.scenario-dropdown .Select-control {
  border-radius: 14px;
  border: 1px solid rgba(20, 20, 20, 0.12);
  min-height: 44px;
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.08);
}

.scenario-dropdown .Select-control:hover {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--ring);
}

.scenario-dropdown .Select-placeholder,
.scenario-dropdown .Select-value-label {
  font-weight: 600;
  color: var(--ink);
}

.scenario-dropdown .Select-menu-outer,
.scenario-dropdown .Select-menu {
  border-radius: 12px;
  border: 1px solid rgba(20, 20, 20, 0.08);
  box-shadow: var(--shadow);
  z-index: 9999 !important;
}

.slider-wrap {
  margin-top: 14px;
}

.kpi-card {
  padding: 18px 20px;
  position: relative;
  overflow: hidden;
  margin-bottom: 24px;
}

.kpi-card::before {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: 18px;
  border: 2px solid transparent;
  background: linear-gradient(120deg, rgba(15, 139, 141, 0.35), rgba(255, 107, 53, 0.25)) border-box;
  -webkit-mask:
    linear-gradient(#fff 0 0) padding-box,
    linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  opacity: 0.6;
  pointer-events: none;
}

.kpi-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 20px;
  margin: 0 0 12px;
}

.smax-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
  position: relative;
  z-index: 3;
}

.smax-card {
  padding: 22px;
  position: relative;
  overflow: hidden;
}

.smax-card::before {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: 18px;
  opacity: 0.9;
  background: linear-gradient(140deg, rgba(15, 139, 141, 0.2), rgba(255, 255, 255, 0.5));
  pointer-events: none;
}

.smax-card--pred::before {
  background: linear-gradient(140deg, rgba(255, 107, 53, 0.22), rgba(255, 255, 255, 0.55));
}

.smax-label {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.16em;
  color: var(--muted);
  margin-bottom: 10px;
  position: relative;
  z-index: 1;
}

.smax-value {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 36px;
  font-weight: 700;
  position: relative;
  z-index: 1;
}

.smax-unit {
  font-size: 16px;
  color: var(--muted);
  margin-left: 8px;
}

.smax-note {
  font-size: 12px;
  color: var(--muted);
  margin-top: 10px;
  position: relative;
  z-index: 1;
}

.kpi-line {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 14px;
  margin: 8px 0;
}

.kpi-label-text {
  font-weight: 600;
}

.badge {
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.04em;
  color: #fff;
}

.badge-yes {
  background: #ff6b35;
}

.badge-no {
  background: #2a9d8f;
}

.badge-na {
  background: #8d99ae;
}

.graph-card {
  padding: 16px 16px 6px;
  margin-bottom: 24px;
}

.section-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin: 4px 4px 10px;
}

.section-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 20px;
  margin: 0;
}

.section-subtitle {
  font-size: 12px;
  color: var(--muted);
}

.about-card {
  padding: 18px 20px;
}

.about-list {
  list-style: none;
  padding: 0;
  margin: 12px 0 0;
}

.about-list li {
  padding: 8px 0 8px 22px;
  position: relative;
  font-size: 14px;
  color: var(--muted);
}

.about-list li::before {
  content: "";
  position: absolute;
  left: 0;
  top: 14px;
  width: 10px;
  height: 10px;
  border-radius: 3px;
  background: linear-gradient(120deg, var(--accent), var(--accent-2));
}

.footer-note {
  margin-top: 18px;
  font-size: 12px;
  color: var(--muted);
  list-style: none;
  padding: 0;
}

.footer-note li {
  margin: 4px 0;
}

.reveal {
  animation: riseFade 0.8s cubic-bezier(0.2, 0.7, 0.2, 1) both;
}

@keyframes riseFade {
  from { opacity: 0; transform: translateY(18px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 980px) {
  .hero {
    grid-template-columns: 1fr;
  }
  .hero-title {
    font-size: 34px;
  }
}
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dash app for coastal flood scenarios (DaMS4)."
    )
    parser.add_argument(
        "--scenarios_dir",
        default="mon_dossier_csv_converti",
        help="Directory containing Dataset_X_sc_<ID>.csv files",
    )
    parser.add_argument(
        "--features_file",
        default="resume_Smax_final.csv",
        help="CSV file containing aggregated scenario features",
    )
    parser.add_argument(
        "--y_file",
        default="Y_Smax.csv",
        help="CSV file containing real Y_Smax values",
    )
    parser.add_argument(
        "--model_path",
        default="best_model.joblib",
        help="Path to best_model.joblib",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Dash server host")
    parser.add_argument("--port", type=int, default=8050, help="Dash server port")
    return parser.parse_args()


def resolve_path(base_dir, path_str):
    if path_str is None:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = base_dir / path
    return path


def normalize_key(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def parse_scenario_id(name):
    match = re.search(r"Dataset_X_sc_(\d+)", str(name))
    if match:
        return match.group(1)
    return None


def list_scenarios(scenarios_dir):
    scenarios = []
    if not scenarios_dir.exists():
        return scenarios
    pattern = re.compile(r"Dataset_X_sc_(\d+)\.csv$")
    for path in sorted(scenarios_dir.glob("Dataset_X_sc_*.csv")):
        match = pattern.match(path.name)
        scenario_id = match.group(1) if match else path.stem
        scenarios.append((str(scenario_id), path))
    scenarios.sort(key=lambda item: int(item[0]) if item[0].isdigit() else item[0])
    return scenarios


def safe_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number):
        return None
    return number


def build_app_css(base_dir):
    bg_path = base_dir / "background.png"
    if not bg_path.exists():
        return APP_CSS.replace("__BG_LAYER__", "none")

    try:
        encoded = base64.b64encode(bg_path.read_bytes()).decode("ascii")
        bg_layer = f"url('data:image/png;base64,{encoded}') center / cover no-repeat fixed"
    except Exception:
        bg_layer = "none"

    return APP_CSS.replace("__BG_LAYER__", bg_layer)


def load_features_table(features_file):
    if features_file is None or not features_file.exists():
        return None, "fichier features introuvable"
    try:
        df = pd.read_csv(features_file)
    except Exception:
        return None, "lecture features impossible"
    return df, ""


def index_features_table(df):
    mapping = {}
    if df is None or df.empty:
        return mapping

    col_map = {normalize_key(col): col for col in df.columns}
    name_col = col_map.get("nomfichier")
    scenario_col = col_map.get("sc") or col_map.get("scenario") or col_map.get("id")

    for _, row in df.iterrows():
        scenario_id = None
        if name_col:
            scenario_id = parse_scenario_id(row.get(name_col))
        elif scenario_col:
            raw_id = row.get(scenario_col)
            if raw_id is not None and str(raw_id).strip() != "":
                scenario_id = str(raw_id).strip()

        if scenario_id is None:
            continue

        mapping[str(scenario_id)] = row.to_dict()

    return mapping


def load_y_lookup(y_file):
    if y_file is None or not y_file.exists():
        return {}, "fichier Y_Smax introuvable"
    try:
        df = pd.read_csv(y_file)
    except Exception:
        return {}, "lecture Y_Smax impossible"

    col_map = {normalize_key(col): col for col in df.columns}
    y_col = None
    for cand in ["y_smax", "ysmax", "smax", "s_max"]:
        norm = normalize_key(cand)
        if norm in col_map:
            y_col = col_map[norm]
            break

    if y_col is None:
        return {}, "colonne Y_Smax introuvable"

    name_col = col_map.get("nomfichier")
    scenario_col = col_map.get("sc") or col_map.get("scenario") or col_map.get("id")

    lookup = {}
    for _, row in df.iterrows():
        scenario_id = None
        if name_col:
            scenario_id = parse_scenario_id(row.get(name_col))
        elif scenario_col:
            raw_id = row.get(scenario_col)
            if raw_id is not None and str(raw_id).strip() != "":
                scenario_id = str(raw_id).strip()

        if scenario_id is None:
            continue

        value = safe_float(row.get(y_col))
        if value is None:
            continue
        lookup[str(scenario_id)] = value

    if not lookup:
        return {}, "Y_Smax reel indisponible"

    return lookup, ""


def load_model(model_path):
    if model_path is None or not model_path.exists():
        return None, "modele non charge"
    try:
        payload = joblib.load(model_path)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "module manquant")
        return None, f"modele non charge ({missing})"
    except Exception:
        return None, "modele non charge"

    model = None
    feature_names = None
    target = None
    if isinstance(payload, dict):
        model = payload.get("model")
        feature_names = payload.get("feature_names")
        target = payload.get("target")
    else:
        model = payload
        feature_names = getattr(payload, "feature_names_in_", None)

    if model is None or feature_names is None:
        return None, "modele non charge (format invalide)"

    return (
        {
            "model": model,
            "feature_names": list(feature_names),
            "target": target,
        },
        "",
    )


def build_feature_vector(feature_names, features_dict):
    data = {name: float(features_dict.get(name, 0.0)) for name in feature_names}
    return pd.DataFrame([data], columns=feature_names)


def load_timeseries(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def build_timeseries_figure(df, scenario_id):
    fig = go.Figure()
    base_layout = dict(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color="#1b1b1b"),
        colorway=COLORWAY,
        legend_title="variables",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    if df is None or df.empty:
        fig.update_layout(
            title=f"Scenario {scenario_id} - aucune donnee",
            xaxis_title="t_min",
            yaxis_title="valeur",
            **base_layout,
        )
        return fig

    t_values = None
    if "t_min" in df.columns:
        t_values = pd.to_numeric(df["t_min"], errors="coerce")

    traces = 0
    for col in df.columns:
        if col == "t_min":
            continue
        y_values = pd.to_numeric(df[col], errors="coerce")
        if y_values.isna().all():
            continue
        x = t_values if t_values is not None else np.arange(len(df))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_values,
                mode="lines",
                name=col,
            )
        )
        traces += 1

    if traces == 0:
        fig.update_layout(
            title=f"Scenario {scenario_id} - aucune serie exploitable",
            xaxis_title="t_min",
            yaxis_title="valeur",
            **base_layout,
        )
        return fig

    fig.update_layout(
        title=f"Scenario {scenario_id} - series temporelles (6h)",
        xaxis_title="t_min",
        yaxis_title="valeur",
        hovermode="x unified",
        **base_layout,
    )
    return fig


def format_surface(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.0f}".replace(",", " ")
    except Exception:
        return "N/A"


def format_threshold(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.0f}".replace(",", " ")
    except Exception:
        return "N/A"


def compute_niveau_total(feature_values):
    if feature_values is None:
        return None

    niveau_total = safe_float(feature_values.get("Niveau_Total_t0"))
    if niveau_total is not None:
        return niveau_total

    nm = safe_float(feature_values.get("NM_t0"))
    s = safe_float(feature_values.get("S_t0"))
    hs = safe_float(feature_values.get("Hs_t0"))
    if nm is None or s is None or hs is None:
        return None
    return nm + s + hs


def row_to_numeric_features(row_dict):
    if not row_dict:
        return {}
    features = {}
    for key, value in row_dict.items():
        if normalize_key(key) == "nomfichier":
            continue
        num = safe_float(value)
        if num is not None:
            features[key] = num

    if "Niveau_Total_t0" not in features:
        niveau_total = compute_niveau_total(features)
        if niveau_total is not None:
            features["Niveau_Total_t0"] = niveau_total

    return features


def predict_smax(model_bundle, feature_row):
    if model_bundle is None:
        return None, "modele non charge"
    if feature_row is None:
        return None, "donnees features manquantes"

    features = row_to_numeric_features(feature_row)
    if not features:
        return None, "donnees features manquantes"

    x_vec = build_feature_vector(model_bundle["feature_names"], features)
    try:
        pred = model_bundle["model"].predict(x_vec)[0]
        return pred, ""
    except Exception:
        return None, "erreur prediction"


def digue_status(niveau_total):
    if niveau_total is None:
        return "N/A"
    return "OUI" if niveau_total > H_DIGUE else "NON"


def significant_status(digue, pred_value, threshold):
    if digue == "OUI":
        return "OUI"
    if pred_value is not None and threshold is not None:
        return "OUI" if pred_value > threshold else "NON"
    if digue == "NON":
        return "NON"
    return "N/A"


def badge_class(status):
    if status == "OUI":
        return "badge badge-yes"
    if status == "NON":
        return "badge badge-no"
    return "badge badge-na"


def make_app(
    scenarios,
    features_map,
    y_lookup,
    model_bundle,
    model_status,
    threshold_default,
    threshold_max,
    threshold_marks,
    app_css,
):
    scenario_options = [
        {"label": f"Scenario {scenario_id}", "value": scenario_id}
        for scenario_id, _ in scenarios
    ]
    default_value = scenario_options[0]["value"] if scenario_options else None
    scenario_map = {scenario_id: path for scenario_id, path in scenarios}

    app = Dash(__name__)
    app.title = "DaMS4 - Inondations cotieres"
    app.index_string = (
        """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
"""
        + app_css
        + """
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""
    )

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("DaMS4 / Theme 1", className="eyebrow"),
                            html.H1(
                                "Inondations cotieres a Gavres",
                                className="hero-title",
                            ),
                            html.P(
                                "Explore les series temporelles 6h, la houle, le vent, la maree, "
                                "et la surcote. Compare un scenario reel et sa prediction Smax.",
                                className="hero-subtitle",
                            ),
                            html.Div(
                                [
                                    html.Span("Houle", className="chip"),
                                    html.Span("Vent", className="chip"),
                                    html.Span("Maree", className="chip"),
                                    html.Span("Surcote", className="chip"),
                                    html.Span("Directions", className="chip"),
                                ],
                                className="chip-row",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div("Selection du scenario", className="card-title"),
                            html.Label("Scenario", className="scenario-label"),
                            dcc.Dropdown(
                                id="scenario-dropdown",
                                options=scenario_options,
                                value=default_value,
                                clearable=False,
                                className="scenario-dropdown",
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        "Seuil de surface inondee (m2)",
                                        className="scenario-label",
                                    ),
                                    dcc.Slider(
                                        id="surface-threshold",
                                        min=0,
                                        max=threshold_max,
                                        step=1000,
                                        value=threshold_default,
                                        marks=threshold_marks,
                                        tooltip={"placement": "bottom"},
                                    ),
                                ],
                                className="slider-wrap",
                            ),
                            html.Div(
                                "Serie 6h superposees, colonnes disponibles incluses.",
                                className="card-hint",
                            ),
                        ],
                        className="control-card",
                    ),
                ],
                className="hero reveal",
                style={"animationDelay": "0.05s"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Smax reelle", className="smax-label"),
                            html.Div(
                                [
                                    html.Span("N/A", id="smax-real-value"),
                                    html.Span("m2", className="smax-unit"),
                                ],
                                className="smax-value",
                            ),
                            html.Div(
                                "source: Y_Smax.csv",
                                id="smax-real-note",
                                className="smax-note",
                            ),
                        ],
                        className="smax-card kpi-card reveal",
                        style={"animationDelay": "0.1s"},
                    ),
                    html.Div(
                        [
                            html.Div("Smax predite", className="smax-label"),
                            html.Div(
                                [
                                    html.Span("N/A", id="smax-pred-value"),
                                    html.Span("m2", className="smax-unit"),
                                ],
                                className="smax-value",
                            ),
                            html.Div(
                                "modele non charge",
                                id="smax-pred-note",
                                className="smax-note",
                            ),
                        ],
                        className="smax-card smax-card--pred kpi-card reveal",
                        style={"animationDelay": "0.14s"},
                    ),
                ],
                className="smax-grid",
            ),
            html.Div(
                [
                    html.H3("Decision et prediction", className="kpi-title"),
                    html.Div(id="kpi-digue", className="kpi-line"),
                    html.Div(id="kpi-signif", className="kpi-line"),
                    html.Div(
                        "Le modele predit Smax. Le depassement de digue est une regle physique. "
                        "L'inondation est une interpretation operationnelle.",
                        className="card-hint",
                    ),
                ],
                className="kpi-card reveal",
                style={"animationDelay": "0.12s"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Series temporelles", className="section-title"),
                            html.Div(
                                "superposition des variables disponibles",
                                className="section-subtitle",
                            ),
                        ],
                        className="section-head",
                    ),
                    dcc.Graph(
                        id="timeseries-graph",
                        config={"displayModeBar": False, "displaylogo": False},
                    ),
                ],
                className="graph-card reveal",
                style={"animationDelay": "0.2s"},
            ),
            html.Div(
                [
                    html.H3("A propos", className="section-title"),
                    html.Ul(
                        [
                            html.Li(
                                "Le modele predit Smax (surface maximale inondee).",
                                className="about-item",
                            ),
                            html.Li(
                                "Le depassement de digue est une regle physique.",
                                className="about-item",
                            ),
                            html.Li(
                                "L'inondation est une interpretation operationnelle.",
                                className="about-item",
                            ),
                        ],
                        className="about-list",
                    ),
                ],
                className="about-card reveal",
                style={"animationDelay": "0.28s"},
            ),
            html.Ul(
                [
                    html.Li("DaMS4 - Theme 1 : Inondations cotieres (BRGM)."),
                    html.Li("App interactive pour la lecture scenario par scenario."),
                ],
                className="footer-note",
            ),
        ],
        className="app-shell",
    )

    @app.callback(
        Output("timeseries-graph", "figure"),
        Output("smax-real-value", "children"),
        Output("smax-real-note", "children"),
        Output("smax-pred-value", "children"),
        Output("smax-pred-note", "children"),
        Output("kpi-digue", "children"),
        Output("kpi-signif", "children"),
        Input("scenario-dropdown", "value"),
        Input("surface-threshold", "value"),
    )
    def update_view(scenario_id, threshold):
        fig = build_timeseries_figure(None, scenario_id)
        threshold_value = threshold if threshold is not None else threshold_default

        smax_real_value = "N/A"
        smax_real_note = "source: Y_Smax.csv"
        smax_pred_value = "N/A"
        smax_pred_note = model_status or "modele non charge"

        digue_line = [
            html.Span("Depassement de la digue (3.6 m):", className="kpi-label-text"),
            html.Span("N/A", className="badge badge-na"),
        ]
        signif_line = [
            html.Span(
                f"Inondation significative (seuil {format_threshold(threshold_value)} m2):",
                className="kpi-label-text",
            ),
            html.Span("N/A", className="badge badge-na"),
        ]

        if scenario_id is None or scenario_id not in scenario_map:
            return (
                fig,
                smax_real_value,
                smax_real_note,
                smax_pred_value,
                smax_pred_note,
                digue_line,
                signif_line,
            )

        df = load_timeseries(scenario_map[scenario_id])
        fig = build_timeseries_figure(df, scenario_id)

        feature_row = features_map.get(scenario_id)
        niveau_total = compute_niveau_total(feature_row) if feature_row else None
        digue = digue_status(niveau_total)

        pred_value, pred_status = predict_smax(model_bundle, feature_row)
        real_value = y_lookup.get(scenario_id)

        digue_line = [
            html.Span("Depassement de la digue (3.6 m):", className="kpi-label-text"),
            html.Span(digue, className=badge_class(digue)),
        ]

        real_text = format_surface(real_value)
        smax_real_value = real_text
        smax_real_note = (
            "source: Y_Smax.csv" if real_value is not None else "source indisponible"
        )

        if pred_value is not None:
            smax_pred_value = format_surface(pred_value)
            smax_pred_note = "metamodele charge"
        else:
            smax_pred_value = "N/A"
            smax_pred_note = pred_status or model_status or "modele non charge"

        signif = significant_status(digue, pred_value, threshold_value)
        signif_line = [
            html.Span(
                f"Inondation significative (seuil {format_threshold(threshold_value)} m2):",
                className="kpi-label-text",
            ),
            html.Span(signif, className=badge_class(signif)),
        ]

        return (
            fig,
            smax_real_value,
            smax_real_note,
            smax_pred_value,
            smax_pred_note,
            digue_line,
            signif_line,
        )

    return app


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    scenarios_dir = resolve_path(base_dir, args.scenarios_dir)
    features_file = resolve_path(base_dir, args.features_file)
    y_file = resolve_path(base_dir, args.y_file)
    model_path = resolve_path(base_dir, args.model_path)

    scenarios = list_scenarios(scenarios_dir)

    features_df, features_status = load_features_table(features_file)
    features_map = index_features_table(features_df)

    y_lookup, y_status = load_y_lookup(y_file)

    model_bundle, model_status = load_model(model_path)

    if features_status:
        print(f"[INFO] {features_status}")
    if y_status:
        print(f"[INFO] {y_status}")
    if model_status:
        print(f"[INFO] {model_status}")

    app_css = build_app_css(base_dir)

    threshold_default = DEFAULT_THRESHOLD
    threshold_max = 200000
    if y_lookup:
        max_val = max(y_lookup.values())
        threshold_max = max(threshold_max, int(max_val * 1.2))
    threshold_marks = {
        0: "0",
        threshold_default: "10k",
        int(threshold_max / 2): "mid",
        threshold_max: "max",
    }

    app = make_app(
        scenarios,
        features_map,
        y_lookup,
        model_bundle,
        model_status,
        threshold_default,
        threshold_max,
        threshold_marks,
        app_css,
    )
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
