"""
app.py - Real Estate Investment Advisor (Streamlit)
---------------------------------------------------
Run from the project root:

    streamlit run app.py

Pages
    1. Overview             - market KPIs and how the advisor works
    2. Investment Predictor - enter a property -> Good Investment? + 5-year price
    3. Property Explorer    - filter listings by city, price, BHK, size ...
    4. Market Insights      - location map, heatmaps, trend charts
    5. Model Performance    - metrics, confusion matrix, feature importance, MLflow
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))
import config as cfg  # noqa: E402
from data_preprocessing import (  # noqa: E402
    engineer_features, expected_growth_rate, run_preprocessing,
)

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon=":material/domain:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Design tokens
# ----------------------------------------------------------------------------
INK = "#1B2A24"        # body text
FOREST = "#1F4D3A"     # primary
DEEP = "#13332A"       # sidebar / hero
BRASS = "#B8892E"      # accent (gold)
SAGE = "#7FA391"       # secondary series
MUTED = "#66766E"      # captions
GOOD = "#2E7D5B"
BAD = "#B4483B"
LINE = "#DCE3DD"
SERIES = [FOREST, BRASS, SAGE, "#4F6D8A", "#C9A66B", "#9DB5A8"]
SEQ = ["#EEF3EF", "#C7D9CE", "#8FB3A0", "#4F8670", "#1F4D3A", "#0F2A20"]

pio.templates["advisor"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=INK),
        title=dict(font=dict(family="Bricolage Grotesque, Inter, sans-serif", size=16, color=INK),
                   x=0, xanchor="left", y=0.97),
        colorway=SERIES,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#E8EDE9", linecolor=LINE, zeroline=False, ticks=""),
        yaxis=dict(gridcolor="#E8EDE9", linecolor=LINE, zeroline=False, ticks=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", title_text=""),
        margin=dict(t=60, b=48, l=16, r=20),
        hoverlabel=dict(bgcolor="white", bordercolor=LINE, font=dict(color=INK)),
        coloraxis=dict(colorscale=SEQ),
    )
)
pio.templates.default = "plotly_white+advisor"

st.html(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Inter:wght@400;500;600&display=swap');

    html, body, .stApp, .stMarkdown p, .stMarkdown li, button, input, textarea, select, label {
        font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
    }
    [data-testid="stIconMaterial"], .material-symbols-rounded {
        font-family: 'Material Symbols Rounded' !important; }
    h1, h2, h3, h4, .disp { font-family: 'Bricolage Grotesque', 'Inter', sans-serif !important;
        letter-spacing: -0.015em; color: #1B2A24; }
    .block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1320px; }
    [data-testid="stHeader"] { background: transparent; }

    /* ---------- sidebar ---------- */
    [data-testid="stSidebar"] { border-right: 1px solid #0E261F; }
    [data-testid="stSidebarNav"] a { border-radius: 8px; margin: 2px 0; }
    [data-testid="stSidebarNav"] a[aria-current="page"] { background: rgba(217,176,100,.16); }
    [data-testid="stSidebarNav"] a[aria-current="page"] span { color: #F1D9A6 !important; font-weight: 600; }
    .brand { display:flex; gap:12px; align-items:center; padding: 4px 4px 18px; }
    .brand-mark { width:38px; height:38px; border-radius:10px; background:#D9B064; color:#13332A;
        display:grid; place-items:center; font-family:'Bricolage Grotesque','Inter','Segoe UI',sans-serif; font-weight:700; font-size:18px; }
    .brand-name { font-family:'Bricolage Grotesque','Inter','Segoe UI',sans-serif; font-weight:700; font-size:17px; color:#F4F6F3; line-height:1.15; }
    .brand-sub { font-size:12px; color:#9DB5A8; }
    .side-facts { border-top:1px solid #2A5245; margin-top:12px; padding-top:14px; font-size:12.5px; color:#B9CCC1; line-height:1.9; }
    .side-facts b { color:#F4F6F3; font-weight:600; }

    /* ---------- page header ---------- */
    .page-head { margin: 0 0 1.4rem; }
    .page-head h1 { font-size: 2.1rem; font-weight: 700; margin: 0 0 .25rem; padding: 0; }
    .page-head p { color:#66766E; font-size: 1rem; margin:0; max-width: 70ch; }

    /* ---------- hero ---------- */
    .hero { background: #13332A; border-radius: 18px; padding: 44px 48px 40px; color:#E8EFEA;
        position: relative; overflow: hidden; margin-bottom: 22px; }
    .hero::after { content:""; position:absolute; right:-40px; top:-40px; width:380px; height:380px;
        background:
          repeating-linear-gradient(0deg, rgba(217,176,100,.14) 0 1px, transparent 1px 28px),
          repeating-linear-gradient(90deg, rgba(217,176,100,.14) 0 1px, transparent 1px 28px);
        mask-image: radial-gradient(circle at 70% 30%, black 0%, transparent 70%);
        -webkit-mask-image: radial-gradient(circle at 70% 30%, black 0%, transparent 70%); }
    .hero h1 { color:#F4F6F3 !important; font-size: 2.9rem; line-height:1.04; font-weight:700; margin:0 0 14px; max-width: 16ch; padding:0; }
    .hero p { font-size: 1.05rem; color:#B9CCC1; max-width: 58ch; margin:0; line-height:1.6; }
    .hero .q { display:flex; gap:28px; margin-top: 28px; flex-wrap: wrap; position:relative; z-index:1; }
    .hero .q div { border-left: 2px solid #D9B064; padding-left: 14px; max-width: 300px; }
    .hero .q b { display:block; color:#F1D9A6; font-family:'Bricolage Grotesque','Inter','Segoe UI',sans-serif; font-size:1.05rem; font-weight:600; margin-bottom:2px; }
    .hero .q span { font-size: .9rem; color:#B9CCC1; }

    /* ---------- KPI strip ---------- */
    .kpis { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        background:#fff; border:1px solid #DCE3DD; border-radius:14px; margin: 0 0 22px; overflow:hidden; }
    .kpi { padding: 18px 20px; border-right:1px solid #E8EDE9; }
    .kpi:last-child { border-right: none; }
    .kpi .v { font-family:'Bricolage Grotesque','Inter','Segoe UI',sans-serif; font-size: 1.65rem; font-weight:700; color:#1B2A24; line-height:1.1; }
    .kpi .l { font-size: .84rem; color:#66766E; margin-top:4px; }
    .kpi .d { font-size: .8rem; margin-top: 6px; font-weight:500; }
    .up { color:#2E7D5B; } .down { color:#B4483B; } .flat { color:#66766E; }

    /* ---------- section heading ---------- */
    .sec { margin: 26px 0 10px; }
    .sec h3 { font-size: 1.3rem; font-weight: 700; margin:0; padding:0; }
    .sec p { color:#66766E; margin: 3px 0 0; font-size:.93rem; }

    /* ---------- containers / widgets ---------- */
    [data-testid="stVerticalBlockBorderWrapper"] { background: #FFFFFF; }
    [data-testid="stForm"] { background:#fff; border:1px solid #DCE3DD; border-radius:14px; padding: 22px 24px; }
    .stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button {
        background:#1F4D3A; border:1px solid #1F4D3A; color:#fff; font-weight:600; padding: .55rem 1.4rem; }
    .stButton button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button:hover {
        background:#183D2E; border-color:#183D2E; color:#fff; }
    [data-testid="stDataFrame"] { border:1px solid #DCE3DD; border-radius: 12px; overflow:hidden; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #DCE3DD; }
    .stTabs [data-baseweb="tab"] { padding: 10px 16px; font-weight:500; }
    [data-testid="stPlotlyChart"] { background:#fff; border:1px solid #DCE3DD; border-radius:14px; overflow:hidden; }
    /* sidebar page links */
    [data-testid="stSidebar"] [data-testid="stPageLink"] a { border-radius: 8px; padding: 6px 10px; }
    [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover { background: rgba(255,255,255,.06); }
    [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"],
    [data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] { background: rgba(217,176,100,.16); }
    [data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] p { color:#F1D9A6 !important; font-weight:600; }
    .nav-label { font-size:12px; color:#7FA391; margin: 8px 4px 4px; }
    /* main-area page links as buttons */
    .stMain [data-testid="stPageLink"] a { border:1px solid #1F4D3A; border-radius: 10px; padding: 8px 14px; background:#fff; }
    .stMain [data-testid="stPageLink"] a:hover { background:#EEF3EF; }
    *:focus-visible { outline: 2px solid #B8892E !important; outline-offset: 2px; }

    /* ---------- valuation report (predictor result) ---------- */
    .report { background:#fff; border:1px solid #DCE3DD; border-radius:18px; overflow:hidden; margin: 8px 0 18px; }
    .report-top { display:grid; grid-template-columns: 1.1fr 1fr; }
    .verdict-panel { padding: 28px 30px; color:#fff; }
    .verdict-panel.good { background: #1F4D3A; }
    .verdict-panel.bad  { background: #7A3128; }
    .verdict-panel .tag { font-size:.85rem; opacity:.8; }
    .verdict-panel h2 { color:#fff !important; font-size:2.2rem; margin:4px 0 10px; padding:0; line-height:1.05; }
    .verdict-panel .conf { font-size: .95rem; opacity:.92; }
    .bar { height:8px; background: rgba(255,255,255,.18); border-radius:8px; margin-top: 14px; overflow:hidden; }
    .bar i { display:block; height:100%; background:#D9B064; border-radius:8px; }
    .price-panel { padding: 28px 30px; border-left: 1px solid #E8EDE9; }
    .price-panel .tag { font-size:.85rem; color:#66766E; }
    .price-panel .big { font-family:'Bricolage Grotesque','Inter','Segoe UI',sans-serif; font-size:2.6rem; font-weight:700; color:#1B2A24; line-height:1.05; margin: 4px 0 8px; }
    .price-panel .from { color:#66766E; font-size:.95rem; }
    .price-panel .from b { color:#2E7D5B; }
    .report-facts { display:grid; grid-template-columns: repeat(4, 1fr); border-top:1px solid #E8EDE9; }
    .report-facts > div { padding: 16px 22px; border-right: 1px solid #E8EDE9; }
    .report-facts > div:last-child { border-right: none; }
    .report-facts .v { font-family:'Bricolage Grotesque','Inter','Segoe UI',sans-serif; font-weight:700; font-size:1.25rem; }
    .report-facts .l { font-size:.82rem; color:#66766E; }

    .checks { display:grid; grid-template-columns: 1fr 1fr; gap: 8px 14px; }
    .chk { display:flex; gap:10px; align-items:flex-start; padding: 10px 14px; border-radius: 10px;
        background:#fff; border:1px solid #E1E8E3; font-size:.93rem; }
    .chk .dot { flex: 0 0 20px; height:20px; border-radius:50%; display:grid; place-items:center;
        font-size:12px; font-weight:700; color:#fff; margin-top:1px; }
    .chk.ok .dot { background:#2E7D5B; } .chk.no .dot { background:#C9CFCB; color:#55625B; }
    .chk.no { color:#66766E; }

    .note { color:#66766E; font-size:.84rem; margin-top: 18px; }
    .empty { background:#fff; border:1px dashed #C7D3CB; border-radius:14px; padding: 36px; text-align:center; color:#66766E; }
    .empty b { display:block; color:#1B2A24; font-family:'Bricolage Grotesque','Inter','Segoe UI',sans-serif; font-size:1.15rem; margin-bottom: 4px; }

    @media (max-width: 900px) {
        .hero { padding: 30px 24px; } .hero h1 { font-size: 2.1rem; }
        .report-top { grid-template-columns: 1fr; } .price-panel { border-left:none; border-top:1px solid #E8EDE9; }
        .report-facts { grid-template-columns: 1fr 1fr; } .checks { grid-template-columns: 1fr; }
    }
    @media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
    </style>
    """
)


# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading housing data…")
def load_data() -> pd.DataFrame:
    if not cfg.PROCESSED_DATA_PATH.exists():
        run_preprocessing()  # first run: build the cleaned dataset automatically
    return pd.read_csv(cfg.PROCESSED_DATA_PATH)


@st.cache_data
def load_city_stats() -> pd.DataFrame:
    return pd.read_csv(cfg.CITY_STATS_PATH)


@st.cache_resource(show_spinner="Loading ML models…")
def load_models():
    try:
        clf = joblib.load(cfg.CLASSIFIER_PATH)
        reg = joblib.load(cfg.REGRESSOR_PATH)
        meta = json.loads(cfg.MODEL_METADATA_PATH.read_text())
        metrics = json.loads(cfg.METRICS_PATH.read_text()) if cfg.METRICS_PATH.exists() else {}
        return clf, reg, meta, metrics, None
    except Exception as exc:  # missing files or library-version mismatch
        return None, None, None, None, str(exc)


@st.cache_data
def city_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg = (df.groupby(["State", "City", "City_Tier"])
           .agg(Listings=("ID", "count"),
                Avg_Price=("Price_in_Lakhs", "mean"),
                Median_PPSF=("Price_per_SqFt_Rs", "median"),
                Good_Share=(cfg.CLASSIFICATION_TARGET, "mean"),
                Avg_Growth=("Expected_Growth_Rate", "mean"))
           .reset_index())
    agg["lat"] = agg["City"].map(lambda c: cfg.CITY_COORDS.get(c, (np.nan, np.nan))[0])
    agg["lon"] = agg["City"].map(lambda c: cfg.CITY_COORDS.get(c, (np.nan, np.nan))[1])
    return agg


df = load_data()
city_stats = load_city_stats()
clf, reg, meta, all_metrics, model_error = load_models()


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def fmt_lakhs(x: float) -> str:
    """₹ in lakhs, switching to crores at 100 L."""
    return f"₹{x / 100:,.2f} Cr" if x >= 100 else f"₹{x:,.1f} L"


def page_head(title: str, sub: str) -> None:
    st.html(f"<div class='page-head'><h1>{title}</h1><p>{sub}</p></div>")


def section(title: str, sub: str = "") -> None:
    st.html(f"<div class='sec'><h3>{title}</h3>{f'<p>{sub}</p>' if sub else ''}</div>")


def kpi_strip(items) -> None:
    """items: list of (value, label, delta_text or None, delta_class)."""
    cells = ""
    for it in items:
        v, l = it[0], it[1]
        d = f"<div class='d {it[3] if len(it) > 3 else 'flat'}'>{it[2]}</div>" if len(it) > 2 and it[2] else ""
        cells += f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{l}</div>{d}</div>"
    st.html(f"<div class='kpis'>{cells}</div>")


def chart(fig, height: int = 380) -> None:
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "displayModeBar": False})


def models_ready() -> bool:
    if model_error:
        st.error(
            "The trained models could not be loaded. From the project folder run "
            "`python src/train_models.py`, then refresh this page.\n\n"
            f"Details: {model_error}", icon=":material/error:")
        return False
    return True


# ============================================================================
# PAGE 1 - OVERVIEW
# ============================================================================
def page_overview():
    st.html(
        f"""
        <div class="hero">
          <h1>Know if a property is worth buying.</h1>
          <p>Machine-learning models trained on {len(df):,} Indian housing listings across
          {df['City'].nunique()} cities score any property against its local market and
          project its value five years out.</p>
          <div class="q">
            <div><b>Is it a good investment?</b><span>XGBoost classifier with a confidence score</span></div>
            <div><b>What will it be worth in 2030?</b><span>XGBoost regressor, R² {meta['regressor']['metrics']['r2'] if meta else '—'}</span></div>
          </div>
        </div>
        """
    )
    c1, c2, _ = st.columns([1.6, 1.4, 4])
    c1.page_link(PAGES["predict"], label="Analyse a property", icon=":material/insights:")
    c2.page_link(PAGES["explore"], label="Browse listings", icon=":material/search:")

    kpi_strip([
        (f"{len(df):,}", "Listings analysed"),
        (fmt_lakhs(df["Price_in_Lakhs"].median()), "Median asking price"),
        (f"₹{df['Price_per_SqFt_Rs'].median():,.0f}", "Median price per sq ft"),
        (f"{df[cfg.CLASSIFICATION_TARGET].mean():.1%}", "Rated good investments"),
        (f"{df['Expected_Growth_Rate'].mean():.2%}", "Average expected growth / yr"),
    ])

    section("How the advisor decides", "Both models see the same 29 engineered features for every property.")
    a, b = st.columns(2, gap="medium")
    with a, st.container(border=True):
        st.markdown("#### :material/verified: Good investment or not")
        st.markdown(
            "Each listing gets a 0–10 investment score from ten factors: price per sq ft and "
            "total price against the city median, family-size BHK, ready-to-move status, "
            "transport access, infrastructure, amenities, security, parking and expected "
            "appreciation. A score of **5.5 or more** is a good investment. The classifier "
            "learns this pattern and returns a probability you can read as confidence."
        )
    with b, st.container(border=True):
        st.markdown("#### :material/trending_up: Price after five years")
        st.markdown(
            f"`Future = Price × (1 + r)⁵`. The yearly rate *r* starts at "
            f"**{cfg.BASE_GROWTH_RATE:.0%}** and moves up or down with city tier, transport, "
            "infrastructure, construction stage, property type, age and amenities, plus a "
            "small market-noise term. The regressor predicts the result directly from the "
            "property's features."
        )

    section("Market snapshot")
    l, r = st.columns(2, gap="medium")
    with l:
        tier = (df.groupby("City_Tier")[cfg.CLASSIFICATION_TARGET].mean()).reset_index()
        fig = px.bar(tier, x="City_Tier", y=cfg.CLASSIFICATION_TARGET, text_auto=".1%",
                     color_discrete_sequence=[FOREST],
                     labels={cfg.CLASSIFICATION_TARGET: "Good-investment share", "City_Tier": ""},
                     title="Good-investment share by city tier")
        fig.update_yaxes(tickformat=".0%")
        fig.update_traces(textposition="outside", marker_cornerradius=6, width=0.55)
        chart(fig, 360)
    with r:
        fig = px.histogram(df, x="Investment_Score", nbins=21, color_discrete_sequence=[SAGE],
                           labels={"Investment_Score": "Investment score (0–10)"},
                           title="How listings score")
        fig.add_vline(x=5.5, line_dash="dash", line_color=BRASS,
                      annotation_text="Good from 5.5", annotation_font_color=BRASS)
        fig.update_layout(bargap=0.08, yaxis_title="Listings")
        chart(fig, 360)


# ============================================================================
# PAGE 2 - INVESTMENT PREDICTOR
# ============================================================================
def page_predict():
    page_head("Investment predictor",
              "Describe a property to get an investment verdict, the model's confidence "
              "and a five-year price forecast.")
    if not models_ready():
        return

    with st.container(border=True):
        st.markdown("**Location**")
        l1, l2, l3 = st.columns(3)
        states = sorted(df["State"].unique())
        state = l1.selectbox("State", states, index=states.index("Tamil Nadu") if "Tamil Nadu" in states else 0)
        city = l2.selectbox("City", sorted(df.loc[df["State"] == state, "City"].unique()))
        locality = l3.selectbox(
            "Locality", sorted(df.loc[df["City"] == city, "Locality"].unique(),
                               key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 0))

    with st.form("predict_form", border=False):
        st.markdown("**The property**")
        c1, c2, c3, c4 = st.columns(4)
        prop_type = c1.selectbox("Property type", ["Apartment", "Independent House", "Villa"])
        bhk = c2.selectbox("BHK", [1, 2, 3, 4, 5], index=2)
        size = c3.number_input("Size (sq ft)", 300, 10000, 1500, step=50)
        price = c4.number_input("Asking price (₹ lakhs)", 5.0, 1000.0, 120.0, step=5.0)

        c1, c2, c3, c4 = st.columns(4)
        year_built = c1.number_input("Year built", 1950, cfg.REFERENCE_YEAR, 2015)
        furnished = c2.selectbox("Furnishing", ["Unfurnished", "Semi-furnished", "Furnished"])
        floor_no = c3.number_input("Floor", 0, 100, 5)
        total_floors = c4.number_input("Total floors in building", 1, 100, 12)

        st.markdown("**Neighbourhood and features**")
        c1, c2, c3, c4 = st.columns(4)
        schools = c1.slider("Schools nearby", 1, 10, 6)
        hospitals = c2.slider("Hospitals nearby", 1, 10, 5)
        transport = c3.selectbox("Public transport access", ["High", "Medium", "Low"])
        facing = c4.selectbox("Facing", ["North", "East", "South", "West"])

        c1, c2, c3, c4 = st.columns(4)
        parking = c1.segmented_control("Parking", ["Yes", "No"], default="Yes")
        security = c2.segmented_control("Security", ["Yes", "No"], default="Yes")
        owner = c3.selectbox("Listed by", ["Owner", "Builder", "Broker"])
        availability = c4.selectbox("Availability", ["Ready_to_Move", "Under_Construction"],
                                    format_func=lambda s: s.replace("_", " "))
        amenities = st.pills("Amenities", cfg.AMENITY_LIST, selection_mode="multi",
                             default=["Gym", "Garden", "Playground"])

        submitted = st.form_submit_button("Analyse property", icon=":material/analytics:")

    if not submitted:
        st.html("<div class='empty'><b>Your valuation report appears here</b>"
                "Fill in the details above and select Analyse property.</div>")
        return

    parking = parking or "No"
    security = security or "No"
    amenities = amenities or []
    if floor_no > total_floors:
        st.warning("The floor is higher than the building's total floors. Check those two values.",
                   icon=":material/warning:")

    raw = pd.DataFrame([{
        "ID": 0, "State": state, "City": city, "Locality": locality,
        "Property_Type": prop_type, "BHK": bhk, "Size_in_SqFt": size,
        "Price_in_Lakhs": price, "Year_Built": year_built,
        "Furnished_Status": furnished, "Floor_No": floor_no, "Total_Floors": total_floors,
        "Age_of_Property": cfg.REFERENCE_YEAR - year_built,
        "Nearby_Schools": schools, "Nearby_Hospitals": hospitals,
        "Public_Transport_Accessibility": transport, "Parking_Space": parking,
        "Security": security, "Amenities": ", ".join(amenities), "Facing": facing,
        "Owner_Type": owner, "Availability_Status": availability,
        "Price_per_SqFt_Rs": price * 1e5 / size,
    }])
    row = engineer_features(raw, city_stats)
    X = row[cfg.MODEL_FEATURES]

    proba = float(clf.predict_proba(X)[0, 1])
    is_good = proba >= 0.5
    confidence = proba if is_good else 1 - proba
    future = float(reg.predict(X)[0])
    cagr = (future / price) ** (1 / cfg.FORECAST_YEARS) - 1
    ppsf = price * 1e5 / size
    ppsf_gap = row["PPSF_vs_City_Median"].iloc[0] - 1
    ref = city_stats.set_index("City").loc[city]

    st.html(f"""
    <div class="report">
      <div class="report-top">
        <div class="verdict-panel {'good' if is_good else 'bad'}">
          <div class="tag">{prop_type} · {bhk} BHK · {locality.replace('_', ' ')}, {city}</div>
          <h2>{'Good investment' if is_good else 'Not a good investment'}</h2>
          <div class="conf">Model confidence <b>{'over 99.9%' if confidence > 0.999 else f'{confidence:.1%}'}</b></div>
          <div class="bar"><i style="width:{confidence * 100:.1f}%"></i></div>
        </div>
        <div class="price-panel">
          <div class="tag">Estimated value in {cfg.REFERENCE_YEAR + cfg.FORECAST_YEARS}</div>
          <div class="big">{fmt_lakhs(future)}</div>
          <div class="from">From {fmt_lakhs(price)} today. Gain of <b>{fmt_lakhs(future - price)}</b>,
          or <b>{cagr:.2%}</b> a year.</div>
        </div>
      </div>
      <div class="report-facts">
        <div><div class="v">₹{ppsf:,.0f}</div>
             <div class="l">Per sq ft, <span class="{'up' if ppsf_gap <= 0 else 'down'}">{abs(ppsf_gap):.1%} {'below' if ppsf_gap <= 0 else 'above'}</span> the {city} median</div></div>
        <div><div class="v">{fmt_lakhs(ref['City_Median_Price'])}</div><div class="l">{city} median price</div></div>
        <div><div class="v">{row['Infrastructure_Score'].iloc[0]:.1f} / 10</div><div class="l">Infrastructure score</div></div>
        <div><div class="v">{row['City_Tier'].iloc[0]}</div><div class="l">City tier</div></div>
      </div>
    </div>
    """)

    g1, g2 = st.columns([3, 2], gap="medium")
    with g1:
        years = np.arange(0, cfg.FORECAST_YEARS + 1)
        proj = pd.DataFrame({
            "Year": cfg.REFERENCE_YEAR + years,
            "Model forecast": price * (1 + cagr) ** years,
            f"Flat {cfg.BASE_GROWTH_RATE:.0%} a year": price * (1 + cfg.BASE_GROWTH_RATE) ** years,
        })
        fig = px.line(proj, x="Year", y=proj.columns[1:], markers=True,
                      labels={"value": "Price (₹ lakhs)", "variable": ""},
                      title="Five-year price path", color_discrete_sequence=[FOREST, "#AEB9B2"])
        fig.update_traces(line_width=3, selector=dict(name="Model forecast"))
        fig.update_traces(line_dash="dot", selector=dict(name=f"Flat {cfg.BASE_GROWTH_RATE:.0%} a year"))
        chart(fig, 340)
    with g2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=proba * 100,
            number={"suffix": "%", "font": {"family": "Bricolage Grotesque, Inter, sans-serif", "size": 44}},
            title={"text": "Probability of a good investment", "font": {"size": 14}},
            gauge={"axis": {"range": [0, 100], "tickcolor": LINE},
                   "bar": {"color": FOREST if is_good else BAD, "thickness": 0.28},
                   "bgcolor": "#F1F4F2", "borderwidth": 0,
                   "steps": [{"range": [0, 50], "color": "#F6E6E3"},
                             {"range": [50, 100], "color": "#E3EFE8"}],
                   "threshold": {"line": {"color": BRASS, "width": 3}, "value": 50}}))
        fig.update_layout(margin=dict(t=60, b=10, l=30, r=30))
        chart(fig, 340)

    section("What drove the verdict", "The ten factors behind the investment score, checked for this property.")
    base_r = float(expected_growth_rate(row).iloc[0])
    checks = [
        ("Price per sq ft at or below the city median", row["PPSF_vs_City_Median"].iloc[0] <= 1),
        ("Total price at or below the city median", row["Price_vs_City_Median"].iloc[0] <= 1),
        ("3 BHK or larger, where family demand is strongest", bhk >= 3),
        ("Ready to move, so no construction risk", availability == "Ready_to_Move"),
        ("High public transport access", transport == "High"),
        ("Infrastructure score of 5 or more", row["Infrastructure_Score"].iloc[0] >= 5),
        ("At least 3 amenities", len(amenities) >= 3),
        ("Security available", security == "Yes"),
        ("Parking available", parking == "Yes"),
        (f"Expected growth of {cfg.BASE_GROWTH_RATE:.0%}+ a year (this one: {base_r:.2%})",
         base_r >= cfg.BASE_GROWTH_RATE),
    ]
    items = "".join(
        f"<div class='chk {'ok' if ok else 'no'}'><span class='dot'>{'✓' if ok else '–'}</span>{label}</div>"
        for label, ok in checks)
    met = sum(ok for _, ok in checks)
    st.html(f"<p style='margin:0 0 10px;color:{MUTED}'>{met} of 10 factors met</p><div class='checks'>{items}</div>")

    section("What the classifier weighs most")
    fi = pd.Series(meta["classifier"]["feature_importance"]).head(10)[::-1]
    fig = px.bar(x=fi.values, y=[i.replace("_", " ") for i in fi.index], orientation="h",
                 labels={"x": "Relative importance", "y": ""}, color_discrete_sequence=[FOREST])
    fig.update_traces(marker_cornerradius=4)
    fig.update_layout(margin=dict(t=20))
    chart(fig, 380)
    st.html("<p class='note'>Forecasts are model estimates built on rule-based targets for this "
            "dataset. They are not financial advice.</p>")


# ============================================================================
# PAGE 3 - PROPERTY EXPLORER
# ============================================================================
def page_explore():
    page_head("Property explorer", "Filter the 250,000 listings by location, budget and size, "
                                   "then download what you find.")

    with st.container(border=True):
        f1, f2, f3 = st.columns(3)
        states = f1.multiselect("State", sorted(df["State"].unique()), placeholder="All states")
        city_pool = df[df["State"].isin(states)] if states else df
        cities = f2.multiselect("City", sorted(city_pool["City"].unique()), placeholder="All cities")
        ptypes = f3.multiselect("Property type", sorted(df["Property_Type"].unique()), placeholder="All types")

        f1, f2, f3 = st.columns(3)
        bhk_rng = f1.slider("BHK", 1, 5, (1, 5))
        price_rng = f2.slider("Price (₹ lakhs)", 10, 500, (10, 500), step=10)
        size_rng = f3.slider("Size (sq ft)", 500, 5000, (500, 5000), step=100)

        f1, f2, f3 = st.columns(3)
        avail = f1.multiselect("Availability", sorted(df["Availability_Status"].unique()),
                               format_func=lambda s: s.replace("_", " "), placeholder="Any")
        transport = f2.multiselect("Public transport", ["High", "Medium", "Low"], placeholder="Any")
        f3.write("")
        only_good = f3.toggle("Only good investments")

    q = df[df["BHK"].between(*bhk_rng)
           & df["Price_in_Lakhs"].between(*price_rng)
           & df["Size_in_SqFt"].between(*size_rng)]
    if states: q = q[q["State"].isin(states)]
    if cities: q = q[q["City"].isin(cities)]
    if ptypes: q = q[q["Property_Type"].isin(ptypes)]
    if avail: q = q[q["Availability_Status"].isin(avail)]
    if transport: q = q[q["Public_Transport_Accessibility"].isin(transport)]
    if only_good: q = q[q[cfg.CLASSIFICATION_TARGET] == 1]

    if q.empty:
        st.html("<div class='empty'><b>No listings match these filters</b>"
                "Widen the price or size range, or clear a filter.</div>")
        return

    kpi_strip([
        (f"{len(q):,}", "Matching listings"),
        (fmt_lakhs(q["Price_in_Lakhs"].median()), "Median price"),
        (f"{q[cfg.CLASSIFICATION_TARGET].mean():.1%}", "Good-investment share"),
        (fmt_lakhs(q[cfg.REGRESSION_TARGET].median()), "Median value in 5 years"),
    ])

    cols = ["ID", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt",
            "Price_in_Lakhs", "Price_per_SqFt_Rs", "Public_Transport_Accessibility",
            "Availability_Status", "Investment_Score", cfg.CLASSIFICATION_TARGET, cfg.REGRESSION_TARGET]
    view = q[cols].sort_values("Investment_Score", ascending=False).copy()
    view[cfg.CLASSIFICATION_TARGET] = view[cfg.CLASSIFICATION_TARGET].map({1: "Good", 0: "Not good"})
    view["Availability_Status"] = view["Availability_Status"].str.replace("_", " ")

    section("Top listings", "Sorted by investment score. Showing the first 1,000.")
    st.dataframe(
        view.head(1000), hide_index=True, height=440, width="stretch",
        column_config={
            "ID": st.column_config.NumberColumn("ID", format="%d", width="small"),
            "Locality": st.column_config.TextColumn("Locality"),
            "Property_Type": "Type",
            "Size_in_SqFt": st.column_config.NumberColumn("Size (sq ft)", format="%,d"),
            "Price_in_Lakhs": st.column_config.NumberColumn("Price (₹ L)", format="%.1f"),
            "Price_per_SqFt_Rs": st.column_config.NumberColumn("₹ / sq ft", format="%,.0f"),
            "Public_Transport_Accessibility": "Transport",
            "Availability_Status": "Availability",
            "Investment_Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=10, format="%.1f"),
            cfg.CLASSIFICATION_TARGET: "Verdict",
            cfg.REGRESSION_TARGET: st.column_config.NumberColumn("Value in 5 yrs (₹ L)", format="%.1f"),
        },
    )
    st.download_button("Download filtered listings (CSV)", view.to_csv(index=False).encode("utf-8"),
                       "filtered_properties.csv", "text/csv", icon=":material/download:")

    a, b = st.columns(2, gap="medium")
    with a:
        fig = px.histogram(q, x="Price_in_Lakhs", nbins=40, color_discrete_sequence=[SAGE],
                           labels={"Price_in_Lakhs": "Price (₹ lakhs)"},
                           title="Price spread of these listings")
        fig.update_layout(bargap=0.06, yaxis_title="Listings")
        chart(fig, 360)
    with b:
        agg = (q.groupby("City").agg(Listings=("ID", "count"), Good=(cfg.CLASSIFICATION_TARGET, "mean"))
               .reset_index().sort_values("Good", ascending=True).tail(12))
        fig = px.bar(agg, x="Good", y="City", orientation="h", color_discrete_sequence=[FOREST],
                     labels={"Good": "Good-investment share", "City": ""},
                     title="Cities with the highest good-investment share")
        fig.update_xaxes(tickformat=".0%")
        fig.update_traces(marker_cornerradius=4)
        chart(fig, 360)


# ============================================================================
# PAGE 4 - MARKET INSIGHTS
# ============================================================================
def page_market():
    page_head("Market insights", "Where prices, growth and good investments sit across India.")
    city_agg = city_summary(df)

    labels = {"Good_Share": "Good-investment share", "Avg_Growth": "Expected growth",
              "Median_PPSF": "Median ₹ / sq ft", "Avg_Price": "Average price"}
    metric_choice = st.segmented_control("Colour the map by", list(labels), default="Good_Share",
                                         format_func=labels.get) or "Good_Share"
    fig = px.scatter_geo(city_agg.dropna(subset=["lat"]), lat="lat", lon="lon",
                         color=metric_choice, size="Listings", size_max=22, hover_name="City",
                         hover_data={"State": True, "Avg_Price": ":.1f", "Median_PPSF": ":,.0f",
                                     "Good_Share": ":.1%", "Avg_Growth": ":.2%",
                                     "lat": False, "lon": False, "Listings": ":,"},
                         labels=labels, color_continuous_scale=SEQ[1:])
    fig.update_geos(fitbounds="locations", showcountries=True, countrycolor="#B9C4BD",
                    showland=True, landcolor="#EEF2EF", showocean=True, oceancolor="#F7F9F8",
                    showframe=False, projection_type="mercator")
    fig.update_layout(margin=dict(t=10, b=0, l=0, r=0),
                      coloraxis_colorbar=dict(title="", thickness=12, len=0.7))
    chart(fig, 580)

    tab1, tab2, tab3 = st.tabs(["Heatmaps", "Growth and trends", "City table"])
    with tab1:
        a, b = st.columns([3, 2], gap="medium")
        with a:
            pivot = df.pivot_table(index="State", columns="Property_Type",
                                   values="Price_per_SqFt_Rs", aggfunc="median")
            fig = px.imshow(pivot, text_auto=",.0f", aspect="auto", color_continuous_scale=SEQ,
                            labels=dict(color="₹/sq ft", x="", y=""),
                            title="Median ₹ / sq ft by state and property type")
            fig.update_layout(coloraxis_showscale=False)
            chart(fig, 640)
        with b:
            pivot = df.pivot_table(index="City_Tier", columns="Public_Transport_Accessibility",
                                   values=cfg.CLASSIFICATION_TARGET, aggfunc="mean")[["Low", "Medium", "High"]]
            fig = px.imshow(pivot, text_auto=".1%", aspect="auto", color_continuous_scale=SEQ,
                            labels=dict(x="Transport access", y="", color="Share"),
                            title="Good-investment share by tier and transport")
            fig.update_layout(coloraxis_showscale=False)
            chart(fig, 310)
            grow = (df.groupby(["City_Tier", "Availability_Status"])["Expected_Growth_Rate"]
                    .mean().reset_index())
            grow["Availability_Status"] = grow["Availability_Status"].str.replace("_", " ")
            fig = px.bar(grow, x="City_Tier", y="Expected_Growth_Rate", color="Availability_Status",
                         barmode="group", title="Expected yearly growth by tier",
                         labels={"Expected_Growth_Rate": "", "City_Tier": ""},
                         color_discrete_sequence=[FOREST, BRASS])
            fig.update_yaxes(tickformat=".1%")
            fig.update_traces(marker_cornerradius=4)
            chart(fig, 310)
    with tab2:
        a, b = st.columns(2, gap="medium")
        with a:
            trend = df.groupby("Year_Built")[["Price_in_Lakhs", cfg.REGRESSION_TARGET]].mean().reset_index()
            trend.columns = ["Year built", "Current price", "Price in 5 years"]
            fig = px.line(trend, x="Year built", y=["Current price", "Price in 5 years"], markers=True,
                          labels={"value": "Average (₹ lakhs)", "variable": ""},
                          title="Current vs five-year price by construction year",
                          color_discrete_sequence=[SAGE, FOREST])
            chart(fig, 400)
        with b:
            top = city_agg.sort_values("Avg_Growth", ascending=True).tail(15)
            fig = px.bar(top, x="Avg_Growth", y="City", orientation="h",
                         color="City_Tier", color_discrete_map={"Tier 1": FOREST, "Tier 2": BRASS, "Tier 3": SAGE},
                         labels={"Avg_Growth": "Expected growth / yr", "City": "", "City_Tier": ""},
                         title="Fastest expected growth, top 15 cities")
            fig.update_xaxes(tickformat=".1%", range=[top["Avg_Growth"].min() * 0.97, top["Avg_Growth"].max() * 1.005])
            fig.update_traces(marker_cornerradius=4)
            chart(fig, 400)
    with tab3:
        st.dataframe(
            city_agg.drop(columns=["lat", "lon"]).sort_values("Good_Share", ascending=False),
            hide_index=True, width="stretch", height=480,
            column_config={
                "City_Tier": "Tier",
                "Listings": st.column_config.NumberColumn(format="%,d"),
                "Avg_Price": st.column_config.NumberColumn("Avg price (₹ L)", format="%.1f"),
                "Median_PPSF": st.column_config.NumberColumn("Median ₹ / sq ft", format="%,.0f"),
                "Good_Share": st.column_config.ProgressColumn("Good-investment share", min_value=0,
                                                              max_value=1, format="percent"),
                "Avg_Growth": st.column_config.NumberColumn("Growth / yr", format="percent"),
            })


# ============================================================================
# PAGE 5 - MODEL PERFORMANCE
# ============================================================================
def page_models():
    page_head("Model performance",
              "Three algorithms were trained for each task and compared on the same held-out test set.")
    if not models_ready():
        return

    cm_ = meta["classifier"]["metrics"]
    rm_ = meta["regressor"]["metrics"]
    kpi_strip([
        (f"{cm_['accuracy']:.1%}", f"Accuracy, {meta['classifier']['model']}"),
        (f"{cm_['f1_score']:.3f}", "F1 score"),
        (f"{cm_['roc_auc']:.4f}", "ROC-AUC"),
        (f"{rm_['r2']:.4f}", f"R², {meta['regressor']['model']}"),
        (f"₹{rm_['rmse']:.1f} L", "RMSE on 5-year price"),
        (f"{meta['test_rows']:,}", "Test listings"),
    ])

    tab1, tab2, tab3 = st.tabs(["Classification", "Regression", "MLflow tracking"])
    with tab1:
        cls = pd.DataFrame(all_metrics.get("classification", {})).T.reset_index(names="Model")
        cls["Champion"] = np.where(cls["Model"] == meta["classifier"]["model"], "Champion", "")
        st.dataframe(cls, hide_index=True, width="stretch", column_config={
            "accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0.8, max_value=1, format="%.4f"),
            "precision": st.column_config.NumberColumn("Precision", format="%.4f"),
            "recall": st.column_config.NumberColumn("Recall", format="%.4f"),
            "f1_score": st.column_config.ProgressColumn("F1", min_value=0.8, max_value=1, format="%.4f"),
            "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.4f"),
            "train_time_sec": st.column_config.NumberColumn("Train time (s)", format="%.1f"),
            "Champion": "",
        })
        a, b = st.columns(2, gap="medium")
        with a:
            cm = np.array(meta["classifier"]["confusion_matrix"])
            fig = px.imshow(cm, text_auto=",d", color_continuous_scale=SEQ,
                            x=["Predicted not good", "Predicted good"], y=["Actually not good", "Actually good"],
                            title=f"Confusion matrix, {meta['classifier']['model']}")
            fig.update_layout(coloraxis_showscale=False)
            fig.update_traces(textfont_size=18)
            chart(fig, 400)
        with b:
            fi = pd.Series(meta["classifier"]["feature_importance"]).head(12)[::-1]
            fig = px.bar(x=fi.values, y=[i.replace("_", " ") for i in fi.index], orientation="h",
                         color_discrete_sequence=[FOREST], labels={"x": "Importance", "y": ""},
                         title="Feature importance, classifier")
            fig.update_traces(marker_cornerradius=4)
            chart(fig, 400)
    with tab2:
        regm = pd.DataFrame(all_metrics.get("regression", {})).T.reset_index(names="Model")
        regm["Champion"] = np.where(regm["Model"] == meta["regressor"]["model"], "Champion", "")
        st.dataframe(regm, hide_index=True, width="stretch", column_config={
            "rmse": st.column_config.NumberColumn("RMSE (₹ L)", format="%.2f"),
            "mae": st.column_config.NumberColumn("MAE (₹ L)", format="%.2f"),
            "r2": st.column_config.ProgressColumn("R²", min_value=0.99, max_value=1, format="%.4f"),
            "train_time_sec": st.column_config.NumberColumn("Train time (s)", format="%.1f"),
            "Champion": "",
        })
        a, b = st.columns(2, gap="medium")
        with a:
            fi = pd.Series(meta["regressor"]["feature_importance"]).head(10)[::-1]
            fig = px.bar(x=fi.values, y=[i.replace("_", " ") for i in fi.index], orientation="h",
                         color_discrete_sequence=[BRASS], labels={"x": "Importance", "y": ""},
                         title="Feature importance, regressor")
            fig.update_traces(marker_cornerradius=4)
            chart(fig, 400)
        with b:
            sample = df.sample(3000, random_state=7)
            pred = reg.predict(sample[cfg.MODEL_FEATURES])
            fig = px.scatter(x=sample[cfg.REGRESSION_TARGET], y=pred, opacity=0.35,
                             labels={"x": "Actual 5-year price (₹ L)", "y": "Predicted (₹ L)"},
                             title="Actual vs predicted, 3,000-listing sample",
                             color_discrete_sequence=[FOREST])
            lim = [float(sample[cfg.REGRESSION_TARGET].min()), float(sample[cfg.REGRESSION_TARGET].max())]
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=BRASS, dash="dash"),
                                     name="Perfect prediction"))
            fig.update_traces(marker_size=4, selector=dict(mode="markers"))
            chart(fig, 400)
    with tab3:
        a, b = st.columns(2, gap="medium")
        with a, st.container(border=True):
            st.markdown("#### :material/science: Experiment")
            st.markdown(
                f"**{cfg.MLFLOW_EXPERIMENT}** holds six runs, three classifiers and three regressors, "
                "each with its parameters, metrics, plots and the fitted model.")
        with b, st.container(border=True):
            st.markdown("#### :material/workspace_premium: Model registry")
            st.markdown(
                f"`{meta['classifier']['registry_name']}` v{meta['classifier']['registry_version']}  \n"
                f"`{meta['regressor']['registry_name']}` v{meta['regressor']['registry_version']}  \n"
                "Both carry the alias **champion**.")
        st.markdown("Open the tracking UI from the project folder, then visit http://127.0.0.1:5000")
        st.code("mlflow ui --backend-store-uri sqlite:///mlflow.db", language="bash")


# ----------------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------------
PAGES = {
    "overview": st.Page(page_overview, title="Overview", icon=":material/space_dashboard:",
                        url_path="overview", default=True),
    "predict": st.Page(page_predict, title="Investment predictor", icon=":material/insights:",
                       url_path="predict"),
    "explore": st.Page(page_explore, title="Property explorer", icon=":material/search:",
                       url_path="explore"),
    "market": st.Page(page_market, title="Market insights", icon=":material/map:",
                      url_path="market"),
    "models": st.Page(page_models, title="Model performance", icon=":material/model_training:",
                      url_path="models"),
}

nav = st.navigation(list(PAGES.values()), position="hidden")

with st.sidebar:
    st.html("""<div class="brand"><div class="brand-mark">RE</div>
               <div><div class="brand-name">Investment Advisor</div>
               <div class="brand-sub">Indian residential property</div></div></div>""")
    for pg in PAGES.values():
        st.page_link(pg)
    model_line = (f"<b>{meta['classifier']['model']}</b> and <b>{meta['regressor']['model']}</b>"
                  if meta else "Models not loaded")
    st.html(f"""<div class="side-facts">
        <b>{len(df):,}</b> listings<br>
        <b>{df['City'].nunique()}</b> cities in <b>{df['State'].nunique()}</b> states<br>
        Models: {model_line}</div>""")

nav.run()
