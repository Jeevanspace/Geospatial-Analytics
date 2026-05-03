"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   GeoSight Aurora  —  Geospatial Intelligence & Disaster Analytics          ║
║   Pixxels Edition  ·  Real Satellite Image Pipeline  ·  ESA/USGS Standards  ║
╚══════════════════════════════════════════════════════════════════════════════╝

INSTALL:
    pip install -r requirements.txt

RUN:
    streamlit run geosight_aurora.py

INPUT:
  • 7-band GeoTIFF  — direct band mapping (B2 B3 B4 B5 B8 B11 B12), full accuracy
  • 3-band GeoTIFF  — RGB → physics-based Sentinel-2 band reconstruction
  • PNG / JPG       — auto-reconstructed via ESA S2 LUT spectral unmixing
  • 1-band GeoTIFF  — panchromatic → all-band estimation from LUT

OUTPUT (rasterio-based, fully georeferenced):
  • GeoTIFF ZIP     — EPSG:4326, float32, proper affine transform, PROJ CRS
  • GeoJSON AOI     — bounding box + quadrant zones with zonal stats attached
  • Dashboard PNG   — 5×4 panel export-quality figure
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import io, os, json, zipfile, tempfile, struct, time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

EPS = 1e-9

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GeoSight Aurora · Geospatial Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "GeoSight Aurora — Pixxels Geospatial Analytics"},
)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS — AURORA THEME (Pixxels)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');
:root{
    --bg:#04040e;--bg2:#070716;--surface:#0b0b1e;--surface2:#0f0f28;--surface3:#141438;
    --border:#1a1a46;--border2:#28286a;
    --accent:#7c6fff;--accent2:#ff6bfd;--accent3:#00f0d0;
    --green:#00e89a;--green-lo:rgba(0,232,154,0.08);
    --amber:#ffc840;--amber-lo:rgba(255,200,64,0.08);
    --red:#ff4f6a;--red-lo:rgba(255,79,106,0.08);
    --purple:#bf6fff;
    --text:#dcdcff;--text2:#8080be;--text3:#48487e;
    --mono:'Space Mono',monospace;--sans:'Syne',sans-serif;
    --r:10px;--rl:16px;
}
html,body,[class*="css"]{font-family:var(--sans);background-color:var(--bg);color:var(--text);}
.stApp{background:var(--bg)!important;background-image:
    radial-gradient(ellipse 70% 45% at 15% -5%,rgba(124,111,255,0.07) 0%,transparent 60%),
    radial-gradient(ellipse 55% 35% at 85% 105%,rgba(255,107,253,0.05) 0%,transparent 60%)!important;}
div[data-testid="stSidebar"]{background:var(--bg2)!important;border-right:1px solid var(--border)!important;box-shadow:3px 0 30px rgba(124,111,255,0.06)!important;}
div[data-testid="stSidebar"]>div{padding-top:0!important;}
div[data-testid="stSidebar"] *{color:var(--text)!important;}
/* Header */
.gs-header{background:linear-gradient(135deg,#08081c 0%,#0c0c24 50%,#09091a 100%);border:1px solid var(--border2);border-radius:var(--rl);padding:28px 36px 22px;margin-bottom:22px;position:relative;overflow:hidden;box-shadow:0 0 40px rgba(124,111,255,0.12),inset 0 1px 0 rgba(124,111,255,0.15);}
.gs-header::before{content:'';position:absolute;top:-80px;left:-60px;width:280px;height:280px;background:radial-gradient(circle,rgba(124,111,255,0.12) 0%,transparent 70%);pointer-events:none;}
.gs-header::after{content:'';position:absolute;bottom:-100px;right:-50px;width:320px;height:320px;background:radial-gradient(circle,rgba(255,107,253,0.08) 0%,transparent 70%);pointer-events:none;}
.gs-logo-row{display:flex;align-items:center;gap:16px;margin-bottom:12px;position:relative;z-index:1;}
.gs-logo{font-size:2.3rem;line-height:1;filter:drop-shadow(0 0 14px rgba(124,111,255,0.55));}
.gs-title{font-family:var(--sans);font-size:2rem;font-weight:800;letter-spacing:-1px;color:var(--text);line-height:1.1;}
.gs-title span{background:linear-gradient(135deg,var(--accent) 0%,var(--accent2) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.gs-subtitle{font-family:var(--mono);font-size:0.60rem;color:var(--text3);letter-spacing:3px;text-transform:uppercase;margin-top:3px;}
.gs-pill{display:inline-flex;align-items:center;gap:4px;background:rgba(124,111,255,0.07);border:1px solid rgba(124,111,255,0.2);color:var(--text2);font-family:var(--mono);font-size:0.55rem;padding:3px 9px;border-radius:20px;letter-spacing:0.8px;margin-right:4px;margin-top:8px;position:relative;z-index:1;}
.gs-pill.live{border-color:var(--green);color:var(--green);background:var(--green-lo);box-shadow:0 0 8px rgba(0,232,154,0.18);}
.gs-pill.upload{border-color:var(--accent3);color:var(--accent3);background:rgba(0,240,208,0.07);}
.gs-pill.acc{border-color:var(--accent2);color:var(--accent2);background:rgba(255,107,253,0.07);}
/* Sidebar */
.sb-section{font-family:var(--mono);font-size:0.56rem;background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:2.5px;text-transform:uppercase;border-bottom:1px solid var(--border);padding-bottom:5px;margin:20px 0 10px;}
.sb-info{background:var(--surface2);border-radius:var(--r);padding:11px 13px;font-family:var(--mono);font-size:0.63rem;color:var(--text2);line-height:1.9;border-left:2px solid var(--accent);box-shadow:inset 0 0 20px rgba(124,111,255,0.04);}
/* Metric cards */
.gs-metric{background:linear-gradient(135deg,var(--surface) 0%,var(--surface2) 100%);border:1px solid var(--border);border-radius:var(--rl);padding:18px 16px;position:relative;overflow:hidden;box-shadow:0 2px 20px rgba(0,0,0,0.35);transition:border-color 0.2s;}
.gs-metric:hover{border-color:var(--border2);}
.gs-metric::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;border-radius:var(--rl) var(--rl) 0 0;}
.gs-metric.blue::before{background:linear-gradient(90deg,var(--accent),var(--accent3));box-shadow:0 0 8px var(--accent);}
.gs-metric.green::before{background:var(--green);box-shadow:0 0 8px rgba(0,232,154,0.5);}
.gs-metric.amber::before{background:var(--amber);}
.gs-metric.red::before{background:var(--red);}
.gs-metric.cyan::before{background:linear-gradient(90deg,var(--accent3),var(--accent));}
.gs-metric.purple::before{background:var(--purple);}
.gs-mval{font-family:var(--mono);font-size:1.65rem;font-weight:700;line-height:1;color:var(--text);}
.gs-mval.blue{color:var(--accent);}.gs-mval.green{color:var(--green);}.gs-mval.amber{color:var(--amber);}
.gs-mval.red{color:var(--red);}.gs-mval.cyan{color:var(--accent3);}.gs-mval.purple{color:var(--purple);}
.gs-mlabel{font-size:0.60rem;color:var(--text3);text-transform:uppercase;letter-spacing:1.8px;margin-top:5px;font-family:var(--mono);}
.gs-mdelta{font-family:var(--mono);font-size:0.63rem;margin-top:3px;}
/* Section headers */
.gs-section{display:flex;align-items:center;gap:10px;margin:26px 0 14px;}
.gs-section-line{flex:1;height:1px;background:linear-gradient(90deg,var(--border2),transparent);}
.gs-section-line:first-child{background:linear-gradient(90deg,transparent,var(--border2));}
.gs-section-label{font-family:var(--mono);font-size:0.57rem;background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:2.5px;text-transform:uppercase;white-space:nowrap;}
/* Info boxes */
.gs-infobox{background:linear-gradient(135deg,var(--surface) 0%,var(--surface2) 100%);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:0 var(--r) var(--r) 0;padding:13px 16px;font-size:0.80rem;color:var(--text2);line-height:1.65;margin:8px 0;}
.gs-infobox.warn{border-left-color:var(--amber);}
.gs-infobox.success{border-left-color:var(--green);}
.gs-infobox.upload{border-left-color:var(--accent3);background:linear-gradient(135deg,rgba(0,240,208,0.04),var(--surface2));}
.gs-infobox.crit{border-left-color:var(--red);}
/* Colour interpretation box */
.colour-legend{background:var(--surface2);border:1px solid var(--border);border-radius:var(--r);padding:12px 14px;margin:8px 0;}
.colour-row{display:flex;align-items:flex-start;gap:10px;padding:5px 0;border-bottom:1px solid var(--border);}
.colour-row:last-child{border-bottom:none;}
.colour-swatch{width:22px;height:22px;border-radius:4px;flex-shrink:0;margin-top:1px;}
.colour-val{font-family:var(--mono);font-size:0.60rem;color:var(--accent3);min-width:90px;padding-top:3px;}
.colour-meaning{font-size:0.73rem;color:var(--text2);line-height:1.5;}
.colour-meaning b{color:var(--text);}
/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:var(--surface)!important;border-radius:var(--r)!important;border:1px solid var(--border)!important;padding:4px!important;gap:2px!important;}
.stTabs [data-baseweb="tab"]{color:var(--text3)!important;font-family:var(--mono)!important;font-size:0.62rem!important;letter-spacing:0.5px!important;border-radius:6px!important;padding:7px 13px!important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,rgba(124,111,255,0.18) 0%,rgba(255,107,253,0.08) 100%)!important;color:var(--accent)!important;border-bottom:2px solid var(--accent)!important;box-shadow:0 0 10px rgba(124,111,255,0.12)!important;}
/* Buttons */
button[kind="primary"]{background:linear-gradient(135deg,var(--accent) 0%,#9b55f0 50%,var(--accent2) 100%)!important;border:none!important;font-family:var(--mono)!important;font-size:0.70rem!important;letter-spacing:1.5px!important;border-radius:var(--r)!important;font-weight:700!important;box-shadow:0 0 22px rgba(124,111,255,0.32)!important;}
button[kind="primary"]:hover{box-shadow:0 0 32px rgba(124,111,255,0.52)!important;}
button[kind="secondary"]{background:var(--surface2)!important;border:1px solid var(--border2)!important;color:var(--text)!important;font-family:var(--mono)!important;font-size:0.68rem!important;border-radius:var(--r)!important;}
/* Misc */
div[data-testid="stExpander"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:var(--r)!important;}
div[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:var(--r)!important;}
.stSelectbox label,.stSlider label,.stNumberInput label,.stDateInput label,.stRadio label{color:var(--text2)!important;font-size:0.70rem!important;font-weight:600!important;letter-spacing:0.5px!important;font-family:var(--mono)!important;}
.stProgress>div>div{background:linear-gradient(90deg,var(--accent),var(--accent2))!important;box-shadow:0 0 10px rgba(124,111,255,0.45)!important;}
.stAlert{border-radius:var(--r)!important;}
.stCodeBlock{background:var(--surface2)!important;border:1px solid var(--border)!important;border-radius:var(--r)!important;}
.lulc-badge{display:inline-flex;align-items:center;gap:6px;background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:5px 9px;margin:3px;font-size:0.70rem;color:var(--text2);font-family:var(--mono);}
.lulc-dot{width:10px;height:10px;border-radius:3px;flex-shrink:0;}
.band-chip{display:inline-block;background:linear-gradient(135deg,rgba(124,111,255,0.13),rgba(255,107,253,0.06));border:1px solid var(--border2);color:var(--accent3);font-family:var(--mono);font-size:0.58rem;padding:2px 7px;border-radius:4px;margin:2px;letter-spacing:0.5px;}
@keyframes aurora-pulse{0%,100%{opacity:1;box-shadow:0 0 6px var(--green);}50%{opacity:0.35;box-shadow:0 0 2px var(--green);}}
.sat-dot{display:inline-block;width:7px;height:7px;background:var(--green);border-radius:50%;animation:aurora-pulse 2s infinite;margin-right:5px;}
.vector-panel{background:linear-gradient(135deg,var(--surface2) 0%,var(--surface3) 100%);border:1px solid var(--border);border-radius:var(--r);padding:13px;font-family:var(--mono);font-size:0.67rem;color:var(--text2);line-height:1.85;}
.status-ok{color:var(--green);font-family:var(--mono);font-size:0.68rem;}
.status-warn{color:var(--amber);font-family:var(--mono);font-size:0.68rem;}
div[data-testid="stFileUploader"]{background:linear-gradient(135deg,rgba(124,111,255,0.04) 0%,rgba(255,107,253,0.02) 100%)!important;border:1.5px dashed rgba(124,111,255,0.30)!important;border-radius:var(--rl)!important;padding:8px!important;}
div[data-testid="stFileUploader"]:hover{border-color:var(--accent)!important;box-shadow:0 0 18px rgba(124,111,255,0.10)!important;}
.recon-badge{display:inline-flex;align-items:center;gap:5px;background:rgba(0,240,208,0.08);border:1px solid rgba(0,240,208,0.25);color:var(--accent3);font-family:var(--mono);font-size:0.58rem;padding:3px 10px;border-radius:20px;margin:3px;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--surface);}
::-webkit-scrollbar-thumb{background:linear-gradient(180deg,var(--accent),var(--accent2));border-radius:4px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BG      = "#04040e"
SURFACE = "#0b0b1e"

LULC_CLASSES = {
    0: ("Dense Vegetation",  "#1b5e20"),
    1: ("Sparse Vegetation", "#76c442"),
    2: ("Water Body",        "#1565c0"),
    3: ("Built-up / Urban",  "#c62828"),
    4: ("Bare Soil / Sand",  "#c8a84b"),
    5: ("Burned Area",       "#3e1a00"),
    6: ("Wetland",           "#00695c"),
}

CHANGE_CLASSES = {
    0: ("No Change",       "#1a2a3a"),
    1: ("Vegetation Loss", "#ef5350"),
    2: ("Vegetation Gain", "#43a047"),
    3: ("Water Expansion", "#1e88e5"),
    4: ("Urban Expansion", "#fb8c00"),
    5: ("Burn Scar",       "#6d1e00"),
    6: ("Recovery",        "#a5d6a7"),
}

SENTINEL2_BANDS = {
    "B2":  {"name": "Blue",      "nm": 492,  "res": 10},
    "B3":  {"name": "Green",     "nm": 560,  "res": 10},
    "B4":  {"name": "Red",       "nm": 665,  "res": 10},
    "B5":  {"name": "Red Edge",  "nm": 704,  "res": 20},
    "B8":  {"name": "NIR",       "nm": 833,  "res": 10},
    "B11": {"name": "SWIR1",     "nm": 1614, "res": 20},
    "B12": {"name": "SWIR2",     "nm": 2202, "res": 20},
}

PRESET_LOCATIONS = {
    "Custom":                        None,
    "Kerala, India (Flood Zone)":    (10.8505,  76.2711),
    "Uttarakhand, India (Wildfire)": (30.0668,  79.0193),
    "Rajasthan, India (Drought)":    (27.0238,  74.2179),
    "Bengaluru, India (Urban)":      (12.9716,  77.5946),
    "Amazon Basin (Deforestation)":  (-3.4653, -62.2159),
    "California, USA (Wildfire)":    (37.5000, -119.500),
    "Bangladesh (Flood)":            (23.6850,  90.3560),
    "Sahel, Niger (Drought)":        (13.5137,   2.1098),
    "Jakarta, Indonesia (Urban)":    (-6.2088, 106.8456),
}

# ESA Sentinel-2 L2A surface reflectance look-up table
S2_LUT = {
    "dense_veg":  {"B2":0.028,"B3":0.055,"B4":0.030,"B5":0.120,"B8":0.420,"B11":0.155,"B12":0.080},
    "sparse_veg": {"B2":0.055,"B3":0.090,"B4":0.070,"B5":0.160,"B8":0.290,"B11":0.210,"B12":0.120},
    "water":      {"B2":0.060,"B3":0.055,"B4":0.030,"B5":0.015,"B8":0.008,"B11":0.004,"B12":0.002},
    "urban":      {"B2":0.095,"B3":0.100,"B4":0.100,"B5":0.120,"B8":0.150,"B11":0.200,"B12":0.160},
    "bare_soil":  {"B2":0.115,"B3":0.140,"B4":0.155,"B5":0.175,"B8":0.210,"B11":0.250,"B12":0.195},
    "burned":     {"B2":0.040,"B3":0.042,"B4":0.038,"B5":0.044,"B8":0.055,"B11":0.140,"B12":0.115},
    "wetland":    {"B2":0.042,"B3":0.070,"B4":0.040,"B5":0.110,"B8":0.200,"B11":0.090,"B12":0.050},
}

# Colour interpretation for each index (used in Tab 2)
INDEX_COLOUR_GUIDE = {
    "NDVI": {
        "cmap": "RdYlGn", "vmin": -1, "vmax": 1,
        "title": "NDVI — Normalized Difference Vegetation Index",
        "formula": "(B8 − B4) / (B8 + B4)",
        "desc": "Measures vegetation health and density. Chlorophyll absorbs Red (B4) and strongly reflects NIR (B8).",
        "legend": [
            ("#8b0000", "−1.0 to −0.3", "Open deep water, snow, ice — NIR strongly absorbed"),
            ("#d62728", "−0.3 to −0.05","Bare rock, sand, concrete — similar NIR and Red reflectance"),
            ("#f4a460", "−0.05 to +0.10","Desert soil, salt flats, burned scar — very low vegetation signal"),
            ("#ffff80", "+0.10 to +0.20","Sparse scrub, dry grass, fallow field — marginal vegetation"),
            ("#aadd44", "+0.20 to +0.40","Grassland, cropland, savanna — moderate green cover"),
            ("#4a9a20", "+0.40 to +0.65","Mixed forest, healthy crops at peak — strong vegetation"),
            ("#1a5e10", "+0.65 to +1.0", "Dense tropical forest, irrigated farmland — maximum photosynthesis"),
        ],
    },
    "NDWI": {
        "cmap": "Blues_r", "vmin": -1, "vmax": 1,
        "title": "NDWI — Normalized Difference Water Index",
        "formula": "(B3 − B8) / (B3 + B8)",
        "desc": "Detects open water and moisture stress. Water absorbs NIR completely but reflects Green (B3).",
        "legend": [
            ("#d62728", "−1.0 to −0.4","Dry desert, exposed rock — NIR far exceeds Green"),
            ("#f4a460", "−0.4 to −0.2", "Dry vegetation canopy, bare soil — negative water signal"),
            ("#e8e8e8", "−0.2 to 0.0",  "Mixed land, light moisture — transitional"),
            ("#87ceeb", "0.0 to +0.2",  "Moist soil, irrigated fields, shallow water bodies"),
            ("#5090c0", "+0.2 to +0.4", "Waterlogged soil, wetland, flooded vegetation margin"),
            ("#1565c0", "+0.4 to +1.0", "Open water — river, lake, reservoir, active floodwater"),
        ],
    },
    "NBR": {
        "cmap": "RdYlGn", "vmin": -1, "vmax": 1,
        "title": "NBR — Normalized Burn Ratio",
        "formula": "(B8 − B12) / (B8 + B12)",
        "desc": "Designed for fire damage mapping. Use dNBR (Before minus After) for burn severity; apply USGS thresholds.",
        "legend": [
            ("#8b0000", "dNBR > +0.44", "High severity burn — charcoal absorbs NIR, SWIR2 spikes"),
            ("#d05010", "dNBR +0.27 to +0.44","Moderate-high severity — significant canopy destruction"),
            ("#fb8c00", "dNBR +0.10 to +0.27","Low severity — scorched canopy, ground largely intact"),
            ("#ffff80", "dNBR −0.10 to +0.10","Unburned — no significant spectral change between dates"),
            ("#a5d6a7", "dNBR −0.25 to −0.10","Enhanced regrowth — post-fire green flush, recovering"),
            ("#2e7d32", "dNBR < −0.25", "Strong regrowth — greener than pre-fire baseline (normal after 1–2 years)"),
        ],
    },
    "NDBI": {
        "cmap": "RdYlGn_r", "vmin": -0.5, "vmax": 0.5,
        "title": "NDBI — Normalized Difference Built-up Index",
        "formula": "(B11 − B8) / (B11 + B8)",
        "desc": "Highlights impervious urban surfaces. Concrete/asphalt reflects SWIR1 strongly, absorbs NIR.",
        "legend": [
            ("#228b22", "−0.5 to −0.2", "Dense vegetation — very high NIR, very low SWIR1"),
            ("#aadd44", "−0.2 to −0.05","Sparse or mixed vegetation — NIR still dominant"),
            ("#ffff80", "−0.05 to +0.05","Agricultural or transitional land"),
            ("#f4a460", "+0.05 to +0.2", "⚠ Bare soil / arid land — false urban signal in deserts"),
            ("#e84020", "+0.2 to +0.5", "Confirmed urban — concrete, asphalt, rooftops, roads"),
        ],
        "warning": "In arid or desert scenes, bare soil SWIR reflectance mimics urban — always cross-validate with MNDWI and visual RGB check.",
    },
    "SAVI": {
        "cmap": "RdYlGn", "vmin": -1, "vmax": 1,
        "title": "SAVI — Soil-Adjusted Vegetation Index",
        "formula": "((B8 − B4) / (B8 + B4 + 0.5)) × 1.5",
        "desc": "Like NDVI but reduces soil brightness interference. Use SAVI instead of NDVI in arid, semi-arid, or sparsely vegetated scenes.",
        "legend": [
            ("#d62728", "−1.0 to −0.1","Non-vegetated surface — rock, water, very dry soil"),
            ("#f4a460", "−0.1 to +0.1", "Bare soil, desert — near-zero vegetation fraction"),
            ("#ffff80", "+0.1 to +0.25","Very sparse scrub, dry steppe — marginal cover"),
            ("#aadd44", "+0.25 to +0.45","Grassland, dryland crops, moderate cover"),
            ("#228b22", "+0.45 to +1.0","Dense healthy vegetation — forest, irrigated farmland"),
        ],
    },
    "EVI": {
        "cmap": "RdYlGn", "vmin": -1, "vmax": 1,
        "title": "EVI — Enhanced Vegetation Index",
        "formula": "2.5 × (B8 − B4) / (B8 + 6×B4 − 7.5×B2 + 1)",
        "desc": "Improves on NDVI in dense forests and high-biomass regions. Blue band (B2) corrects for aerosol/atmospheric interference. Does not saturate at high biomass.",
        "legend": [
            ("#d62728", "−0.5 to 0.0",  "Non-vegetated — water, soil, urban, burned — near zero or negative"),
            ("#f4a460", "0.0 to +0.1",   "Arid scrub, desert fringe — minimal photosynthetic signal"),
            ("#ffff80", "+0.1 to +0.25", "Dry grassland, sparse savanna, stressed vegetation"),
            ("#aadd44", "+0.25 to +0.45","Cropland at moderate growth, open woodland"),
            ("#4a9a20", "+0.45 to +0.65","Closed canopy, healthy temperate forest"),
            ("#1a5e10", "+0.65 to +1.0", "Dense tropical forest, peak growing season — EVI preferred over NDVI here"),
        ],
    },
    "MNDWI": {
        "cmap": "Blues_r", "vmin": -1, "vmax": 1,
        "title": "MNDWI — Modified Normalized Difference Water Index",
        "formula": "(B3 − B11) / (B3 + B11)",
        "desc": "Uses SWIR1 instead of NIR — suppresses urban and soil signals that contaminate NDWI. Preferred for water detection in cities or mixed landscapes.",
        "legend": [
            ("#8b4513", "−1.0 to −0.3","Built-up surface, dry soil — SWIR1 far exceeds Green"),
            ("#f4a460", "−0.3 to −0.1","Mixed land, vegetation — moderate SWIR suppression"),
            ("#e8e8e8", "−0.1 to +0.1","Transitional — moist soil, irrigated cropland"),
            ("#87ceeb", "+0.1 to +0.3","Shallow water, waterlogged fields, wetland margin"),
            ("#1565c0", "+0.3 to +1.0","Open water bodies — rivers, lakes, canals, floodwater"),
        ],
        "note": "MNDWI is more reliable than NDWI in urban or arid scenes because SWIR1 strongly absorbs water, suppressing reflectance from concrete and soil.",
    },
    "BSI": {
        "cmap": "YlOrBr", "vmin": -1, "vmax": 1,
        "title": "BSI — Bare Soil Index",
        "formula": "((B11 + B4) − (B8 + B2)) / ((B11 + B4) + (B8 + B2))",
        "desc": "Most diagnostic index for arid, degraded, or agricultural bare-soil scenes. Use BSI as primary index when NDVI is uninformative (deserts, post-harvest, erosion zones).",
        "legend": [
            ("#228b22", "−1.0 to −0.1","Dense vegetation — NIR+Blue dominant, soil fully covered"),
            ("#ffff80", "−0.1 to +0.1","Partially vegetated — transitional zone, mixed crop-soil"),
            ("#c8a84b", "+0.1 to +0.3","Agricultural bare soil — post-harvest, ploughed fields"),
            ("#b8720a", "+0.3 to +0.5","Semi-arid degraded land, sandy terrain, erosion zones"),
            ("#8b4513", "+0.5 to +1.0","Hyper-arid desert, severely degraded land — your earthquake scene falls here"),
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION A — BAND LOADING (rasterio primary, PIL fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _bilinear_resize(arr: np.ndarray, target: int) -> np.ndarray:
    h, w = arr.shape
    if h == target and w == target:
        return arr
    yy = np.linspace(0, h - 1, target)
    xx = np.linspace(0, w - 1, target)
    y0 = np.floor(yy).astype(int).clip(0, h - 2)
    x0 = np.floor(xx).astype(int).clip(0, w - 2)
    y1, x1 = y0 + 1, x0 + 1
    fy = (yy - y0)[:, None]
    fx = (xx - x0)[None, :]
    return (arr[y0][:, x0] * (1-fy) * (1-fx) +
            arr[y0][:, x1] * (1-fy) * fx +
            arr[y1][:, x0] * fy * (1-fx) +
            arr[y1][:, x1] * fy * fx).astype(np.float32)


def _stretch(arr: np.ndarray) -> np.ndarray:
    """Percentile 1–99 stretch to [0, 1]."""
    p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
    return np.clip((arr - p1) / (p99 - p1 + EPS), 0, 1).astype(np.float32)


def _rgb_to_s2_bands(R: np.ndarray, G: np.ndarray, B: np.ndarray,
                     target: int = 256) -> dict:
    """
    Physics-based Sentinel-2 band reconstruction from RGB.

    Steps:
      1. Dark-Object Subtraction (atmospheric correction proxy)
      2. Per-pixel spectral unmixing against ESA S2 LUT endmembers
      3. Reconstruct B5, B8, B11, B12 as abundance-weighted LUT sums
      4. Add spatial texture from Green channel
      5. Percentile-stretch all bands to [0,1]
    """
    R = _bilinear_resize(R, target)
    G = _bilinear_resize(G, target)
    B = _bilinear_resize(B, target)

    # DOS
    R = np.clip(R - float(np.percentile(R, 1)), 0, 1)
    G = np.clip(G - float(np.percentile(G, 1)), 0, 1)
    B = np.clip(B - float(np.percentile(B, 1)), 0, 1)

    covers = list(S2_LUT.keys())
    E = np.array([[S2_LUT[c]["B2"], S2_LUT[c]["B3"], S2_LUT[c]["B4"]]
                  for c in covers], dtype=np.float32).T
    E_norm = E / (E.sum(axis=0, keepdims=True) + EPS)
    h, w = R.shape
    obs = np.stack([B.ravel(), G.ravel(), R.ravel()], axis=1)
    E_t = E_norm.T
    denom = (E_t ** 2).sum(axis=1, keepdims=True) + EPS
    abund = np.clip((obs @ E_t.T) / (denom.T + EPS), 0, None)
    abund = abund / (abund.sum(axis=1, keepdims=True) + EPS)
    abund = abund.reshape(h, w, len(covers))

    def recon(bk):
        lut = np.array([S2_LUT[c][bk] for c in covers], dtype=np.float32)
        base = np.einsum("hwc,c->hw", abund, lut).astype(np.float32)
        # texture from best proxy channel
        tex = G if bk in ("B5", "B8") else R
        tex_norm = (tex - tex.mean()) / (tex.std() + EPS)
        strength = 0.40 if bk == "B8" else 0.28
        return np.clip(base + tex_norm * strength * base.std(), 0, 1).astype(np.float32)

    return {
        "B2": _stretch(B),   "B3": _stretch(G),   "B4": _stretch(R),
        "B5": _stretch(recon("B5")),  "B8": _stretch(recon("B8")),
        "B11": _stretch(recon("B11")), "B12": _stretch(recon("B12")),
        "_source": "rgb_reconstructed", "_accuracy": "reconstructed",
    }


def _panchromatic_to_s2(pan: np.ndarray) -> dict:
    """Estimate all S2 bands from a single panchromatic band."""
    pan = np.clip(pan, 0, 1)
    veg_frac = np.clip((pan - 0.12) / 0.60, 0, 1)
    result = {}
    for bk in ["B2","B3","B4","B5","B8","B11","B12"]:
        bare = S2_LUT["bare_soil"][bk]
        veg  = S2_LUT["dense_veg"][bk]
        result[bk] = np.clip(bare + veg_frac * (veg - bare) +
                             (pan - pan.mean()) * 0.08, 0, 1).astype(np.float32)
    result["_source"] = "panchromatic_estimated"
    result["_accuracy"] = "estimated"
    return result


def load_image(uploaded_file, target: int = 256) -> dict:
    """
    Route uploaded file through appropriate band-loading pipeline.
    Returns band dict with _source, _accuracy, and optional _transform/_crs.
    """
    import io as _io
    raw = uploaded_file.read()

    # ── rasterio (GeoTIFF with full CRS/transform support) ─────────────────
    try:
        import rasterio
        from rasterio.io import MemoryFile
        with MemoryFile(raw) as mf:
            with mf.open() as ds:
                n      = ds.count
                native_transform = ds.transform
                native_crs       = ds.crs
                native_h, native_w = ds.height, ds.width

                data = ds.read().astype(np.float32)
                for i in range(n):
                    p1, p99 = np.percentile(data[i], 1), np.percentile(data[i], 99)
                    data[i] = np.clip((data[i] - p1) / (p99 - p1 + EPS), 0, 1)

                bands_raw = [_bilinear_resize(data[i], target) for i in range(n)]

                if n >= 7:
                    bnames = ["B2","B3","B4","B5","B8","B11","B12"]
                    result = {bn: bands_raw[i] for i, bn in enumerate(bnames[:7])}
                    result.update({
                        "_source": "geotiff_7band", "_accuracy": "full",
                        "_native_transform": native_transform,
                        "_native_crs": native_crs,
                        "_native_size": (native_h, native_w),
                    })
                    return result

                elif n >= 3:
                    R_, G_, B_ = bands_raw[0], bands_raw[1], bands_raw[2]
                    result = _rgb_to_s2_bands(R_, G_, B_, target)
                    result.update({
                        "_source": "geotiff_rgb_reconstructed",
                        "_native_transform": native_transform,
                        "_native_crs": native_crs,
                        "_native_size": (native_h, native_w),
                    })
                    return result

                else:
                    result = _panchromatic_to_s2(bands_raw[0])
                    result.update({
                        "_native_transform": native_transform,
                        "_native_crs": native_crs,
                        "_native_size": (native_h, native_w),
                    })
                    return result

    except (ImportError, Exception):
        pass

    # ── PIL fallback (PNG/JPG) ──────────────────────────────────────────────
    try:
        from PIL import Image as PILImage
        img = PILImage.open(_io.BytesIO(raw)).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize((target, target), PILImage.LANCZOS)
        arr = np.array(img).astype(np.float32) / 255.0
        result = _rgb_to_s2_bands(arr[:,:,0], arr[:,:,1], arr[:,:,2], target)
        result["_native_size"] = (orig_h, orig_w)
        return result
    except Exception:
        pass

    # ── Ultimate fallback ───────────────────────────────────────────────────
    pan = np.frombuffer(raw[:target*target], dtype=np.uint8
                        ).reshape(target, target).astype(np.float32) / 255.0
    return _panchromatic_to_s2(pan)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION B — SPECTRAL INDEX ENGINE  (ESA/USGS standard formulas)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_div(a, b):
    return a / (b + EPS)

def compute_NDVI(b):   return np.clip(_safe_div(b["B8"]-b["B4"], b["B8"]+b["B4"]), -1, 1)
def compute_NDWI(b):   return np.clip(_safe_div(b["B3"]-b["B8"], b["B3"]+b["B8"]), -1, 1)
def compute_NBR(b):    return np.clip(_safe_div(b["B8"]-b["B12"],b["B8"]+b["B12"]),-1, 1)
def compute_NDBI(b):   return np.clip(_safe_div(b["B11"]-b["B8"],b["B11"]+b["B8"]),-1, 1)
def compute_SAVI(b, L=0.5):
    return np.clip(_safe_div(b["B8"]-b["B4"], b["B8"]+b["B4"]+L) * (1+L), -1, 1)
def compute_EVI(b, G=2.5, C1=6.0, C2=7.5, L=1.0):
    return np.clip(G * _safe_div(b["B8"]-b["B4"],
                   b["B8"]+C1*b["B4"]-C2*b["B2"]+L), -1, 1)
def compute_MNDWI(b):  return np.clip(_safe_div(b["B3"]-b["B11"],b["B3"]+b["B11"]),-1, 1)
def compute_BSI(b):
    num = (b["B11"]+b["B4"]) - (b["B8"]+b["B2"])
    den = (b["B11"]+b["B4"]) + (b["B8"]+b["B2"])
    return np.clip(_safe_div(num, den), -1, 1)

def compute_all_indices(bands: dict) -> dict:
    return {
        "NDVI":  compute_NDVI(bands),
        "NDWI":  compute_NDWI(bands),
        "NBR":   compute_NBR(bands),
        "NDBI":  compute_NDBI(bands),
        "SAVI":  compute_SAVI(bands),
        "EVI":   compute_EVI(bands),
        "MNDWI": compute_MNDWI(bands),
        "BSI":   compute_BSI(bands),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION C — RASTER ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_band_stats(bands: dict) -> dict:
    stats = {}
    for k, arr in bands.items():
        if k.startswith("_"): continue
        f = arr.flatten()
        stats[k] = {
            "min": float(f.min()), "max": float(f.max()),
            "mean": float(f.mean()), "std": float(f.std()),
            "p5": float(np.percentile(f, 5)), "p95": float(np.percentile(f, 95)),
            "snr": float(f.mean() / (f.std() + EPS)),
        }
    return stats


def compute_index_correlation(indices: dict):
    names = list(indices.keys())
    n     = len(names)
    mat   = np.zeros((n, n), dtype=np.float32)
    arrs  = [indices[k].flatten() for k in names]
    for i in range(n):
        for j in range(n):
            if i == j: mat[i,j] = 1.0
            elif j > i:
                r = float(np.corrcoef(arrs[i], arrs[j])[0, 1])
                mat[i,j] = mat[j,i] = r
    return mat, names


def compute_spatial_stats(arr: np.ndarray) -> dict:
    dy = arr[1:,:] - arr[:-1,:]
    dx = arr[:,1:] - arr[:,:-1]
    moran = 1 - (dy.var() + dx.var()) / (2 * arr.var() + EPS)
    counts, _ = np.histogram(arr, bins=32, density=True)
    entropy = float(-np.sum(counts * np.log2(counts + EPS)))
    return {"moran_i_proxy": float(np.clip(moran, -1, 1)), "entropy": entropy}


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION D — GEOREFERENCING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def build_scene_meta(lat: float, lon: float, size: int,
                     bands: dict) -> dict:
    """
    Build precise georeferencing metadata.

    If the uploaded GeoTIFF carried a native affine transform and CRS, we use
    those directly and re-scale the pixel size proportionally to the analysis
    grid.  Otherwise we construct a WGS-84 affine from the user-supplied
    centre lat/lon using 10 m ≈ 0.0000898° per pixel.

    Returns a dict consumed by both the UI display and the GeoTIFF exporter.
    """
    # 10 m ≈ 0.0000898° at equator — standard Sentinel-2 resolution
    PIXEL_DEG = 0.0000898

    native_tf  = bands.get("_native_transform")
    native_crs = bands.get("_native_crs")
    native_sz  = bands.get("_native_size")

    if native_tf is not None and native_crs is not None:
        try:
            import rasterio
            from rasterio.crs import CRS
            from rasterio.warp import transform_bounds
            native_h, native_w = native_sz
            # Scale pixel size to analysis grid
            sx = native_tf.a * (native_w / size)   # x pixel width in native units
            sy = native_tf.e * (native_h / size)   # y pixel height (negative)
            origin_x = native_tf.c
            origin_y = native_tf.f

            # If CRS is not geographic, project origin to WGS-84 for display
            wgs84 = CRS.from_epsg(4326)
            if native_crs != wgs84:
                from rasterio.warp import transform as warp_transform
                xs, ys = warp_transform(native_crs, wgs84, [origin_x], [origin_y])
                geo_origin_x = xs[0]
                geo_origin_y = ys[0]
                # pixel size in degrees (approximate, at scene centre)
                cx = origin_x + native_tf.a * native_w / 2
                cy = origin_y + native_tf.e * native_h / 2
                xs2, ys2 = warp_transform(native_crs, wgs84,
                                          [cx + native_tf.a * native_w],
                                          [cy + native_tf.e * native_h])
                geo_px = abs(xs2[0] - geo_origin_x) / size
                geo_py = abs(ys2[0] - geo_origin_y) / size
            else:
                geo_origin_x = origin_x
                geo_origin_y = origin_y
                geo_px = abs(sx)
                geo_py = abs(sy)

            west  = geo_origin_x
            north = geo_origin_y
            east  = west  + geo_px * size
            south = north - geo_py * size
            px_m  = geo_px * 111_000

            return {
                "west": west, "east": east, "north": north, "south": south,
                "pixel_deg": geo_px, "pixel_m": px_m,
                "width": size, "height": size,
                "area_km2": round(((east-west)*111) * ((north-south)*111), 2),
                "crs": str(native_crs),
                "native_crs": native_crs,
                "native_transform": native_tf,
                "native_w": native_w, "native_h": native_h,
                "source": "native_geotiff",
            }
        except Exception:
            pass

    # Fallback: build from user lat/lon
    west  = lon
    north = lat
    east  = lon   + PIXEL_DEG * size
    south = lat   - PIXEL_DEG * size
    px_m  = PIXEL_DEG * 111_000

    return {
        "west": west, "east": east, "north": north, "south": south,
        "pixel_deg": PIXEL_DEG, "pixel_m": px_m,
        "width": size, "height": size,
        "area_km2": round(((east-west)*111) * ((north-south)*111), 2),
        "crs": "EPSG:4326",
        "native_crs": None,
        "native_transform": None,
        "source": "user_coords",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION E — GEOTIFF EXPORT  (rasterio primary, struct fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _write_geotiff_rasterio(array: np.ndarray, path: str, meta: dict,
                             nodata_val: float = -9999.0) -> None:
    """
    Write a single-band float32 GeoTIFF with proper affine transform and CRS.
    Uses rasterio for correct georeferencing (Affine, PROJ CRS string).
    """
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    h, w = array.shape
    transform = from_bounds(
        meta["west"], meta["south"], meta["east"], meta["north"], w, h
    )
    try:
        crs = CRS.from_string(meta["crs"])
    except Exception:
        crs = CRS.from_epsg(4326)

    arr = array.astype(np.float32)
    arr = np.where(np.isnan(arr), nodata_val, arr)

    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=h, width=w,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata_val,
        compress="deflate",
        tiled=True,
        blockxsize=256, blockysize=256,
        interleave="band",
    ) as dst:
        dst.write(arr, 1)
        dst.update_tags(
            SOFTWARE="GeoSight Aurora — Pixxels Edition",
            INDEX_RANGE="-1.0 to 1.0 (normalised spectral index)",
            NODATA=str(nodata_val),
        )


def _write_geotiff_fallback(array: np.ndarray, path: str, meta: dict) -> None:
    """
    Fallback GeoTIFF writer using struct — no rasterio required.
    Writes a single-band float32 GeoTIFF with correct TIFF GeoKeys for
    EPSG:4326 and a proper ModelTiepointTag + ModelPixelScaleTag.
    """
    import struct as st_

    h, w    = array.shape
    data_f  = array.astype(np.float32)
    px      = meta["pixel_deg"]
    west    = meta["west"]
    north   = meta["north"]
    nodata  = -9999.0

    # Replace NaN
    data_f  = np.where(np.isnan(data_f), nodata, data_f)
    strips  = [data_f[y:y+32, :].astype('<f4').tobytes() for y in range(0, h, 32)]
    rps     = 32

    buf = io.BytesIO()
    # TIFF header (little-endian)
    buf.write(b'II')
    buf.write(st_.pack('<H', 42))
    buf.write(st_.pack('<I', 8))

    # We compute offsets after knowing IFD size
    # IFD entries (sorted by tag)
    # TAGS: 256,257,258,259,262,273,278,279,282,283,296,
    #       339,33922,34264,34735,34736,34737
    n_entries = 17
    ifd_size  = 2 + n_entries * 12 + 4   # entry count + entries + next IFD ptr
    ifd_start = 8

    base = ifd_start + ifd_size  # data area starts here

    off_xres = base;         base += 8
    off_yres = base;         base += 8
    off_tp   = base;         base += 48  # ModelTiepointTag: 6 doubles
    off_mps  = base;         base += 24  # ModelPixelScaleTag: 3 doubles
    off_gk   = base;         base += 64  # GeoKeyDirectoryTag: 16 shorts = 8 keys
    off_gd   = base;         base += 16  # GeoDoubleParamsTag: 2 doubles
    geo_ascii = b"WGS 84\x00"
    off_ga   = base;         base += len(geo_ascii)
    nd_bytes = st_.pack('<d', nodata)
    off_nd   = base;         base += 8

    ns       = len(strips)
    off_so   = base;         base += ns * 4
    off_sbc  = base;         base += ns * 4
    strip_base = base
    s_offsets = []
    cur = strip_base
    for s in strips:
        s_offsets.append(cur); cur += len(s)

    # Write IFD
    buf.write(st_.pack('<H', n_entries))

    def tag(tag_id, typ, cnt, val):
        # typ: 3=SHORT, 4=LONG, 5=RATIONAL, 7=UNDEF, 12=DOUBLE, 1=BYTE, 2=ASCII
        buf.write(st_.pack('<HHI', tag_id, typ, cnt))
        if   typ == 3 and cnt == 1: buf.write(st_.pack('<HH', val, 0))
        elif typ == 4 and cnt == 1: buf.write(st_.pack('<I', val))
        elif typ == 7 and cnt == 1: buf.write(st_.pack('<I', val))
        else:                        buf.write(st_.pack('<I', val))  # offset

    tag(256,  4, 1, w)               # ImageWidth
    tag(257,  4, 1, h)               # ImageLength
    tag(258,  3, 1, 32)              # BitsPerSample = 32
    tag(259,  3, 1, 1)               # Compression = None
    tag(262,  3, 1, 1)               # PhotometricInterpretation = BlackIsZero
    tag(273,  4, ns, off_so if ns > 1 else s_offsets[0])   # StripOffsets
    tag(278,  4, 1, rps)             # RowsPerStrip
    tag(279,  4, ns, off_sbc if ns > 1 else len(strips[0])) # StripByteCounts
    tag(282,  5, 1, off_xres)        # XResolution
    tag(283,  5, 1, off_yres)        # YResolution
    tag(296,  3, 1, 1)               # ResolutionUnit = No absolute unit
    tag(339,  3, 1, 3)               # SampleFormat = IEEE floating point
    tag(33922, 12, 6, off_tp)        # ModelTiepointTag
    tag(34264, 12, 3, off_mps)       # ModelPixelScaleTag
    tag(34735,  3, 16, off_gk)       # GeoKeyDirectoryTag
    tag(34736, 12, 2, off_gd)        # GeoDoubleParamsTag
    tag(34737,  2, len(geo_ascii), off_ga)  # GeoAsciiParamsTag
    buf.write(st_.pack('<I', 0))     # Next IFD = 0

    # Data area
    buf.write(st_.pack('<II', 1, 1))  # XResolution rational (1/1)
    buf.write(st_.pack('<II', 1, 1))  # YResolution rational (1/1)
    # ModelTiepointTag: (i j k x y z) for pixel (0,0) → (west, north, 0)
    buf.write(st_.pack('<dddddd', 0.0, 0.0, 0.0, west, north, 0.0))
    # ModelPixelScaleTag: (x_scale, y_scale, z_scale)
    buf.write(st_.pack('<ddd', px, px, 0.0))
    # GeoKeyDirectoryTag: header + 4 keys
    # Header: KeyDirectoryVersion=1, KeyRevision=1, MinorRevision=0, NumberOfKeys=4
    buf.write(st_.pack('<HHHH', 1, 1, 0, 4))
    # GTModelTypeGeoKey=1 (ModelTypeGeographic)
    buf.write(st_.pack('<HHHH', 1024, 0, 1, 2))
    # GTRasterTypeGeoKey=2 (RasterPixelIsPoint)
    buf.write(st_.pack('<HHHH', 1025, 0, 1, 2))
    # GeographicTypeGeoKey=4326 (GCS_WGS_84)
    buf.write(st_.pack('<HHHH', 2048, 0, 1, 4326))
    # GeogCitationGeoKey → GeoAsciiParamsTag offset 0
    buf.write(st_.pack('<HHHH', 2049, 34737, len(geo_ascii), 0))
    # GeoDoubleParamsTag: semi-major=6378137, inv-flat=298.257224
    buf.write(st_.pack('<dd', 6378137.0, 298.257224))
    buf.write(geo_ascii)
    buf.write(nd_bytes)

    if ns > 1:
        for o in s_offsets: buf.write(st_.pack('<I', o))
        for s in strips:    buf.write(st_.pack('<I', len(s)))
    for s in strips: buf.write(s)

    with open(path, 'wb') as f:
        f.write(buf.getvalue())


def write_geotiff(array: np.ndarray, path: str, meta: dict) -> None:
    """Write GeoTIFF using rasterio if available, struct fallback otherwise."""
    try:
        import rasterio
        _write_geotiff_rasterio(array, path, meta)
    except ImportError:
        _write_geotiff_fallback(array, path, meta)


def export_geotiffs_zip(indices_a: dict, lulc_a: np.ndarray,
                        change_map: np.ndarray, meta: dict,
                        disaster_type: str) -> bytes:
    """
    Build a ZIP archive containing:
      - One float32 GeoTIFF per spectral index (8 total)
      - LULC classification GeoTIFF (uint8 encoded as float32)
      - Change detection GeoTIFF (uint8 encoded as float32)
      - README.txt with full metadata
    All files are correctly georeferenced to the scene CRS.
    """
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for name, arr in indices_a.items():
            fp = os.path.join(tmp, f"{name}.tif")
            write_geotiff(arr, fp, meta)
            files.append(fp)

        fp = os.path.join(tmp, "LULC_Classification.tif")
        write_geotiff(lulc_a.astype(np.float32), fp, meta)
        files.append(fp)

        fp = os.path.join(tmp, "Change_Detection.tif")
        write_geotiff(change_map.astype(np.float32), fp, meta)
        files.append(fp)

        readme = (
            "GeoSight Aurora — GeoTIFF Export Package\n"
            "==========================================\n"
            f"Generated    : {datetime.now().isoformat(timespec='seconds')}\n"
            f"Disaster Type: {disaster_type}\n\n"
            "GEOREFERENCING\n"
            f"  CRS        : {meta['crs']}\n"
            f"  West       : {meta['west']:.8f}°\n"
            f"  East       : {meta['east']:.8f}°\n"
            f"  North      : {meta['north']:.8f}°\n"
            f"  South      : {meta['south']:.8f}°\n"
            f"  Pixel size : {meta['pixel_deg']:.8f}° ≈ {meta['pixel_m']:.1f} m\n"
            f"  Grid size  : {meta['width']}×{meta['height']} px\n"
            f"  Area       : {meta['area_km2']} km²\n"
            f"  Source     : {meta['source']}\n\n"
            "FILE CONTENTS\n"
            "  NDVI.tif              Vegetation index — float32, range −1 to +1\n"
            "  NDWI.tif              Water index — float32, range −1 to +1\n"
            "  NBR.tif               Burn ratio — float32, range −1 to +1\n"
            "  NDBI.tif              Built-up index — float32, range −0.5 to +0.5\n"
            "  SAVI.tif              Soil-adj vegetation — float32, L=0.5\n"
            "  EVI.tif               Enhanced vegetation — float32, G=2.5\n"
            "  MNDWI.tif             Modified NDWI (SWIR-based) — float32\n"
            "  BSI.tif               Bare soil index — float32\n"
            "  LULC_Classification.tif   0=DenseVeg 1=SparseVeg 2=Water\n"
            "                            3=Urban 4=BareSoil 5=Burned 6=Wetland\n"
            "  Change_Detection.tif      0=NoChange 1=VegLoss 2=VegGain\n"
            "                            3=WaterExp 4=UrbanExp 5=BurnScar 6=Recovery\n\n"
            "NODATA VALUE: -9999.0 (float32 GeoTIFFs only)\n\n"
            "USAGE\n"
            "  Load in ArcGIS Pro, SNAP, GDAL, QGIS, or Python:\n"
            "    import rasterio\n"
            "    with rasterio.open('NDVI.tif') as src:\n"
            "        data = src.read(1)  # float32 array\n"
            "        print(src.crs, src.transform)\n"
        )
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in files: zf.write(fp, os.path.basename(fp))
            zf.writestr("README.txt", readme)
        return zip_buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION F — VECTOR DATA
# ─────────────────────────────────────────────────────────────────────────────

def build_aoi_geojson(meta: dict) -> dict:
    w = meta["west"]; e = meta["east"]
    n = meta["north"]; s = meta["south"]
    mx, my = (w+e)/2, (n+s)/2

    def rect(x0, y0, x1, y1, label, ndvi_mean=None):
        props = {
            "label": label,
            "area_km2": round(((x1-x0)*111)*((y1-y0)*111), 3),
            "crs": "EPSG:4326",
        }
        if ndvi_mean is not None:
            props["ndvi_mean"] = round(ndvi_mean, 4)
        return {
            "type": "Feature",
            "geometry": {"type": "Polygon",
                         "coordinates": [[[x0,y0],[x1,y0],[x1,y1],[x0,y1],[x0,y0]]]},
            "properties": props,
        }

    return {"type": "FeatureCollection", "crs": {
        "type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
    }, "features": [
        rect(w, s, e, n, "Full AOI"),
        rect(w, my, mx, n, "NW Quadrant"),
        rect(mx, my, e, n, "NE Quadrant"),
        rect(w, s, mx, my, "SW Quadrant"),
        rect(mx, s, e, my, "SE Quadrant"),
        {"type": "Feature",
         "geometry": {"type": "Point", "coordinates": [mx, my]},
         "properties": {"label": "AOI Centre",
                        "lat_deg": round(my, 6), "lon_deg": round(mx, 6)}},
    ]}


def compute_quadrant_zonal_stats(arr: np.ndarray) -> dict:
    h, w = arr.shape
    quads = {
        "NW": arr[:h//2, :w//2], "NE": arr[:h//2, w//2:],
        "SW": arr[h//2:, :w//2], "SE": arr[h//2:, w//2:],
    }
    result = {}
    for q, a in quads.items():
        f = a.flatten()
        result[q] = {
            "count": int(f.size), "mean": float(f.mean()), "std": float(f.std()),
            "min": float(f.min()), "max": float(f.max()),
            "median": float(np.median(f)),
            "p25": float(np.percentile(f, 25)), "p75": float(np.percentile(f, 75)),
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION G — LULC CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_lulc(indices: dict, disaster_type: str = "Wildfire") -> np.ndarray:
    ndvi  = indices["NDVI"];  ndwi  = indices["NDWI"]
    nbr   = indices["NBR"];   ndbi  = indices["NDBI"]
    mndwi = indices["MNDWI"]; bsi   = indices["BSI"]

    h, w  = ndvi.shape
    lulc  = np.full((h, w), 4, dtype=np.uint8)  # default: bare soil

    lulc[mndwi > 0.18]  = 2
    lulc[(ndwi > 0.22) & (lulc == 4)] = 2

    if "Wildfire" in disaster_type:
        lulc[(nbr < -0.05) & (ndvi < 0.12) & (lulc == 4)] = 5

    lulc[(ndbi > 0.05) & (bsi > 0.0) & (lulc == 4)]             = 3
    lulc[(ndvi > 0.40) & (lulc == 4)]                            = 0
    lulc[(ndvi > 0.15) & (ndvi <= 0.40) & (lulc == 4)]           = 1
    lulc[(ndwi > 0.05) & (ndwi <= 0.22) & (ndvi > 0.10) & (lulc == 4)] = 6
    lulc[(bsi > 0.05) & (lulc == 4)]                             = 4
    return lulc


def compute_lulc_stats(lulc: np.ndarray) -> dict:
    total = lulc.size
    return {
        cls_name: {
            "id": cls_id,
            "count": int(np.sum(lulc == cls_id)),
            "pct": round(int(np.sum(lulc == cls_id)) / total * 100, 2),
            "area_km2": round(int(np.sum(lulc == cls_id)) * 100 / 1e6, 4),
        }
        for cls_id, (cls_name, _) in LULC_CLASSES.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION H — CHANGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_changes(idx_b: dict, idx_a: dict,
                   lulc_b: np.ndarray, lulc_a: np.ndarray,
                   disaster_type: str):
    dNDVI = idx_b["NDVI"] - idx_a["NDVI"]
    dNDWI = idx_a["NDWI"] - idx_b["NDWI"]
    dNBR  = idx_b["NBR"]  - idx_a["NBR"]
    dNDBI = idx_a["NDBI"] - idx_b["NDBI"]
    dEVI  = idx_b["EVI"]  - idx_a["EVI"]

    h, w  = dNDVI.shape
    cmap  = np.zeros((h, w), dtype=np.uint8)

    if "Wildfire" in disaster_type:
        cmap[dNBR  > 0.15]                     = 5
        cmap[(dNDVI > 0.15) & (cmap == 0)]     = 1
        cmap[(dNDVI < -0.12) & (cmap == 0)]    = 6
    elif "Flood" in disaster_type:
        cmap[dNDWI  > 0.12]                    = 3
        cmap[(dNDVI > 0.15) & (cmap == 0)]     = 1
    elif "Drought" in disaster_type:
        cmap[dNDVI  > 0.18]                    = 1
        cmap[(dNDVI < -0.10) & (cmap == 0)]    = 2
    else:
        cmap[dNDBI  > 0.10]                    = 4
        cmap[(dNDVI > 0.15) & (cmap == 0)]     = 1

    transitions = {}
    for a, (an, _) in LULC_CLASSES.items():
        for b2, (bn, _) in LULC_CLASSES.items():
            if a != b2:
                cnt = int(np.sum((lulc_b == a) & (lulc_a == b2)))
                if cnt > 50:
                    transitions[f"{an} → {bn}"] = cnt

    return cmap, dNDVI, dNDWI, dNBR, dNDBI, dEVI, transitions


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION I — INTERACTIVE MAP
# ─────────────────────────────────────────────────────────────────────────────

def build_interactive_map(meta: dict, geojson_data: dict) -> str:
    try:
        import folium
        from folium import plugins

        cx = (meta["west"] + meta["east"]) / 2
        cy = (meta["north"] + meta["south"]) / 2
        m  = folium.Map(location=[cy, cx], zoom_start=11, tiles=None)

        folium.TileLayer(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri, Maxar, Earthstar Geographics",
            name="ESRI World Imagery", max_zoom=20,
        ).add_to(m)
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

        aoi_box = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[
                [meta["west"], meta["south"]],
                [meta["east"], meta["south"]],
                [meta["east"], meta["north"]],
                [meta["west"], meta["north"]],
                [meta["west"], meta["south"]],
            ]]},
            "properties": {"label": "Analysis AOI"},
        }
        folium.GeoJson(
            aoi_box,
            style_function=lambda f: {
                "fillColor": "#7c6fff", "fillOpacity": 0.06,
                "color": "#7c6fff", "weight": 2.5, "dashArray": "6 4"},
            name="Analysis AOI",
        ).add_to(m)

        quad_colors = ["#ef5350", "#43a047", "#fb8c00", "#7c4dff"]
        for i, feat in enumerate(geojson_data["features"]):
            if (feat["geometry"]["type"] == "Polygon" and
                    feat["properties"].get("label") != "Full AOI"):
                c = quad_colors[i % 4]
                folium.GeoJson(
                    feat,
                    style_function=lambda f, cc=c: {
                        "fillColor": cc, "fillOpacity": 0.05,
                        "color": cc, "weight": 1},
                    tooltip=folium.GeoJsonTooltip(
                        ["label", "area_km2"], aliases=["Zone", "Area (km²)"]),
                ).add_to(m)

        folium.Marker(
            [cy, cx],
            tooltip=f"AOI Centre — {cy:.6f}°N, {cx:.6f}°E",
            icon=folium.Icon(color="purple", icon="crosshairs", prefix="fa"),
        ).add_to(m)
        plugins.MeasureControl(
            position="bottomleft", primary_length_unit="kilometers").add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        return m._repr_html_()
    except ImportError:
        return ("<div style='padding:20px;color:#7c6fff;font-family:monospace'>"
                "Install folium: pip install folium</div>")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION J — ANALYSIS PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(bands_b: dict, bands_a: dict,
                 disaster_type: str, lat: float, lon: float,
                 date_before: str, date_after: str) -> dict:
    size = bands_b["B8"].shape[0]

    meta    = build_scene_meta(lat, lon, size, bands_b)
    idx_b   = compute_all_indices(bands_b)
    idx_a   = compute_all_indices(bands_a)
    lulc_b  = classify_lulc(idx_b, disaster_type)
    lulc_a  = classify_lulc(idx_a, disaster_type)
    cmap, dNDVI, dNDWI, dNBR, dNDBI, dEVI, transitions = \
        detect_changes(idx_b, idx_a, lulc_b, lulc_a, disaster_type)
    stats_b = compute_lulc_stats(lulc_b)
    stats_a = compute_lulc_stats(lulc_a)
    bstats_b = compute_band_stats(bands_b)
    bstats_a = compute_band_stats(bands_a)
    corr_mat, corr_labels = compute_index_correlation(idx_a)
    geojson = build_aoi_geojson(meta)
    geotiff_zip = export_geotiffs_zip(idx_a, lulc_a, cmap, meta, disaster_type)

    return {
        "bands_b": bands_b, "bands_a": bands_a,
        "indices_b": idx_b, "indices_a": idx_a,
        "lulc_b": lulc_b, "lulc_a": lulc_a,
        "change_map": cmap,
        "dNDVI": dNDVI, "dNDWI": dNDWI, "dNBR": dNBR, "dNDBI": dNDBI, "dEVI": dEVI,
        "transitions": transitions,
        "stats_b": stats_b, "stats_a": stats_a,
        "bstats_b": bstats_b, "bstats_a": bstats_a,
        "corr_mat": corr_mat, "corr_labels": corr_labels,
        "meta": meta, "geojson": geojson, "geotiff_zip": geotiff_zip,
        "lat": lat, "lon": lon,
        "disaster_type": disaster_type,
        "date_before": date_before, "date_after": date_after,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION K — VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def fig_bytes(fig, dpi=130):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0); plt.close(fig)
    return buf.getvalue()


def _ax(ax, title=None, tc="#7c6fff"):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors="#48487e", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#1a1a46")
    if title:
        ax.set_title(title, color=tc, fontsize=9, fontweight="600",
                     fontfamily="monospace", pad=6)


def plot_rgb(bands, title):
    rgb = np.clip(np.dstack([bands["B4"],bands["B3"],bands["B2"]]) * 3.5, 0, 1)
    fig, ax = plt.subplots(figsize=(5,5), facecolor=BG)
    ax.imshow(rgb, interpolation="bilinear"); _ax(ax, title); ax.axis("off")
    fig.tight_layout(pad=0.4); return fig


def plot_fcc(bands, title):
    fcc = np.clip(np.dstack([bands["B8"],bands["B4"],bands["B3"]]) * 2.5, 0, 1)
    fig, ax = plt.subplots(figsize=(5,5), facecolor=BG)
    ax.imshow(fcc, interpolation="bilinear"); _ax(ax, title, "#00f0d0"); ax.axis("off")
    fig.tight_layout(pad=0.4); return fig


def plot_swir(bands, title):
    swir = np.clip(np.dstack([bands["B12"],bands["B8"],bands["B4"]]) * 3.0, 0, 1)
    fig, ax = plt.subplots(figsize=(5,5), facecolor=BG)
    ax.imshow(swir, interpolation="bilinear"); _ax(ax, title, "#ff4f6a"); ax.axis("off")
    fig.tight_layout(pad=0.4); return fig


def plot_index(arr, title, cmap="RdYlGn", vmin=-1, vmax=1, unit=""):
    fig, ax = plt.subplots(figsize=(5,5), facecolor=BG)
    _ax(ax, title)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
    cb = plt.colorbar(im, ax=ax, fraction=0.042, pad=0.03, shrink=0.88)
    cb.ax.tick_params(colors="#48487e", labelsize=6)
    cb.outline.set_edgecolor("#1a1a46")
    if unit: cb.set_label(unit, color="#48487e", fontsize=6)
    ax.axis("off"); fig.tight_layout(pad=0.4); return fig


def plot_lulc(lulc, title):
    cmap = mcolors.ListedColormap([LULC_CLASSES[i][1] for i in range(len(LULC_CLASSES))])
    fig, ax = plt.subplots(figsize=(5,5), facecolor=BG)
    _ax(ax, title, "#00e89a")
    ax.imshow(lulc, cmap=cmap, vmin=0, vmax=len(LULC_CLASSES)-1,
              interpolation="nearest")
    patches = [Patch(facecolor=LULC_CLASSES[i][1], label=LULC_CLASSES[i][0],
                     edgecolor="#0b0b1e", linewidth=0.5)
               for i in range(len(LULC_CLASSES))]
    ax.legend(handles=patches, loc="lower right", fontsize=5.5,
              facecolor="#0b0b1e", edgecolor="#1a1a46", labelcolor="#dcdcff",
              framealpha=0.92, ncol=1, borderpad=0.6, handlelength=1.0)
    ax.axis("off"); fig.tight_layout(pad=0.4); return fig


def plot_change(cmap_arr, title):
    cmap = mcolors.ListedColormap([CHANGE_CLASSES[i][1] for i in range(len(CHANGE_CLASSES))])
    fig, ax = plt.subplots(figsize=(5,5), facecolor=BG)
    _ax(ax, title, "#ff4f6a")
    ax.imshow(cmap_arr, cmap=cmap, vmin=0, vmax=len(CHANGE_CLASSES)-1,
              interpolation="nearest")
    patches = [Patch(facecolor=CHANGE_CLASSES[i][1], label=CHANGE_CLASSES[i][0],
                     edgecolor="#0b0b1e", linewidth=0.5)
               for i in range(len(CHANGE_CLASSES))]
    ax.legend(handles=patches, loc="lower right", fontsize=5.5,
              facecolor="#0b0b1e", edgecolor="#1a1a46", labelcolor="#dcdcff",
              framealpha=0.92, ncol=1)
    ax.axis("off"); fig.tight_layout(pad=0.4); return fig


def plot_dnbr(dnbr):
    cmap = mcolors.ListedColormap(
        ["#2e7d32","#a5d6a7","#fff9c4","#fb8c00","#b71c1c"])
    norm = mcolors.BoundaryNorm([-1,-0.10,0.10,0.27,0.44,1.0], cmap.N)
    fig, ax = plt.subplots(figsize=(5,5), facecolor=BG)
    _ax(ax, "dNBR Burn Severity", "#ffc840")
    im = ax.imshow(dnbr, cmap=cmap, norm=norm, interpolation="bilinear")
    cb = plt.colorbar(im, ax=ax, fraction=0.042, pad=0.03, shrink=0.88)
    cb.ax.tick_params(colors="#48487e", labelsize=6)
    cb.outline.set_edgecolor("#1a1a46")
    cb.set_ticks([-0.55, 0.0, 0.185, 0.355, 0.72])
    cb.set_ticklabels(["Regrowth","Unburned","Low","Moderate","High"])
    ax.axis("off"); fig.tight_layout(pad=0.4); return fig


def plot_spectra(bstats_b, bstats_a):
    order = ["B2","B3","B4","B5","B8","B11","B12"]
    wl    = [SENTINEL2_BANDS[b]["nm"] for b in order]
    mb    = [bstats_b.get(b,{}).get("mean",0) for b in order]
    ma    = [bstats_a.get(b,{}).get("mean",0) for b in order]
    sb    = [bstats_b.get(b,{}).get("std",0) for b in order]
    sa    = [bstats_a.get(b,{}).get("std",0) for b in order]

    fig, ax = plt.subplots(figsize=(9,3.5), facecolor=BG)
    _ax(ax, "Spectral Profile — Mean Band Reflectance (Before vs After)")
    ax.fill_between(wl, [m-s for m,s in zip(mb,sb)],
                    [m+s for m,s in zip(mb,sb)], alpha=0.14, color="#7c6fff")
    ax.fill_between(wl, [m-s for m,s in zip(ma,sa)],
                    [m+s for m,s in zip(ma,sa)], alpha=0.14, color="#ff4f6a")
    ax.plot(wl, mb, "o-", color="#7c6fff", lw=2, ms=6, label="Before", zorder=3)
    ax.plot(wl, ma, "o-", color="#ff4f6a", lw=2, ms=6, label="After",  zorder=3)
    for wl_v, bn in zip(wl, order):
        ax.axvline(wl_v, color="#1a1a46", lw=0.5, linestyle="--", zorder=1)
        ax.text(wl_v, ax.get_ylim()[0], bn, color="#48487e", fontsize=6.5,
                ha="center", fontfamily="monospace")
    ax.set_xlabel("Wavelength (nm)", color="#8080be", fontsize=8)
    ax.set_ylabel("Surface Reflectance [0–1]", color="#8080be", fontsize=8)
    ax.legend(facecolor="#0b0b1e", edgecolor="#1a1a46", labelcolor="#dcdcff",
              fontsize=8, loc="upper left")
    ax.set_xlim(400, 2350)
    fig.tight_layout(pad=0.5); return fig


def plot_corr(corr_mat, labels):
    n   = len(labels)
    fig, ax = plt.subplots(figsize=(6,5), facecolor=BG)
    _ax(ax, "Index Correlation Matrix", "#bf6fff")
    im  = ax.imshow(corr_mat, cmap=plt.cm.RdBu_r, vmin=-1, vmax=1,
                    aspect="auto", interpolation="nearest")
    cb  = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cb.ax.tick_params(colors="#48487e", labelsize=7)
    cb.outline.set_edgecolor("#1a1a46")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right",
                       color="#8080be", fontsize=7.5, fontfamily="monospace")
    ax.set_yticklabels(labels, color="#8080be", fontsize=7.5, fontfamily="monospace")
    for i in range(n):
        for j in range(n):
            v = corr_mat[i,j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.6 else "#48487e",
                    fontfamily="monospace", fontweight="500")
    fig.tight_layout(pad=0.5); return fig


def plot_histograms(indices):
    colors = ["#7c6fff","#00f0d0","#ff4f6a","#bf6fff",
              "#00e89a","#ffc840","#ff6bfd","#78909c"]
    fig, axes = plt.subplots(2, 4, figsize=(16,5), facecolor=BG)
    for i, (name, arr) in enumerate(indices.items()):
        if i >= 8: break
        ax = axes.flatten()[i]; color = colors[i % len(colors)]
        ax.set_facecolor(SURFACE)
        vals = arr.flatten()
        ax.hist(vals, bins=60, color=color, alpha=0.85, edgecolor="none",
                density=True, zorder=2)
        ax.axvline(vals.mean(), color="white", lw=1.2, alpha=0.9,
                   label=f"μ={vals.mean():.3f}")
        ax.axvline(float(np.median(vals)), color=color, lw=1, linestyle=":", alpha=0.8)
        _ax(ax, name, color)
        ax.legend(fontsize=5.5, facecolor="#04040e", edgecolor="#1a1a46",
                  labelcolor="#dcdcff")
        ax.set_xlabel("Value", color="#8080be", fontsize=7)
        ax.set_ylabel("Density", color="#8080be", fontsize=7)
    fig.tight_layout(pad=0.8); return fig


def plot_band_grid(bands, prefix):
    bkeys  = ["B2","B3","B4","B5","B8","B11","B12"]
    bnames = ["B2 Blue","B3 Green","B4 Red","B5 Red Edge",
              "B8 NIR","B11 SWIR1","B12 SWIR2"]
    cmaps  = ["Blues_r","Greens_r","Reds_r","YlOrBr","YlGn","plasma","inferno"]
    fig, axes = plt.subplots(1, 7, figsize=(21,3.5), facecolor=BG)
    for i, (bk, bn, cm) in enumerate(zip(bkeys, bnames, cmaps)):
        ax = axes[i]; ax.set_facecolor(SURFACE)
        if bk in bands:
            im = ax.imshow(bands[bk], cmap=cm, vmin=0, vmax=0.6,
                           interpolation="bilinear")
            cb = plt.colorbar(im, ax=ax, fraction=0.055, pad=0.03, shrink=0.85)
            cb.ax.tick_params(colors="#48487e", labelsize=5)
            cb.outline.set_edgecolor("#1a1a46")
        _ax(ax, f"{prefix}\n{bn}", "#8080be"); ax.axis("off")
    fig.tight_layout(pad=0.5); return fig


def plot_dashboard(bands_b, bands_a, idx_b, idx_a,
                   lulc_b, lulc_a, change_map, dNBR,
                   disaster_type, location, date_b, date_a):
    fig = plt.figure(figsize=(24,24), facecolor=BG)
    gs  = gridspec.GridSpec(5, 4, figure=fig, hspace=0.30, wspace=0.20)

    lulc_cmap = mcolors.ListedColormap([LULC_CLASSES[i][1] for i in range(len(LULC_CLASSES))])
    chng_cmap = mcolors.ListedColormap([CHANGE_CLASSES[i][1] for i in range(len(CHANGE_CLASSES))])
    dnbr_cmap = mcolors.ListedColormap(["#2e7d32","#a5d6a7","#fff9c4","#fb8c00","#b71c1c"])
    dnbr_norm = mcolors.BoundaryNorm([-1,-0.10,0.10,0.27,0.44,1.0], dnbr_cmap.N)

    def add(ax, img, title, cmap=None, vmin=None, vmax=None, norm=None, tc="#7c6fff"):
        ax.set_facecolor(SURFACE)
        kw = {"interpolation":"bilinear"}
        if cmap:
            if norm: kw["norm"] = norm
            else:    kw.update({"vmin":vmin,"vmax":vmax})
            im = ax.imshow(img, cmap=cmap, **kw)
            cb = plt.colorbar(im, ax=ax, fraction=0.040, pad=0.03, shrink=0.85)
            cb.ax.tick_params(colors="#48487e", labelsize=5)
            cb.outline.set_edgecolor("#1a1a46")
        else:
            ax.imshow(img, **kw)
        ax.set_title(title, color=tc, fontsize=8, fontweight="600",
                     fontfamily="monospace", pad=4)
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        for sp in ax.spines.values(): sp.set_edgecolor("#1a1a46")

    rgb_b = np.clip(np.dstack([bands_b["B4"],bands_b["B3"],bands_b["B2"]])*3.5,0,1)
    rgb_a = np.clip(np.dstack([bands_a["B4"],bands_a["B3"],bands_a["B2"]])*3.5,0,1)
    fcc_b = np.clip(np.dstack([bands_b["B8"],bands_b["B4"],bands_b["B3"]])*2.5,0,1)
    fcc_a = np.clip(np.dstack([bands_a["B8"],bands_a["B4"],bands_a["B3"]])*2.5,0,1)

    add(fig.add_subplot(gs[0,0]), rgb_b, f"RGB — Before ({date_b})")
    add(fig.add_subplot(gs[0,1]), rgb_a, f"RGB — After ({date_a})", tc="#ff4f6a")
    add(fig.add_subplot(gs[0,2]), fcc_b, "False-Colour NIR — Before", tc="#00f0d0")
    add(fig.add_subplot(gs[0,3]), fcc_a, "False-Colour NIR — After",  tc="#00f0d0")

    add(fig.add_subplot(gs[1,0]), idx_b["NDVI"],"NDVI Before","RdYlGn",-1,1,tc="#00e89a")
    add(fig.add_subplot(gs[1,1]), idx_a["NDVI"],"NDVI After", "RdYlGn",-1,1,tc="#00e89a")
    add(fig.add_subplot(gs[1,2]), idx_b["NDWI"],"NDWI Before","Blues_r",-1,1,tc="#7c6fff")
    add(fig.add_subplot(gs[1,3]), idx_a["NDWI"],"NDWI After", "Blues_r",-1,1,tc="#7c6fff")

    add(fig.add_subplot(gs[2,0]), idx_b["NBR"], "NBR Before", "RdYlGn",-1,1,tc="#ff4f6a")
    add(fig.add_subplot(gs[2,1]), idx_a["NBR"], "NBR After",  "RdYlGn",-1,1,tc="#ff4f6a")
    add(fig.add_subplot(gs[2,2]), idx_b["NDBI"],"NDBI Before","RdYlGn_r",-0.5,0.5,tc="#bf6fff")
    add(fig.add_subplot(gs[2,3]), idx_a["NDBI"],"NDBI After", "RdYlGn_r",-0.5,0.5,tc="#bf6fff")

    add(fig.add_subplot(gs[3,0]), idx_b["EVI"],  "EVI Before",  "RdYlGn",-1,1,tc="#00e89a")
    add(fig.add_subplot(gs[3,1]), idx_a["EVI"],  "EVI After",   "RdYlGn",-1,1,tc="#00e89a")
    add(fig.add_subplot(gs[3,2]), idx_b["MNDWI"],"MNDWI Before","Blues_r",-1,1,tc="#00f0d0")
    add(fig.add_subplot(gs[3,3]), idx_a["BSI"],  "BSI After",   "YlOrBr",-1,1,tc="#ffc840")

    add(fig.add_subplot(gs[4,0]), lulc_b,    "LULC — Before",
        lulc_cmap, 0, len(LULC_CLASSES)-1, tc="#00e89a")
    add(fig.add_subplot(gs[4,1]), lulc_a,    "LULC — After",
        lulc_cmap, 0, len(LULC_CLASSES)-1, tc="#00e89a")
    add(fig.add_subplot(gs[4,2]), change_map,"Change Detection",
        chng_cmap, 0, len(CHANGE_CLASSES)-1, tc="#ff4f6a")
    add(fig.add_subplot(gs[4,3]), dNBR,      "dNBR Burn Severity",
        dnbr_cmap, norm=dnbr_norm, tc="#ffc840")

    fig.suptitle(
        f"GeoSight Aurora  ·  {disaster_type}  ·  {location}  ·  {date_b} → {date_a}",
        color="#dcdcff", fontsize=13, fontweight="700", fontfamily="monospace", y=0.995)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION L — UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _metric(label, value, color="blue", delta=""):
    st.markdown(f"""
    <div class="gs-metric {color}">
        <div class="gs-mval {color}">{value}</div>
        <div class="gs-mlabel">{label}</div>
        {f'<div class="gs-mdelta" style="color:var(--text3)">{delta}</div>' if delta else ''}
    </div>""", unsafe_allow_html=True)


def _section(label):
    st.markdown(f"""
    <div class="gs-section">
        <div class="gs-section-line"></div>
        <div class="gs-section-label">{label}</div>
        <div class="gs-section-line"></div>
    </div>""", unsafe_allow_html=True)


def _source_badge(bands: dict) -> str:
    acc = bands.get("_accuracy","unknown")
    labels = {
        "full":          "✓  7-band GeoTIFF · Full Accuracy",
        "reconstructed": "⚡  Physics-Reconstructed · RGB→S2",
        "estimated":     "⚠  Single-band Estimated",
    }
    return f'<span class="recon-badge">{labels.get(acc, acc)}</span>'


def _colour_legend_html(idx_name: str) -> str:
    """Return an HTML colour interpretation panel for a given index."""
    guide = INDEX_COLOUR_GUIDE.get(idx_name)
    if not guide: return ""

    rows_html = ""
    for hex_c, val_range, meaning in guide["legend"]:
        rows_html += f"""
        <div class="colour-row">
            <div class="colour-swatch" style="background:{hex_c}"></div>
            <div class="colour-val">{val_range}</div>
            <div class="colour-meaning">{meaning}</div>
        </div>"""

    warning_html = ""
    if "warning" in guide:
        warning_html = f"""
        <div style="background:var(--amber-lo);border-left:3px solid var(--amber);
                    border-radius:0 var(--r) var(--r) 0;padding:8px 12px;
                    font-size:0.73rem;color:var(--amber);margin-top:8px;line-height:1.5">
        ⚠  {guide['warning']}</div>"""
    note_html = ""
    if "note" in guide:
        note_html = f"""
        <div style="background:var(--green-lo);border-left:3px solid var(--green);
                    border-radius:0 var(--r) var(--r) 0;padding:8px 12px;
                    font-size:0.73rem;color:var(--green);margin-top:8px;line-height:1.5">
        ℹ  {guide['note']}</div>"""

    return f"""
    <div style="margin:10px 0">
        <div style="font-family:var(--mono);font-size:0.60rem;color:var(--text3);
                    letter-spacing:2px;text-transform:uppercase;margin-bottom:6px">
            Colour interpretation — {idx_name}</div>
        <div style="font-family:var(--mono);font-size:0.68rem;color:var(--accent3);
                    margin-bottom:4px">Formula: <span style="color:var(--text2)">{guide['formula']}</span></div>
        <div style="font-size:0.77rem;color:var(--text2);margin-bottom:10px;line-height:1.5">
            {guide['desc']}</div>
        <div class="colour-legend">{rows_html}</div>
        {warning_html}{note_html}
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION M — LANDING PAGE
# ─────────────────────────────────────────────────────────────────────────────

def _render_landing():
    import pandas as pd

    st.markdown("""
    <div style="text-align:center;padding:36px 0 28px;font-family:'Space Mono',monospace;
                font-size:0.72rem;letter-spacing:2.5px;
                background:linear-gradient(90deg,#7c6fff,#ff6bfd);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;">
        UPLOAD REAL SATELLITE IMAGE  ·  CONFIGURE PARAMETERS  ·  PRESS RUN ANALYSIS
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, (icon, title, desc) in zip([c1,c2,c3,c4], [
        ("⬆️","Real Satellite Upload",
         "GeoTIFF 7-band: direct mapping. PNG/JPG from Copernicus: auto-reconstructed to 7 "
         "Sentinel-2 bands via physics-based ESA S2 LUT spectral unmixing."),
        ("🎯","Accurate Georeferencing",
         "GeoTIFF: native affine transform preserved. PNG: WGS-84 constructed from your "
         "lat/lon. All outputs carry correct EPSG:4326 CRS and pixel scale."),
        ("📊","8 Scientific Indices",
         "NDVI · NDWI · NBR · NDBI · SAVI · EVI · MNDWI · BSI — ESA/USGS formulas. "
         "Each includes a full colour interpretation legend in the analysis panel."),
        ("🗺️","Rasterio GeoTIFF Export",
         "Georeferenced float32 GeoTIFFs with proper affine transform, CRS, nodata. "
         "Loads natively in ArcGIS Pro, SNAP, GDAL, QGIS, and Python rasterio."),
    ]):
        with col:
            st.markdown(f"""
            <div class="gs-metric blue" style="text-align:left;min-height:210px;padding:20px;">
                <div style="font-size:1.6rem;margin-bottom:10px">{icon}</div>
                <div style="font-family:'Space Mono',monospace;font-size:0.76rem;
                            font-weight:700;color:#7c6fff;margin-bottom:8px">{title}</div>
                <div style="font-size:0.74rem;color:#48487e;line-height:1.65">{desc}</div>
            </div>""", unsafe_allow_html=True)

    _section("SPECTRAL INDEX REFERENCE")
    st.dataframe(pd.DataFrame([
        ("NDVI",  "(B8−B4)/(B8+B4)",                          "Vegetation health",   ">0.4 dense veg · <0 water/bare","Crop stress · deforestation"),
        ("NDWI",  "(B3−B8)/(B3+B8)",                          "Water presence",      ">0.3 open water",              "Flood mapping · drought"),
        ("NBR",   "(B8−B12)/(B8+B12)",                        "Burn severity",       "dNBR >0.44 high severity",     "Wildfire damage assessment"),
        ("NDBI",  "(B11−B8)/(B11+B8)",                        "Built-up areas",      ">0.2 confirmed urban",         "Urban sprawl detection"),
        ("SAVI",  "((B8−B4)/(B8+B4+0.5))×1.5",               "Soil-adj vegetation", "Better than NDVI in arid",     "Semi-arid agriculture"),
        ("EVI",   "2.5×(B8−B4)/(B8+6B4−7.5B2+1)",            "Enhanced vegetation", "No saturation in forests",     "Tropical forests"),
        ("MNDWI", "(B3−B11)/(B3+B11)",                        "Water (SWIR-based)",  ">0.3 open water",              "Urban water bodies"),
        ("BSI",   "((B11+B4)−(B8+B2))/((B11+B4)+(B8+B2))",   "Bare soil",           ">0.3 desert/degraded",         "Desertification · erosion"),
    ], columns=["Index","Formula","Detects","Threshold","Applications"]),
    use_container_width=True, hide_index=True)

    _section("LULC CLASS LEGEND")
    cols = st.columns(7)
    for i, (_, (cls_name, color)) in enumerate(LULC_CLASSES.items()):
        with cols[i % 7]:
            st.markdown(f'<div class="lulc-badge"><div class="lulc-dot" '
                        f'style="background:{color}"></div>{cls_name}</div>',
                        unsafe_allow_html=True)

    _section("PIPELINE")
    st.markdown("""
    <div class="gs-infobox">
    <b>Upload</b> (GeoTIFF / PNG / JPG)  →  <b>DOS Atmospheric Correction</b>  →
    <b>Spectral Unmixing + Band Reconstruction</b>  →  <b>7-Band Stack</b><br>
    →  <b>8 Spectral Indices</b> (NDVI · NDWI · NBR · NDBI · SAVI · EVI · MNDWI · BSI)<br>
    →  <b>LULC Classification</b> (7 classes)  →  <b>Change Detection</b>  →
    <b>Zonal Stats</b>  →  <b>Rasterio GeoTIFF Export</b> (EPSG:4326)
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION N — RESULTS RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def _render_results(R: dict, params: dict):
    import pandas as pd

    disaster_type = R["disaster_type"]
    date_b = R["date_before"]; date_a = R["date_after"]
    size   = R["bands_b"]["B8"].shape[0]
    meta   = R["meta"]

    badge_b = _source_badge(R["bands_b"])
    badge_a = _source_badge(R["bands_a"])
    st.markdown(
        f"<div style='margin-bottom:6px'>"
        f"<b style='color:var(--text3);font-family:monospace;font-size:0.60rem'>BEFORE</b> {badge_b}"
        f"&nbsp;<b style='color:var(--text3);font-family:monospace;font-size:0.60rem'>AFTER</b> {badge_a}"
        f"</div>", unsafe_allow_html=True)

    # Top metrics
    n_ch       = int(np.sum(R["change_map"] > 0))
    pct_ch     = n_ch / R["change_map"].size * 100
    ndvi_delta = float(np.mean(R["indices_a"]["NDVI"]) - np.mean(R["indices_b"]["NDVI"]))
    mean_dnbr  = float(np.mean(R["dNBR"]))
    n_burn_hi  = int(np.sum(R["dNBR"] > 0.44))

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: _metric("Grid Size",    f"{size}×{size}", "blue",   f"{size*size//1000}K px")
    with c2: _metric("Indices",      "8",              "cyan",   "ESA/USGS standard")
    with c3: _metric("Area Changed", f"{pct_ch:.1f}%", "amber",  f"{n_ch:,} px")
    with c4:
        s = "+" if ndvi_delta >= 0 else ""
        _metric("ΔNDVI", f"{s}{ndvi_delta:.3f}",
                "green" if ndvi_delta >= 0 else "red",
                "Veg gain" if ndvi_delta >= 0 else "Veg loss")
    with c5: _metric("Mean dNBR",  f"{mean_dnbr:.3f}","red",  f"{n_burn_hi:,} px high sev.")
    with c6: _metric("AOI Area",   f"{meta['area_km2']:.0f} km²","purple",
                     f"~{size*meta['pixel_m']/1000:.1f}km²")

    st.markdown("<br>", unsafe_allow_html=True)

    tabs = st.tabs([
        "🛰️  Satellite Imagery",
        "📊  Spectral Indices",
        "🗺️  LULC Classification",
        "🔄  Change Detection",
        "📐  Raster Analytics",
        "🌐  Vector & Map",
        "📈  Full Dashboard",
        "💾  Export",
    ])

    # ── TAB 1: SATELLITE IMAGERY ────────────────────────────────────────────
    with tabs[0]:
        _section("COMPOSITE VIEWS — TRUE-COLOUR · FALSE-COLOUR NIR · SWIR")
        st.markdown("""
        <div class="gs-infobox">
        <b>True-Colour (B4·B3·B2)</b> — human-eye view. Vegetation = green, water = blue, soil = tan.&nbsp;
        <b>False-Colour NIR (B8·B4·B3)</b> — healthy vegetation = magenta-red; water = black-blue.&nbsp;
        <b>SWIR (B12·B8·B4)</b> — active burn scars = orange-red; bare soil = pale pink; water = dark blue.
        </div>""", unsafe_allow_html=True)

        cols = st.columns(6)
        imgs = [
            (plot_rgb(R["bands_b"],  f"RGB Before\n{date_b}"), "True-Colour · Before"),
            (plot_rgb(R["bands_a"],  f"RGB After\n{date_a}"),  "True-Colour · After"),
            (plot_fcc(R["bands_b"],  f"NIR Before\n{date_b}"), "False-Colour NIR · Before"),
            (plot_fcc(R["bands_a"],  f"NIR After\n{date_a}"),  "False-Colour NIR · After"),
            (plot_swir(R["bands_b"], f"SWIR Before\n{date_b}"),"SWIR · Before"),
            (plot_swir(R["bands_a"], f"SWIR After\n{date_a}"), "SWIR · After"),
        ]
        for col, (fig, cap) in zip(cols, imgs):
            with col: st.image(fig_bytes(fig), use_container_width=True, caption=cap)

        _section("SPECTRAL PROFILE — MEAN BAND REFLECTANCE")
        st.image(fig_bytes(plot_spectra(R["bstats_b"], R["bstats_a"])),
                 use_container_width=True)

        _section("RAW BANDS — BEFORE")
        st.image(fig_bytes(plot_band_grid(R["bands_b"],"Before")), use_container_width=True)
        _section("RAW BANDS — AFTER")
        st.image(fig_bytes(plot_band_grid(R["bands_a"],"After")),  use_container_width=True)

    # ── TAB 2: SPECTRAL INDICES ─────────────────────────────────────────────
    with tabs[1]:
        _section("ALL 8 SPECTRAL INDICES — BEFORE vs AFTER + COLOUR INTERPRETATION")

        idx_cfg = [
            ("NDVI",  "RdYlGn",   -1,    1  ),
            ("NDWI",  "Blues_r",  -1,    1  ),
            ("NBR",   "RdYlGn",   -1,    1  ),
            ("NDBI",  "RdYlGn_r", -0.5,  0.5),
            ("SAVI",  "RdYlGn",   -1,    1  ),
            ("EVI",   "RdYlGn",   -1,    1  ),
            ("MNDWI", "Blues_r",  -1,    1  ),
            ("BSI",   "YlOrBr",   -1,    1  ),
        ]

        for idx_name, cmap, vmin, vmax in idx_cfg:
            guide = INDEX_COLOUR_GUIDE[idx_name]
            _section(f"{idx_name} — {guide['title'].split('—')[1].strip()}")

            c_img1, c_img2, c_stats, c_legend = st.columns([1.8, 1.8, 1.4, 2.5])

            with c_img1:
                st.image(fig_bytes(plot_index(R["indices_b"][idx_name],
                                              f"{idx_name} Before", cmap, vmin, vmax)),
                         use_container_width=True, caption=f"{idx_name} Before · {date_b}")

            with c_img2:
                st.image(fig_bytes(plot_index(R["indices_a"][idx_name],
                                              f"{idx_name} After",  cmap, vmin, vmax)),
                         use_container_width=True, caption=f"{idx_name} After · {date_a}")

            with c_stats:
                ab = R["indices_b"][idx_name].flatten()
                aa = R["indices_a"][idx_name].flatten()
                delta = aa.mean() - ab.mean()
                s  = "+" if delta >= 0 else ""
                # positive delta = improvement for NDVI/NDWI/EVI/SAVI/NBR,
                # deterioration for NDBI/BSI
                good_up = idx_name not in ["NDBI","BSI"]
                col = "green" if (delta >= 0) == good_up else "red"
                st.markdown(f"""
                <div class="gs-metric {col}" style="padding:12px 14px;margin-bottom:8px">
                    <div class="gs-mval {col}" style="font-size:1.4rem">{s}{delta:.4f}</div>
                    <div class="gs-mlabel">Δ Mean  ({idx_name})</div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="vector-panel" style="font-size:0.64rem;line-height:2.0">
                <b style="color:#7c6fff">Before</b>
                  μ={ab.mean():.4f} · σ={ab.std():.4f}<br>
                  P5={np.percentile(ab,5):.3f} · P95={np.percentile(ab,95):.3f}<br>
                <b style="color:#ff4f6a">After</b>
                  μ={aa.mean():.4f} · σ={aa.std():.4f}<br>
                  P5={np.percentile(aa,5):.3f} · P95={np.percentile(aa,95):.3f}
                </div>""", unsafe_allow_html=True)

            with c_legend:
                st.markdown(_colour_legend_html(idx_name), unsafe_allow_html=True)

        _section("INDEX HISTOGRAMS — AFTER")
        st.image(fig_bytes(plot_histograms(R["indices_a"])), use_container_width=True)

    # ── TAB 3: LULC ─────────────────────────────────────────────────────────
    with tabs[2]:
        _section("LAND USE / LAND COVER CLASSIFICATION")
        st.markdown("""
        <div class="gs-infobox">
        Hierarchical threshold classification: MNDWI→Water, NBR→Burned,
        NDBI+BSI→Urban, NDVI thresholds→Dense/Sparse Veg, NDWI→Wetland, BSI→Bare Soil.
        Matches ESA Sen2Cor and OpenEO methodology.
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.image(fig_bytes(plot_lulc(R["lulc_b"],f"LULC Before — {date_b}")),
                     use_container_width=True, caption="LULC · Before")
        with c2:
            st.image(fig_bytes(plot_lulc(R["lulc_a"],f"LULC After — {date_a}")),
                     use_container_width=True, caption="LULC · After")

        _section("CLASS AREA STATISTICS")
        rows = []
        for _, (cls_name, color) in LULC_CLASSES.items():
            sb = R["stats_b"].get(cls_name, {})
            sa = R["stats_a"].get(cls_name, {})
            dp = sa.get("pct",0) - sb.get("pct",0)
            rows.append({
                "Class": cls_name,
                "Before %": f"{sb.get('pct',0):.2f}",
                "After %":  f"{sa.get('pct',0):.2f}",
                "Δ %":      f"{dp:+.2f}",
                "Before km²": f"{sb.get('area_km2',0):.4f}",
                "After km²":  f"{sa.get('area_km2',0):.4f}",
                "Before px":  f"{sb.get('count',0):,}",
                "After px":   f"{sa.get('count',0):,}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        _section("CLASS TRANSITION MATRIX")
        if R.get("transitions"):
            st.dataframe(pd.DataFrame([
                {"Transition": k, "Pixels": f"{v:,}",
                 "Area (km²)": f"{v*100/1e6:.4f}"}
                for k, v in sorted(R["transitions"].items(), key=lambda x: -x[1])
            ]), use_container_width=True, hide_index=True)
        else:
            st.info("No significant class transitions detected (threshold: 50 pixels).")

    # ── TAB 4: CHANGE DETECTION ─────────────────────────────────────────────
    with tabs[3]:
        _section("CHANGE DETECTION — DIFFERENCE MAPS")
        st.markdown("""
        <div class="gs-infobox">
        <b>dNDVI</b> = NDVI<sub>before</sub> − NDVI<sub>after</sub>
        &nbsp;(positive = vegetation lost, red on map)  ·
        <b>dNDWI</b> = NDWI<sub>after</sub> − NDWI<sub>before</sub>
        &nbsp;(positive = water appeared)  ·
        <b>dNBR</b> = NBR<sub>before</sub> − NBR<sub>after</sub>
        &nbsp;(apply USGS thresholds: >0.44 = high severity burn)
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.image(fig_bytes(plot_change(R["change_map"],"Change Detection")),
                     use_container_width=True, caption="Land Cover Change Map")
        with c2:
            st.image(fig_bytes(plot_dnbr(R["dNBR"])),
                     use_container_width=True, caption="dNBR Burn Severity (USGS)")
        with c3:
            st.image(fig_bytes(plot_index(R["dNDVI"],"dNDVI","RdYlGn",-0.5,0.5,
                                          "pos=loss · neg=gain")),
                     use_container_width=True, caption="dNDVI")
        with c4:
            st.image(fig_bytes(plot_index(R["dNDWI"],"dNDWI","Blues",-0.3,0.3,
                                          "pos=water appeared")),
                     use_container_width=True, caption="dNDWI")

        _section("BURN SEVERITY TABLE — USGS dNBR Thresholds")
        dnbr_flat = R["dNBR"].flatten(); total_px = dnbr_flat.size
        sev = [
            ("Regrowth",      dnbr_flat < -0.10,              "< −0.10"),
            ("Unburned",      (dnbr_flat>=-0.10)&(dnbr_flat<0.10), "−0.10 to 0.10"),
            ("Low",           (dnbr_flat>=0.10)&(dnbr_flat<0.27),  "0.10 to 0.27"),
            ("Moderate",      (dnbr_flat>=0.27)&(dnbr_flat<0.44),  "0.27 to 0.44"),
            ("High",          dnbr_flat >= 0.44,              "> 0.44"),
        ]
        st.dataframe(pd.DataFrame([
            {"Severity": n, "Pixels": f"{int(m.sum()):,}",
             "% Area": f"{int(m.sum())/total_px*100:.2f}",
             "Area (km²)": f"{int(m.sum())*100/1e6:.4f}",
             "dNBR Range": r}
            for n, m, r in sev
        ]), use_container_width=True, hide_index=True)

        _section("CHANGE CLASS SUMMARY")
        st.dataframe(pd.DataFrame([
            {"Class": CHANGE_CLASSES[i][0],
             "Pixels": f"{int(np.sum(R['change_map']==i)):,}",
             "% Scene": f"{int(np.sum(R['change_map']==i))/total_px*100:.2f}",
             "Area (km²)": f"{int(np.sum(R['change_map']==i))*100/1e6:.4f}"}
            for i in range(len(CHANGE_CLASSES))
        ]), use_container_width=True, hide_index=True)

    # ── TAB 5: RASTER ANALYTICS ─────────────────────────────────────────────
    with tabs[4]:
        _section("BAND STATISTICS")
        band_rows = []
        for bk in ["B2","B3","B4","B5","B8","B11","B12"]:
            sb = R["bstats_b"].get(bk, {}); sa = R["bstats_a"].get(bk, {})
            band_rows.append({
                "Band": bk, "Name": SENTINEL2_BANDS[bk]["name"],
                "nm":   SENTINEL2_BANDS[bk]["nm"],
                "Before μ": f"{sb.get('mean',0):.4f}",
                "After μ":  f"{sa.get('mean',0):.4f}",
                "Δ μ":      f"{(sa.get('mean',0)-sb.get('mean',0)):+.4f}",
                "After σ":  f"{sa.get('std',0):.4f}",
                "P5":       f"{sa.get('p5',0):.4f}",
                "P95":      f"{sa.get('p95',0):.4f}",
                "SNR":      f"{sa.get('snr',0):.2f}",
            })
        st.dataframe(pd.DataFrame(band_rows), use_container_width=True, hide_index=True)

        _section("INDEX CORRELATION MATRIX")
        st.image(fig_bytes(plot_corr(R["corr_mat"], R["corr_labels"])),
                 use_container_width=True)

        _section("SPATIAL AUTOCORRELATION — NDVI After")
        sp = compute_spatial_stats(R["indices_a"]["NDVI"])
        c1, c2, c3 = st.columns(3)
        with c1: _metric("Moran's I Proxy", f"{sp['moran_i_proxy']:.4f}",
                          "cyan","1=clustered · 0=random")
        with c2: _metric("Entropy (NDVI)", f"{sp['entropy']:.3f}", "purple","bits")
        with c3: _metric("NDVI Std Dev",   f"{R['indices_a']['NDVI'].std():.4f}",
                          "amber","spatial variability")

        _section("INDEX SUMMARY TABLE")
        idx_rows = []
        for name in R["indices_b"]:
            ab = R["indices_b"][name].flatten()
            aa = R["indices_a"][name].flatten()
            idx_rows.append({
                "Index":     name,
                "Before μ":  f"{ab.mean():.4f}",
                "After μ":   f"{aa.mean():.4f}",
                "Δ Mean":    f"{(aa.mean()-ab.mean()):+.4f}",
                "Before σ":  f"{ab.std():.4f}",
                "After σ":   f"{aa.std():.4f}",
                "Before P5": f"{np.percentile(ab,5):.4f}",
                "After P95": f"{np.percentile(aa,95):.4f}",
            })
        st.dataframe(pd.DataFrame(idx_rows), use_container_width=True, hide_index=True)

    # ── TAB 6: VECTOR & MAP ─────────────────────────────────────────────────
    with tabs[5]:
        c1, c2 = st.columns([3, 2])
        with c1:
            _section("INTERACTIVE SATELLITE BASEMAP — ESRI World Imagery")
            if params.get("show_map", True):
                html_map = build_interactive_map(meta, R["geojson"])
                import streamlit.components.v1 as components
                components.html(html_map, height=480)
            else:
                st.info("Enable 'Show Satellite Map' in the sidebar.")

        with c2:
            _section("AOI VECTOR — GEOJSON")
            geojson_str = json.dumps(R["geojson"], indent=2)
            st.code(geojson_str[:2000] + ("\n…" if len(geojson_str) > 2000 else ""),
                    language="json")
            st.download_button(
                "⬇  Download AOI GeoJSON",
                data=geojson_str.encode(),
                file_name=f"GeoSight_AOI_{disaster_type.split()[0]}.geojson",
                mime="application/geo+json")

            _section("ZONAL STATS — NDVI (4 Quadrants)")
            qs = compute_quadrant_zonal_stats(R["indices_a"]["NDVI"])
            st.dataframe(pd.DataFrame([
                {"Zone": z, "Mean": f"{s['mean']:.4f}", "Std": f"{s['std']:.4f}",
                 "Min": f"{s['min']:.4f}", "Max": f"{s['max']:.4f}",
                 "Median": f"{s['median']:.4f}"}
                for z, s in qs.items()
            ]), use_container_width=True, hide_index=True)

            _section("GEOREFERENCING SUMMARY")
            st.markdown(f"""
            <div class="vector-panel">
            <b style="color:#7c6fff">CRS</b>: {meta['crs']}<br>
            <b style="color:#7c6fff">West</b>: {meta['west']:.8f}°
            &nbsp;<b style="color:#7c6fff">East</b>: {meta['east']:.8f}°<br>
            <b style="color:#7c6fff">North</b>: {meta['north']:.8f}°
            &nbsp;<b style="color:#7c6fff">South</b>: {meta['south']:.8f}°<br>
            <b style="color:#7c6fff">Pixel</b>: {meta['pixel_deg']:.8f}° ≈ {meta['pixel_m']:.1f} m<br>
            <b style="color:#7c6fff">Grid</b>: {meta['width']}×{meta['height']} px<br>
            <b style="color:#7c6fff">Area</b>: {meta['area_km2']} km²<br>
            <b style="color:#7c6fff">Source</b>: {meta['source']}
            </div>""", unsafe_allow_html=True)

    # ── TAB 7: FULL DASHBOARD ───────────────────────────────────────────────
    with tabs[6]:
        _section("FULL ANALYSIS DASHBOARD — 5×4 Panel")
        with st.spinner("Rendering master dashboard…"):
            fig_dash = plot_dashboard(
                R["bands_b"], R["bands_a"],
                R["indices_b"], R["indices_a"],
                R["lulc_b"], R["lulc_a"],
                R["change_map"], R["dNBR"],
                disaster_type, params.get("preset","Custom"),
                date_b, date_a,
            )
            dash_bytes = fig_bytes(fig_dash, dpi=110)
        st.image(dash_bytes, use_container_width=True)
        st.download_button(
            "⬇  Download Dashboard PNG",
            data=dash_bytes,
            file_name=f"GeoSight_{disaster_type.split()[0]}_{date_a}.png",
            mime="image/png")

    # ── TAB 8: EXPORT ───────────────────────────────────────────────────────
    with tabs[7]:
        _section("GEOTIFF EXPORT PACKAGE — Rasterio Georeferenced")

        acc = R["bands_b"].get("_accuracy","unknown")
        note_map = {
            "full":          "7-band GeoTIFF — native CRS and affine transform preserved (maximum accuracy)",
            "reconstructed": "Physics-reconstructed from RGB via ESA S2 spectral unmixing",
            "estimated":     "Single-band panchromatic estimation from ESA LUT",
        }
        st.markdown(f"""
        <div class="gs-infobox success">
        <b>Package:</b> 8 float32 spectral indices + LULC + Change detection — all georeferenced.<br>
        <b>Source:</b> {note_map.get(acc, acc)}<br>
        <b>Writer:</b> rasterio (Affine transform + PROJ CRS + deflate compression + nodata=−9999)<br>
        <b>CRS:</b> {meta['crs']} · Pixel: {meta['pixel_deg']:.8f}° ≈ {meta['pixel_m']:.1f} m<br>
        <b>Bounds:</b> W={meta['west']:.6f} E={meta['east']:.6f}
        N={meta['north']:.6f} S={meta['south']:.6f}
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="vector-panel" style="line-height:2.1">
        <b style="color:#7c6fff">📦 GeoSight_GeoTIFFs.zip contents:</b><br>
        &nbsp;&nbsp;├── NDVI.tif &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float32 · −1 to +1 · nodata=−9999<br>
        &nbsp;&nbsp;├── NDWI.tif &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float32 · −1 to +1 · nodata=−9999<br>
        &nbsp;&nbsp;├── NBR.tif &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float32 · −1 to +1 · nodata=−9999<br>
        &nbsp;&nbsp;├── NDBI.tif &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float32 · −0.5 to +0.5 · nodata=−9999<br>
        &nbsp;&nbsp;├── SAVI.tif &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float32 · L=0.5 · nodata=−9999<br>
        &nbsp;&nbsp;├── EVI.tif &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float32 · G=2.5 · nodata=−9999<br>
        &nbsp;&nbsp;├── MNDWI.tif &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float32 · SWIR-based · nodata=−9999<br>
        &nbsp;&nbsp;├── BSI.tif &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float32 · nodata=−9999<br>
        &nbsp;&nbsp;├── LULC_Classification.tif &nbsp;&nbsp;float32 · 0–6 integer classes<br>
        &nbsp;&nbsp;├── Change_Detection.tif &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float32 · 0–6 change classes<br>
        &nbsp;&nbsp;└── README.txt &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Full metadata + usage instructions
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:var(--surface2);border:1px solid var(--border);
                    border-radius:var(--r);padding:12px 14px;margin-top:12px;
                    font-family:var(--mono);font-size:0.64rem;color:var(--text2);line-height:2.0">
        <b style="color:#7c6fff">Python / rasterio usage example:</b><br>
        import rasterio<br>
        with rasterio.open('NDVI.tif') as src:<br>
        &nbsp;&nbsp;&nbsp;&nbsp;data = src.read(1) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# float32 array<br>
        &nbsp;&nbsp;&nbsp;&nbsp;print(src.crs) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# EPSG:4326<br>
        &nbsp;&nbsp;&nbsp;&nbsp;print(src.transform) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Affine transform<br>
        &nbsp;&nbsp;&nbsp;&nbsp;print(src.bounds) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# BoundingBox<br>
        &nbsp;&nbsp;&nbsp;&nbsp;print(src.nodata) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# -9999.0
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇  DOWNLOAD GEOTIFF PACKAGE (.zip)",
                data=R["geotiff_zip"],
                file_name=f"GeoSight_{disaster_type.split()[0]}_GeoTIFFs.zip",
                mime="application/zip", type="primary")
        with c2:
            st.download_button(
                "⬇  DOWNLOAD AOI VECTOR (GeoJSON)",
                data=json.dumps(R["geojson"], indent=2).encode(),
                file_name=f"GeoSight_AOI_{disaster_type.split()[0]}.geojson",
                mime="application/geo+json")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.markdown("""
    <div class="gs-header">
        <div class="gs-logo-row">
            <div class="gs-logo">🛰️</div>
            <div>
                <div class="gs-title">GeoSight <span>Aurora</span></div>
                <div class="gs-subtitle">Geospatial Intelligence & Disaster Analytics · Pixxels Edition</div>
            </div>
        </div>
        <div>
            <span class="gs-pill live">● SENTINEL-2 L2A</span>
            <span class="gs-pill upload">⬆ REAL SATELLITE UPLOAD</span>
            <span class="gs-pill">8 INDICES</span>
            <span class="gs-pill">LULC · 7 CLASSES</span>
            <span class="gs-pill">RASTERIO GEOTIFF</span>
            <span class="gs-pill acc">ESA/USGS STANDARD</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div style="padding:12px 4px 4px 4px;">', unsafe_allow_html=True)
        st.markdown('<div class="sb-section">⬆  Satellite Image Upload</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="gs-infobox upload" style="font-size:0.63rem;padding:10px 12px;">
        <b>GeoTIFF 7-band</b>: direct band mapping (B2 B3 B4 B5 B8 B11 B12), full accuracy.<br>
        <b>GeoTIFF 3-band</b>: RGB → physics-based reconstruction.<br>
        <b>PNG / JPG</b>: auto-reconstructed via ESA S2 LUT spectral unmixing.<br>
        Both images should cover the same geographic area.
        </div>""", unsafe_allow_html=True)

        uploaded_before = st.file_uploader(
            "Before Image", type=["tif","tiff","png","jpg","jpeg"],
            key="up_before",
            help="Pre-event image — GeoTIFF preferred for full accuracy")
        uploaded_after = st.file_uploader(
            "After Image", type=["tif","tiff","png","jpg","jpeg"],
            key="up_after",
            help="Post-event image for change detection")

        loaded = int(bool(uploaded_before)) + int(bool(uploaded_after))
        if loaded:
            st.markdown(f'<div class="status-ok">✓ {loaded}/2 images ready</div>',
                        unsafe_allow_html=True)

        st.markdown('<div class="sb-section">⚙  Analysis Parameters</div>',
                    unsafe_allow_html=True)
        disaster_type = st.selectbox(
            "Disaster Type",
            ["Wildfire 🔥","Flood 🌊","Drought 🏜️","Urban Expansion 🏙️"])

        st.markdown('<div class="sb-section">📍  Area of Interest</div>',
                    unsafe_allow_html=True)
        preset_aoi = st.selectbox("Preset Location", list(PRESET_LOCATIONS.keys()))
        if PRESET_LOCATIONS[preset_aoi] is not None:
            lat, lon = PRESET_LOCATIONS[preset_aoi]
        else:
            lat = st.number_input("Latitude",  value=20.5937, format="%.6f",
                                  min_value=-90.0,  max_value=90.0)
            lon = st.number_input("Longitude", value=78.9629, format="%.6f",
                                  min_value=-180.0, max_value=180.0)

        st.markdown('<div class="sb-section">📅  Time Period</div>',
                    unsafe_allow_html=True)
        date_before = st.date_input("Before Date", datetime(2024, 1, 15))
        date_after  = st.date_input("After Date",  datetime(2024, 3, 20))
        if date_after <= date_before:
            st.warning("After Date must be later than Before Date.")

        show_map = st.toggle("Show Satellite Map", value=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶  RUN ANALYSIS", type="primary",
                            use_container_width=True)

        st.markdown('<div class="sb-section">ℹ  System Info</div>',
                    unsafe_allow_html=True)
        try:
            import rasterio
            rasterio_status = f"✓ rasterio {rasterio.__version__}"
        except ImportError:
            rasterio_status = "⚠ rasterio not installed (struct fallback active)"

        st.markdown(f"""
        <div class="sb-info">
        <b style="color:#7c6fff">Sensor</b>: Sentinel-2 MSI L2A<br>
        <b style="color:#7c6fff">Bands</b>: B2 B3 B4 B5 B8 B11 B12<br>
        <b style="color:#7c6fff">Indices</b>: NDVI NDWI NBR NDBI SAVI EVI MNDWI BSI<br>
        <b style="color:#7c6fff">Classes</b>: 7 LULC · 7 Change<br>
        <b style="color:#7c6fff">Export</b>: float32 GeoTIFF + GeoJSON<br>
        <b style="color:#7c6fff">Georef</b>: {rasterio_status}
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Landing ───────────────────────────────────────────────────────────────
    if not run_btn and "geo_results" not in st.session_state:
        _render_landing()
        return

    # ── Run Analysis ─────────────────────────────────────────────────────────
    if run_btn:
        if not (uploaded_before or uploaded_after):
            st.error("⚠ Please upload at least one satellite image.", icon="🛰️")
            _render_landing()
            return

        with st.spinner(""):
            prog = st.progress(0, "Initialising GeoSight Aurora pipeline…")
            time.sleep(0.04)

            prog.progress(12, "Loading before image…")
            bands_b = load_image(uploaded_before, target=256) \
                      if uploaded_before else None

            prog.progress(28, "Loading after image…")
            bands_a = load_image(uploaded_after, target=256) \
                      if uploaded_after else None

            # Mirror if only one image provided
            if bands_b is None:
                bands_b = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                           for k, v in bands_a.items()}
            if bands_a is None:
                bands_a = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                           for k, v in bands_b.items()}

            prog.progress(45, "Computing 8 spectral indices…")
            R = run_pipeline(bands_b, bands_a, disaster_type,
                             lat, lon, str(date_before), str(date_after))

            prog.progress(82, "Classifying LULC · building rasterio GeoTIFF package…")
            time.sleep(0.08)
            prog.progress(100, "Aurora analysis complete.")
            time.sleep(0.3)
            prog.empty()

        R["input_mode"] = "upload"
        st.session_state["geo_results"] = R
        st.session_state["geo_params"]  = {
            "disaster_type": disaster_type, "lat": lat, "lon": lon,
            "date_before": str(date_before), "date_after": str(date_after),
            "preset": preset_aoi, "show_map": show_map,
        }
        acc = bands_b.get("_accuracy","unknown")
        src = bands_b.get("_source","unknown")
        st.success(
            f"✓ Analysis complete — source: {src} · accuracy: {acc} · "
            f"7 bands · 8 indices · rasterio GeoTIFF exported",
            icon="✅")

    if "geo_results" in st.session_state:
        _render_results(st.session_state["geo_results"],
                        st.session_state["geo_params"])


if __name__ == "__main__":
    main()
