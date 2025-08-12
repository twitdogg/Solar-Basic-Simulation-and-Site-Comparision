import math
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import pytz
from geopy.geocoders import Nominatim
import pvlib
from pvlib import location, irradiance
import srtm
from math import cos, radians, atan2, atan

# App config & styling
st.set_page_config(
    page_title="Solar Versus",
    layout="wide",
    page_icon=":sun_with_face:"
)

# Color scheme
PRIMARY = "#0ea5e9"  # Sky-500
ACCENT = "#3b82f6"   # Blue-500
BORDER = "#e2e8f0"   # Slate-200
TEXT = "#1e293b"      # Slate-800
BG = "#f8fafc"        # Slate-50

CSS = f"""
<style>
    .stApp {{
        background-color: {BG};
    }}
    .block-container {{
        padding-top: 0.8rem;
    }}
    .stButton>button {{
        background-color: {PRIMARY};
        color: white;
        border-radius: 0.5rem;
        border: none;
    }}
    .stButton>button:hover {{
        background-color: #0369a1;
    }}
    .stTextInput>div>div>input, .stNumberInput>div>div>input {{
        border-radius: 0.5rem;
        border: 1px solid {BORDER};
    }}
    .stSelectbox>div>div>select, .stMultiSelect>div>div>select {{
        border-radius: 0.5rem;
        border: 1px solid {BORDER};
    }}
    .stDataFrame {{
        border-radius: 0.5rem;
        border: 1px solid {BORDER};
    }}
    .stPlotlyChart {{
        border-radius: 0.5rem;
        border: 1px solid {BORDER};
    }}
    /* Bright header */
    .sv-hero {{
        background: linear-gradient(90deg, {PRIMARY} 0%, {ACCENT} 100%);
        color: white;
        padding: 18px 20px;
        border-radius: 14px;
        margin: 8px 0 14px 0;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    }}
    .sv-hero h1, .sv-hero small {{
        color: white !important;
        margin: 0;
    }}
    /* Step tabs (clickable) */
    .sv-steps {{
        display: flex;
        gap: 8px;
        margin: 0 0 10px 0;
    }}
    .sv-step {{
        background: white;
        border: 1px solid {BORDER};
        color: {TEXT};
        font-weight: 600;
        padding: 8px 12px;
        border-radius: 12px;
        cursor: pointer;
        user-select: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    .sv-step.active {{
        border-color: {ACCENT};
        color: {ACCENT};
        box-shadow: 0 0 0 2px rgba(37,99,235,0.10) inset;
    }}
    /* Cards */
    .sv-card {{
        background: white;
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.03);
    }}
    .sv-title {{
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 12px;
        color: {TEXT};
    }}
    /* Footer */
    .sv-footer {{
        text-align: center;
        color: #64748b;
        padding: 18px 0 8px 0;
        font-size: 13px;
        border-top: 1px solid {BORDER};
        margin-top: 24px;
    }}
    /* Dataframe hover */
    .dataframe tbody tr:hover {{
        background-color: #f1f5f9;
    }}
    /* Custom metric card */
    .metric-card {{
        background: white;
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }}
    .metric-label {{
        font-size: 14px;
        color: #64748b;
        margin-bottom: 4px;
    }}
    .metric-value {{
        font-size: 20px;
        font-weight: 600;
        color: {TEXT};
    }}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# Session defaults
def _init_state():
    ss = st.session_state
    ss.setdefault("active_page", "Locations")
    ss.setdefault("sites_df", pd.DataFrame([
        {"name": "Nairobi, Kenya", "lat": -1.2921, "lon": 36.8219, "tz": "Africa/Nairobi",
         "tilt": 15.0, "azimuth": 180.0, "pitch_m": 6.0, "table_depth_m": 2.1}
    ]))
    # sizing defaults
    ss.setdefault("module_w_stc", 400.0)
    ss.setdefault("gamma_pct_per_c", -0.40)
    ss.setdefault("noct_c", 45.0)
    ss.setdefault("wind_coeff", 0.0)
    ss.setdefault("dc_kw_target", 100.0)
    ss.setdefault("modules_per_string", 18)
    ss.setdefault("strings_per_inverter", 12)
    ss.setdefault("override_total_strings", False)
    ss.setdefault("total_strings_manual", 120)
    ss.setdefault("override_inverter_count", False)
    ss.setdefault("inverter_count_manual", 10)
    ss.setdefault("inv_kw_each", 110.0)
    ss.setdefault("inv_eff", 0.97)
    ss.setdefault("albedo", 0.20)
    ss.setdefault("soiling", 0.02)
    ss.setdefault("dc_ohmic", 0.01)
    ss.setdefault("auto_depth", True)
    ss.setdefault("module_length_m", 2.10)
    ss.setdefault("n_portrait", 3)
    ss.setdefault("gap_vertical_m", 0.02)
    # results cache
    ss.setdefault("last_results", None)

_init_state()

# Helper functions
@st.cache_data(show_spinner=False)
def fetch_tmy_and_meta(lat, lon):
    t = pvlib.iotools.get_pvgis_tmy(latitude=lat, longitude=lon, outputformat="json", usehorizon=True)
    if isinstance(t, tuple):
        data = t[0]
        meta = t[1] if len(t) > 1 else {}
    else:
        data = t
        meta = {}
    return data, meta

def _approx_tz_from_lon(lon):
    try:
        offset = int(round(float(lon) / 15.0))
        offset = max(-12, min(14, offset))
        if offset == 0:
            return "Etc/GMT"
        sign = "-" if offset > 0 else "+"
        return f"Etc/GMT{sign}{abs(offset)}"
    except Exception:
        return "UTC"

def auto_timezone(lat, lon):
    try:
        from timezonefinder import TimezoneFinder
        tf = TimezoneFinder()
        tz = tf.timezone_at(lat=lat, lng=lon)
        if tz:
            return tz
    except Exception:
        pass
    return _approx_tz_from_lon(lon)

@st.cache_data(show_spinner=False)
def geocode_place(q):
    geo = Nominatim(user_agent="solar-versus")
    loc = geo.geocode(q, addressdetails=True, timeout=10)
    if not loc:
        return None
    return dict(name=loc.address, lat=loc.latitude, lon=loc.longitude)

def cell_temperature_noct(poa_wm2, temp_air_c, noct_c=45.0, wind_speed=None, wind_coeff=0.0):
    tcell = temp_air_c + (poa_wm2 / 800.0) * (noct_c - 20.0)
    if wind_speed is not None and wind_coeff > 0.0:
        tcell = tcell - wind_coeff * wind_speed
    return tcell

@st.cache_data(show_spinner=False)
def terrain_slope_aspect(lat, lon, sample_deg=0.001):
    elev = srtm.get_data()
    c = elev.get_elevation(lat, lon)
    n = elev.get_elevation(lat + sample_deg, lon)
    s = elev.get_elevation(lat - sample_deg, lon)
    e = elev.get_elevation(lat, lon + sample_deg)
    w = elev.get_elevation(lat, lon - sample_deg)
    if None in (c, n, s, e, w):
        return None, "unknown"
    m_per_deg_lat = 110574.0
    m_per_deg_lon = 111320.0 * cos(radians(lat))
    dz_dy = (n - s) / (2 * sample_deg * m_per_deg_lat)
    dz_dx = (e - w) / (2 * sample_deg * m_per_deg_lon)
    slope_rad = atan(max(0.0, (dz_dx**2 + dz_dy**2))**0.5)
    slope_deg = np.degrees(slope_rad)
    if dz_dx == 0 and dz_dy == 0:
        return float(slope_deg), "flat"
    aspect_rad = atan2(dz_dx, dz_dy)
    aspect_deg = (np.degrees(aspect_rad) + 360.0) % 360.0
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    return float(slope_deg), dirs[int(round(aspect_deg / 45.0))]

def grading_flag(slope_deg):
    if slope_deg is None:
        return "unknown", "Slope data unavailable. Consider wider terrain sampling."
    if slope_deg < 3:
        return "none", "Flat terrain. Grading typically not required."
    if slope_deg < 7:
        return "minor", "Moderate slope. Expect minor grading or tracker tolerance adjustments."
    return "major", "Steep slope. Plan for significant grading and higher civil costs."

def metric_card(label, value, unit=""):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value} {unit}</div>
    </div>
    """

# Header
st.markdown('''
<div class="sv-hero">
    <h1>Solar Versus</h1>
    <small>Advanced Solar Simulation Software</small>
</div>
''', unsafe_allow_html=True)

# Clickable step bar
PAGES = ["Locations", "Electrical Sizing", "Geometry & Losses", "Results"]
cols = st.columns(len(PAGES))
for i, label in enumerate(PAGES):
    with cols[i]:
        active = "active" if st.session_state.active_page == label else ""
        if st.button(label, use_container_width=True):
            st.session_state.active_page = label
        st.markdown(f'<div class="sv-steps"><div class="sv-step {active}">{label}</div></div>', unsafe_allow_html=True)

# Helper for derived sizing preview
def derived_sizing_preview():
    ss = st.session_state
    module_w_stc = ss.module_w_stc
    if ss.override_total_strings:
        total_strings = int(ss.total_strings_manual)
    else:
        total_strings = int(math.ceil((ss.dc_kw_target * 1000.0) / (ss.modules_per_string * module_w_stc)))
    if ss.override_inverter_count:
        inverter_count = int(ss.inverter_count_manual)
    else:
        inverter_count = int(math.ceil(total_strings / max(1, ss.strings_per_inverter)))
    total_modules = total_strings * ss.modules_per_string
    dc_w_actual = total_modules * module_w_stc
    dc_kw_actual = dc_w_actual / 1000.0
    ac_kw_nameplate = inverter_count * ss.inv_kw_each
    dc_ac_ratio_actual = float("nan") if ac_kw_nameplate <= 0 else dc_kw_actual / ac_kw_nameplate
    return total_strings, inverter_count, dc_kw_actual, ac_kw_nameplate, dc_ac_ratio_actual

# Page: Locations
if st.session_state.active_page == "Locations":
    with st.container():
        st.markdown('<div class="sv-card"><div class="sv-title">Site Management</div>', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Search / Manual Entry", "Upload CSV"])

        with tab1:
            st.markdown("""
            <div style="margin-bottom: 16px;">
                <p>Add new sites by searching for a location or entering coordinates manually.</p>
            </div>
            """, unsafe_allow_html=True)

            qcol, tcol, acol, pcol, dcol = st.columns([2, 1, 1, 1, 1])
            query = qcol.text_input("Search any city or 'lat, lon':", value="", placeholder="e.g., New York or 40.7128, -74.0060")
            default_tilt = tcol.number_input("Tilt (deg)", 0.0, 90.0, 15.0, 0.5)
            default_az = acol.number_input("Azimuth (deg)", 0.0, 360.0, 180.0, 1.0, help="0=N, 180=S")
            default_pitch = pcol.number_input("Pitch (m)", 0.1, 100.0, 6.0, 0.1)
            default_depth = dcol.number_input("Table depth (m)", 0.1, 20.0, 2.1, 0.1)

            if st.button("Add Site", type="primary", use_container_width=True):
                candidate = None
                # lat,lon direct?
                try:
                    if "," in query:
                        p1, p2 = [float(x.strip()) for x in query.split(",", 1)]
                        candidate = {"name": f"{p1:.6f}, {p2:.6f}", "lat": p1, "lon": p2}
                except Exception:
                    candidate = None
                if not candidate and query.strip():
                    candidate = geocode_place(query.strip())
                if not candidate:
                    st.error("Place not found. Try a more specific query or paste lat,lon.")
                else:
                    tz = auto_timezone(candidate["lat"], candidate["lon"])
                    new_row = {
                        "name": candidate["name"], "lat": candidate["lat"], "lon": candidate["lon"], "tz": tz,
                        "tilt": default_tilt, "azimuth": default_az, "pitch_m": default_pitch, "table_depth_m": default_depth
                    }
                    st.session_state.sites_df = pd.concat([st.session_state.sites_df, pd.DataFrame([new_row])], ignore_index=True)
                    st.success(f"Added: {candidate['name']} | Timezone: {tz}")

        with tab2:
            st.markdown("""
            <div style="margin-bottom: 16px;">
                <p>Upload a CSV file containing multiple sites. The CSV should include columns for name, latitude, longitude, and optionally tilt, azimuth, timezone, pitch, and table depth.</p>
            </div>
            """, unsafe_allow_html=True)

            st.caption("CSV columns allowed: name, lat, lon, tilt, azimuth, tz, pitch_m, table_depth_m. Timezone is optional and will be auto-detected if blank.")
            up = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
            if st.button("Add Sites from CSV", use_container_width=True):
                try:
                    if up is None:
                        st.error("Please choose a CSV file first.")
                    else:
                        df = pd.read_csv(up)
                        rename = {c: c.lower() for c in df.columns}
                        df = df.rename(columns=rename)
                        for col, default in [("tilt", 15.0), ("azimuth", 180.0), ("pitch_m", 6.0), ("table_depth_m", 2.1), ("tz", "")]:
                            if col not in df.columns:
                                df[col] = default
                        added = []
                        for _, r in df.iterrows():
                            name = str(r["name"])
                            lat = float(r["lat"])
                            lon = float(r["lon"])
                            tilt = float(r["tilt"])
                            az = float(r["azimuth"])
                            pitch = float(r["pitch_m"])
                            depth = float(r["table_depth_m"])
                            tz = str(r["tz"]).strip() if isinstance(r["tz"], str) and r["tz"].strip() else auto_timezone(lat, lon)
                            added.append({"name": name, "lat": lat, "lon": lon, "tz": tz, "tilt": tilt, "azimuth": az, "pitch_m": pitch, "table_depth_m": depth})
                        st.session_state.sites_df = pd.concat([st.session_state.sites_df, pd.DataFrame(added)], ignore_index=True)
                        st.success(f"Added {len(added)} site(s).")
                except Exception as e:
                    st.error(f"CSV error: {e}")

        # Editable table with timezone dropdown
        st.markdown('<div class="sv-title" style="margin-top: 16px;">Sites List</div>', unsafe_allow_html=True)
        ALL_TZ = [""] + sorted(pytz.all_timezones)
        st.session_state.sites_df = st.data_editor(
            st.session_state.sites_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Name", width="medium"),
                "lat": st.column_config.NumberColumn("Latitude", step=0.000001, format="%.6f", width="small"),
                "lon": st.column_config.NumberColumn("Longitude", step=0.000001, format="%.6f", width="small"),
                "tz": st.column_config.SelectboxColumn("Timezone", options=ALL_TZ, required=False, width="medium"),
                "tilt": st.column_config.NumberColumn("Tilt (deg)", min_value=0.0, max_value=90.0, step=0.5, width="small"),
                "azimuth": st.column_config.NumberColumn("Azimuth (deg)", min_value=0.0, max_value=360.0, step=1.0, width="small"),
                "pitch_m": st.column_config.NumberColumn("Pitch (m)", min_value=0.1, step=0.1, width="small"),
                "table_depth_m": st.column_config.NumberColumn("Table Depth (m)", min_value=0.1, step=0.1, width="small"),
            },
            key="sites_editor",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Map visualization
    try:
        map_df = st.session_state.sites_df.dropna(subset=["lat", "lon"])[["lat", "lon"]]
        if not map_df.empty:
            st.markdown('<div class="sv-card"><div class="sv-title">Site Locations</div>', unsafe_allow_html=True)
            st.map(map_df, zoom=2)
            st.markdown('</div>', unsafe_allow_html=True)
    except Exception:
        pass

# Page: Electrical Sizing
elif st.session_state.active_page == "Electrical Sizing":
    with st.container():
        st.markdown('<div class="sv-card"><div class="sv-title">Electrical System Configuration</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-bottom: 20px;">
            <p>Configure the electrical parameters of your solar PV system. This includes module specifications, inverter settings, and system sizing.</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        st.session_state.module_w_stc = c1.number_input("Module STC Power (W)", 10.0, 1200.0, st.session_state.module_w_stc, 10.0)
        st.session_state.gamma_pct_per_c = c2.number_input("Temperature Coefficient γPdc (%/°C)", -2.0, 0.0, st.session_state.gamma_pct_per_c, 0.01)
        st.session_state.noct_c = c3.number_input("NOCT (°C)", 20.0, 60.0, st.session_state.noct_c, 0.5)
        st.session_state.wind_coeff = c4.number_input("Wind Cooling Coefficient (°C per m/s)", 0.0, 5.0, st.session_state.wind_coeff, 0.1)

        m1, m2, m3 = st.columns(3)
        st.session_state.dc_kw_target = m1.number_input("Target Total DC Size (kWp)", 0.1, 2000000.0, st.session_state.dc_kw_target, 10.0)
        st.session_state.modules_per_string = m2.number_input("Modules per String", 1, 1000, st.session_state.modules_per_string, 1)
        st.session_state.strings_per_inverter = m3.number_input("Strings per Inverter", 1, 2000, st.session_state.strings_per_inverter, 1)

        o1, o2, o3, o4 = st.columns(4)
        st.session_state.override_total_strings = o1.checkbox("Override Total Strings", value=st.session_state.override_total_strings)
        st.session_state.total_strings_manual = o2.number_input("Total Strings (manual)", 1, 10000000, st.session_state.total_strings_manual, 1, disabled=not st.session_state.override_total_strings)
        st.session_state.override_inverter_count = o3.checkbox("Override Inverter Count", value=st.session_state.override_inverter_count)
        st.session_state.inverter_count_manual = o4.number_input("Inverters (manual)", 1, 10000000, st.session_state.inverter_count_manual, 1, disabled=not st.session_state.override_inverter_count)

        i1, i2 = st.columns(2)
        st.session_state.inv_kw_each = i1.number_input("Inverter AC Nameplate (kW each)", 1.0, 10000.0, st.session_state.inv_kw_each, 1.0)
        st.session_state.inv_eff = i2.number_input("Nominal Inverter Efficiency (0–1)", 0.6, 0.999, st.session_state.inv_eff, 0.001)

        # Live preview
        st.markdown('<div class="sv-title" style="margin-top: 24px;">System Sizing Preview</div>', unsafe_allow_html=True)
        ts, invs, dc_kw_act, ac_kw_name, dcac = derived_sizing_preview()

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.markdown(metric_card("Total Strings", ts), unsafe_allow_html=True)
        col2.markdown(metric_card("Modules/String", st.session_state.modules_per_string), unsafe_allow_html=True)
        col3.markdown(metric_card("Inverters", invs), unsafe_allow_html=True)
        col4.markdown(metric_card("DC Actual", f"{dc_kw_act:,.2f}", "kWp"), unsafe_allow_html=True)
        col5.markdown(metric_card("AC Nameplate", f"{ac_kw_name:,.2f}", "kW"), unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# Page: Geometry & Losses
elif st.session_state.active_page == "Geometry & Losses":
    with st.container():
        st.markdown('<div class="sv-card"><div class="sv-title">System Geometry and Losses</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-bottom: 20px;">
            <p>Configure the physical layout of your solar array and account for various system losses.</p>
        </div>
        """, unsafe_allow_html=True)

        l1, l2, l3 = st.columns(3)
        st.session_state.albedo = l1.number_input("Albedo", 0.0, 1.0, st.session_state.albedo, 0.01)
        st.session_state.soiling = l2.number_input("Soiling Loss Fraction", 0.0, 0.5, st.session_state.soiling, 0.01)
        st.session_state.dc_ohmic = l3.number_input("DC Ohmic Loss Fraction", 0.0, 0.5, st.session_state.dc_ohmic, 0.01)

        st.markdown('<div class="sv-title" style="margin-top: 24px;">Auto Table Depth from Module Geometry</div>', unsafe_allow_html=True)

        ad1, ad2, ad3, ad4 = st.columns(4)
        st.session_state.auto_depth = ad1.checkbox("Enable Auto Calculation", value=st.session_state.auto_depth)
        st.session_state.module_length_m = ad2.number_input("Module Length (m) in Tilt Direction", 0.5, 4.0, st.session_state.module_length_m, 0.01)
        st.session_state.n_portrait = ad3.number_input("Modules Stacked in Portrait (Vertical)", 1, 10, st.session_state.n_portrait, 1)
        st.session_state.gap_vertical_m = ad4.number_input("Vertical Gap Between Modules (m)", 0.0, 0.5, st.session_state.gap_vertical_m, 0.01)

        st.caption("Per-site pitch and table depth are edited on the Locations page. If Auto is enabled, table depth is computed as: (N_portrait × module_length + gaps) × cos(tilt).")

        st.markdown('</div>', unsafe_allow_html=True)

# Page: Results
else:
    # Simulation button
    if st.button("Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Running simulation..."):
            st.session_state.last_results = run_simulation()

    res = st.session_state.last_results

    if res is None:
        st.info("Please adjust inputs on the other pages, then click **Run Simulation**.")
    else:
        cap = res["capacity"]

        st.markdown('<div class="sv-card"><div class="sv-title">System Capacity Summary</div>', unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.markdown(metric_card("Total Strings", cap['total_strings']), unsafe_allow_html=True)
        col2.markdown(metric_card("Modules/String", cap['modules_per_string']), unsafe_allow_html=True)
        col3.markdown(metric_card("Inverters", cap['inverter_count']), unsafe_allow_html=True)
        col4.markdown(metric_card("DC Capacity", f"{cap['dc_kw_actual']:,.2f}", "kWp"), unsafe_allow_html=True)
        col5.markdown(metric_card("AC Capacity", f"{cap['ac_kw_nameplate']:,.2f}", "kW"), unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Per-site KPIs
        st.markdown('<div class="sv-card"><div class="sv-title">Site Performance Metrics</div>', unsafe_allow_html=True)
        kpis = res["kpis"]
        st.dataframe(kpis, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Specific Yield Comparison
        st.markdown('<div class="sv-card"><div class="sv-title">Specific Yield Comparison (kWh/kWp)</div>', unsafe_allow_html=True)
        kpis_sy = kpis.dropna(subset=["Specific yield (kWh/kWp)"]).sort_values("Specific yield (kWh/kWp)", ascending=True)
        if not kpis_sy.empty:
            fig_sy = px.bar(
                kpis_sy,
                x="Specific yield (kWh/kWp)",
                y="Site",
                orientation="h",
                color="Site",
                text="Specific yield (kWh/kWp)",
                color_discrete_sequence=px.colors.qualitative.Set2,
                template="plotly_white",
                height=420
            )
            fig_sy.update_traces(textposition="outside")
            fig_sy.update_layout(
                xaxis_title="Specific Yield (kWh/kWp)",
                yaxis_title="",
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="white"
            )
            st.plotly_chart(fig_sy, use_container_width=True)
        else:
            st.info("No specific yield values to plot.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Monthly Energy Grouped
        st.markdown('<div class="sv-card"><div class="sv-title">Monthly Energy Production</div>', unsafe_allow_html=True)
        monthly_all = res["monthly_all"]
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_all["MonthName"] = monthly_all["Month"].map({i+1: m for i, m in enumerate(month_labels)})

        fig_month = px.bar(
            monthly_all,
            x="MonthName",
            y="kWh",
            color="Site",
            barmode="group",
            category_orders={"MonthName": month_labels},
            labels={"kWh": "Energy (kWh)"},
            color_discrete_sequence=px.colors.qualitative.Set2,
            template="plotly_white",
            height=440
        )
        fig_month.update_yaxes(tickformat=",")
        fig_month.update_layout(
            legend_title="",
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white"
        )
        st.plotly_chart(fig_month, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Annual Yield Comparison
        st.markdown('<div class="sv-card"><div class="sv-title">Annual Energy Production Comparison</div>', unsafe_allow_html=True)
        ann = kpis[["Site", "Annual kWh", "Specific yield (kWh/kWp)"]].copy()
        ann_sorted = ann.sort_values("Annual kWh", ascending=True)

        fig_ann = px.bar(
            ann_sorted,
            x="Annual kWh",
            y="Site",
            orientation="h",
            color="Specific yield (kWh/kWp)",
            color_continuous_scale="Viridis",
            text="Annual kWh",
            template="plotly_white",
            height=440
        )
        fig_ann.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_ann.update_layout(
            coloraxis_colorbar=dict(title="kWh/kWp"),
            xaxis_title="Annual Energy (kWh)",
            yaxis_title="",
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white"
        )
        st.plotly_chart(fig_ann, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Terrain Slope and Grading
        st.markdown('<div class="sv-card"><div class="sv-title">Terrain Analysis</div>', unsafe_allow_html=True)
        slope_df = kpis[["Site", "Pitch (m)", "Table depth (m)", "GCR", "Slope (deg)", "Aspect", "Grading", "Recommendation"]].copy()
        slope_sorted = slope_df.sort_values(by=["Slope (deg)"], ascending=True, na_position="last")

        st.dataframe(slope_sorted, use_container_width=True)

        fig_slope = px.bar(
            slope_sorted,
            x="Slope (deg)",
            y="Site",
            orientation="h",
            color="Grading",
            category_orders={"Grading": ["none", "minor", "major", "unknown"]},
            color_discrete_map={"none": "#16a34a", "minor": "#f59e0b", "major": "#dc2626", "unknown": "#64748b"},
            template="plotly_white",
            height=420
        )
        fig_slope.update_layout(
            xaxis_title="Slope (degrees)",
            yaxis_title="",
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white"
        )
        st.plotly_chart(fig_slope, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Downloads
        st.markdown('<div class="sv-card"><div class="sv-title">Download Results</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download Site KPIs (CSV)",
                kpis.to_csv(index=False).encode(),
                file_name="site_kpis.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            monthly_wide = monthly_all.pivot_table(index="MonthName", columns="Site", values="kWh", aggfunc="sum")
            monthly_wide = monthly_wide.reindex(month_labels)
            st.download_button(
                "Download Monthly Energy Data (CSV)",
                monthly_wide.to_csv().encode(),
                file_name="monthly_energy_per_site.csv",
                mime="text/csv",
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('''
<div class="sv-footer">
    Solar Versus © Eyob Aga Muleta — Advanced Solar Simulation Software
</div>
''', unsafe_allow_html=True)

