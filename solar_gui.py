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

# Optional precise tz:
try:
    from timezonefinder import TimezoneFinder  # type: ignore
    _HAS_TZF = True
except Exception:
    TimezoneFinder = None
    _HAS_TZF = False

# Terrain
import srtm
from math import cos, radians, atan2, atan

# ------------------------------ App config & styling ------------------------------
st.set_page_config(page_title="Solar Versus — by Eyob Aga Muleta", layout="wide")

# Define a more modern, bright color palette
PRIMARY = "#2563eb"  # A deep, professional blue
ACCENT = "#34d399"   # A vibrant, lively green for highlights
BG = "#f8f9fa"       # A subtle off-white for the background
CARD_BG = "#ffffff"  # Pure white for cards to create contrast
BORDER = "#e2e8f0"   # Light gray for borders
TEXT = "#1f2937"     # A dark gray for body text

CSS = f"""
<style>
/* Streamlit main container padding */
.block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}

/* Bright, gradient header */
.sv-hero {{
  background: linear-gradient(90deg, {PRIMARY} 0%, #2563eb 100%);
  color: white;
  padding: 2.5rem 2rem;
  border-radius: 1rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 10px 25px rgba(0,0,0,0.1);
  text-align: center;
}}
.sv-hero h1 {{ color: white !important; margin: 0; font-size: 2.5rem; font-weight: 700; }}
.sv-hero small {{ color: rgba(255, 255, 255, 0.8) !important; font-size: 1rem; margin-top: 0.5rem; display: block; }}

/* Step tabs (clickable buttons) */
.stButton>button {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER};
    color: {TEXT};
    font-weight: 600;
    padding: 0.75rem 1.25rem;
    border-radius: 0.75rem;
    cursor: pointer;
    user-select: none;
    transition: all 0.2s ease-in-out;
}}
.stButton>button:hover {{
    border-color: {ACCENT};
    box-shadow: 0 0 0 2px rgba(52, 211, 153, 0.2);
}}
.stButton>button:active {{
    background-color: {BG};
}}
/* Specific styling for active tab */
.stButton>button[data-testid="base-button-secondary"]:focus {{
    border-color: {PRIMARY};
    color: {PRIMARY};
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2);
}}

/* Cards */
.sv-card {{
  background: {CARD_BG};
  border: 1px solid {BORDER};
  border-radius: 1rem;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 4px 18px rgba(0,0,0,0.05);
}}
.sv-title {{ font-weight: 700; font-size: 1.25rem; margin-bottom: 1rem; color:{TEXT}; }}

/* Footer */
.sv-footer {{
  text-align: center;
  color: #64748b;
  padding: 1.5rem 0 1rem 0;
  font-size: 0.8rem;
  border-top: 1px solid {BORDER};
  margin-top: 2rem;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ------------------------------ Session defaults ------------------------------
def _init_state():
    ss = st.session_state
    ss.setdefault("active_page", "Locations")
    ss.setdefault("sites_df", pd.DataFrame([
        {"name":"Nairobi, Kenya","lat":-1.2921,"lon":36.8219,"tz":"Africa/Nairobi","tilt":15.0,"azimuth":180.0,"pitch_m":6.0,"table_depth_m":2.1}
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

# ------------------------------ Helpers ------------------------------
@st.cache_data(show_spinner=False)
def fetch_tmy_and_meta(lat, lon):
    t = pvlib.iotools.get_pvgis_tmy(latitude=lat, longitude=lon, outputformat="json", usehorizon=True)
    if isinstance(t, tuple):
        data = t[0]; meta = t[1] if len(t) > 1 else {}
    else:
        data = t; meta = {}
    return data, meta

def _approx_tz_from_lon(lon):
    try:
        offset = int(round(float(lon) / 15.0))
        offset = max(-12, min(14, offset))
        if offset == 0: return "Etc/GMT"
        sign = "-" if offset > 0 else "+"
        return f"Etc/GMT{sign}{abs(offset)}"
    except Exception:
        return "UTC"

def auto_timezone(lat, lon):
    if _HAS_TZF:
        try:
            tf = TimezoneFinder()
            tz = tf.timezone_at(lat=lat, lng=lon)
            if tz: return tz
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
    c  = elev.get_elevation(lat, lon)
    n  = elev.get_elevation(lat + sample_deg, lon)
    s  = elev.get_elevation(lat - sample_deg, lon)
    e  = elev.get_elevation(lat, lon + sample_deg)
    w  = elev.get_elevation(lat, lon - sample_deg)
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
    dirs = ["N","NE","E","SE","S","SW","W","NW","N"]
    return float(slope_deg), dirs[int(round(aspect_deg / 45.0))]

def grading_flag(slope_deg):
    if slope_deg is None:
        return "unknown", "Slope data unavailable. Consider wider terrain sampling."
    if slope_deg < 3:
        return "none", "Flat terrain. Grading typically not required."
    if slope_deg < 7:
        return "minor", "Moderate slope. Expect minor grading or tracker tolerance adjustments."
    return "major", "Steep slope. Plan for significant grading and higher civil costs."

# ------------------------------ Header ------------------------------
st.markdown('<div class="sv-hero"><h1>Solar Versus</h1><small>Your definitive solar energy simulator</small></div>', unsafe_allow_html=True)

# ------------------------------ Clickable step bar ------------------------------
PAGES = ["Locations", "Electrical sizing", "Geometry & losses", "Results"]
cols = st.columns(len(PAGES))
for i, label in enumerate(PAGES):
    with cols[i]:
        if st.button(label, use_container_width=True, type="secondary" if st.session_state.active_page != label else "primary"):
            st.session_state.active_page = label

# ---------- tiny helper to compute derived sizing preview ----------
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
    dc_w_actual   = total_modules * module_w_stc
    dc_kw_actual  = dc_w_actual / 1000.0
    ac_kw_nameplate = inverter_count * ss.inv_kw_each
    dc_ac_ratio_actual = float("nan") if ac_kw_nameplate <= 0 else dc_kw_actual / ac_kw_nameplate
    return total_strings, inverter_count, dc_kw_actual, ac_kw_nameplate, dc_ac_ratio_actual

# ------------------------------ PAGE: Locations ------------------------------
if st.session_state.active_page == "Locations":
    with st.container():
        st.markdown('<div class="sv-card"><div class="sv-title">Add or edit your sites</div>', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Search / manual", "Upload CSV"])
        with tab1:
            qcol, tcol, acol, pcol, dcol = st.columns([2,1,1,1,1])
            query = qcol.text_input("Search any city or 'lat, lon':", value="")
            default_tilt = tcol.number_input("Tilt (deg)", 0.0, 90.0, 15.0, 0.5)
            default_az   = acol.number_input("Azimuth (deg)", 0.0, 360.0, 180.0, 1.0, help="0=N, 180=S")
            default_pitch= pcol.number_input("Pitch (m)", 0.1, 100.0, 6.0, 0.1)
            default_depth= dcol.number_input("Table depth (m)", 0.1, 20.0, 2.1, 0.1)
            if st.button("Add site", type="primary"):
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
                    st.success(f"Added: {candidate['name']}  | tz: {tz}")

        with tab2:
            st.caption("CSV columns allowed: name, lat, lon, tilt, azimuth, tz, pitch_m, table_depth_m. tz optional (auto if blank).")
            up = st.file_uploader("Upload CSV", type=["csv"])
            if st.button("Add from CSV"):
                try:
                    if up is None:
                        st.error("Choose a CSV file first.")
                    else:
                        df = pd.read_csv(up)
                        rename = {c: c.lower() for c in df.columns}
                        df = df.rename(columns=rename)
                        for col, default in [("tilt", 15.0), ("azimuth", 180.0), ("pitch_m", 6.0), ("table_depth_m", 2.1), ("tz", "")]:
                            if col not in df.columns: df[col] = default
                        added = []
                        for _, r in df.iterrows():
                            name = str(r["name"])
                            lat  = float(r["lat"]); lon = float(r["lon"])
                            tilt = float(r["tilt"]); az = float(r["azimuth"])
                            pitch= float(r["pitch_m"]); depth = float(r["table_depth_m"])
                            tz   = str(r["tz"]).strip() if isinstance(r["tz"], str) and r["tz"].strip() else auto_timezone(lat, lon)
                            added.append({"name":name,"lat":lat,"lon":lon,"tz":tz,"tilt":tilt,"azimuth":az,"pitch_m":pitch,"table_depth_m":depth})
                        st.session_state.sites_df = pd.concat([st.session_state.sites_df, pd.DataFrame(added)], ignore_index=True)
                        st.success(f"Added {len(added)} site(s).")
                except Exception as e:
                    st.error(f"CSV error: {e}")

        # Editable table with timezone dropdown
        st.markdown('<div class="sv-title" style="margin-top:6px;">Sites list</div>', unsafe_allow_html=True)
        ALL_TZ = [""] + pytz.all_timezones
        st.session_state.sites_df = st.data_editor(
            st.session_state.sites_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Name"),
                "lat": st.column_config.NumberColumn("Latitude", step=0.000001, format="%.6f"),
                "lon": st.column_config.NumberColumn("Longitude", step=0.000001, format="%.6f"),
                "tz": st.column_config.SelectboxColumn("Timezone (leave blank = auto)", options=ALL_TZ, required=False),
                "tilt": st.column_config.NumberColumn("Tilt (deg)", min_value=0.0, max_value=90.0, step=0.5),
                "azimuth": st.column_config.NumberColumn("Azimuth (deg)", min_value=0.0, max_value=360.0, step=1.0),
                "pitch_m": st.column_config.NumberColumn("Pitch (m)", min_value=0.1, step=0.1),
                "table_depth_m": st.column_config.NumberColumn("Table depth (m)", min_value=0.1, step=0.1),
            },
            key="sites_editor",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # quick map
    try:
        map_df = st.session_state.sites_df.dropna(subset=["lat","lon"])[["lat","lon"]]
        if not map_df.empty:
            st.map(map_df, zoom=2)
    except Exception:
        pass

# ------------------------------ PAGE: Electrical sizing ------------------------------
elif st.session_state.active_page == "Electrical sizing":
    with st.container():
        st.markdown('<div class="sv-card"><div class="sv-title">Electrical sizing</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        st.session_state.module_w_stc    = c1.number_input("Module STC power (W)", 10.0, 1200.0, st.session_state.module_w_stc, 10.0)
        st.session_state.gamma_pct_per_c = c2.number_input("Temp coeff γPdc (%/°C)", -2.0, 0.0, st.session_state.gamma_pct_per_c, 0.01)
        st.session_state.noct_c          = c3.number_input("NOCT (°C)", 20.0, 60.0, st.session_state.noct_c, 0.5)
        st.session_state.wind_coeff      = c4.number_input("Wind cooling coeff (°C per m/s)", 0.0, 5.0, st.session_state.wind_coeff, 0.1)

        m1, m2, m3 = st.columns(3)
        st.session_state.dc_kw_target        = m1.number_input("Target total DC size (kWp)", 0.1, 2_000_000.0, st.session_state.dc_kw_target, 10.0)
        st.session_state.modules_per_string  = m2.number_input("Modules per string", 1, 1000, st.session_state.modules_per_string, 1)
        st.session_state.strings_per_inverter = m3.number_input("Strings per inverter", 1, 2000, st.session_state.strings_per_inverter, 1)

        o1, o2, o3, o4 = st.columns(4)
        st.session_state.override_total_strings = o1.checkbox("Override total strings", value=st.session_state.override_total_strings)
        st.session_state.total_strings_manual    = o2.number_input("Total strings (manual)", 1, 10_000_000, st.session_state.total_strings_manual, 1, disabled=not st.session_state.override_total_strings)
        st.session_state.override_inverter_count = o3.checkbox("Override inverter count", value=st.session_state.override_inverter_count)
        st.session_state.inverter_count_manual   = o4.number_input("Inverters (manual)", 1, 10_000_000, st.session_state.inverter_count_manual, 1, disabled=not st.session_state.override_inverter_count)

        i1, i2 = st.columns(2)
        st.session_state.inv_kw_each      = i1.number_input("Inverter AC nameplate (kW each)", 1.0, 10_000.0, st.session_state.inv_kw_each, 1.0)
        st.session_state.inv_eff          = i2.number_input("Nominal inverter efficiency (0–1)", 0.6, 0.999, st.session_state.inv_eff, 0.001)

        # live preview
        ts, invs, dc_kw_act, ac_kw_name, dcac = derived_sizing_preview()
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total strings", f"{ts}")
        k2.metric("Modules/string", f"{st.session_state.modules_per_string}")
        k3.metric("Inverters", f"{invs}")
        k4.metric("DC actual", f"{dc_kw_act:,.2f} kWp")
        k5.metric("AC nameplate", f"{ac_kw_name:,.2f} kW")

        st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------ PAGE: Geometry & losses ------------------------------
elif st.session_state.active_page == "Geometry & losses":
    with st.container():
        st.markdown('<div class="sv-card"><div class="sv-title">Geometry & losses</div>', unsafe_allow_html=True)

        l1, l2, l3 = st.columns(3)
        st.session_state.albedo   = l1.number_input("Albedo", 0.0, 1.0, st.session_state.albedo, 0.01)
        st.session_state.soiling  = l2.number_input("Soiling loss fraction", 0.0, 0.5, st.session_state.soiling, 0.01)
        st.session_state.dc_ohmic = l3.number_input("DC ohmic loss fraction", 0.0, 0.5, st.session_state.dc_ohmic, 0.01)

        st.markdown('<div class="sv-title" style="margin-top:6px;">Auto table depth from module geometry</div>', unsafe_allow_html=True)
        ad1, ad2, ad3, ad4 = st.columns(4)
        st.session_state.auto_depth      = ad1.checkbox("Enable auto calculation", value=st.session_state.auto_depth)
        st.session_state.module_length_m = ad2.number_input("Module length (m) in tilt direction", 0.5, 4.0, st.session_state.module_length_m, 0.01)
        st.session_state.n_portrait      = ad3.number_input("Modules stacked in portrait (vertical)", 1, 10, st.session_state.n_portrait, 1)
        st.session_state.gap_vertical_m  = ad4.number_input("Vertical gap between modules (m)", 0.0, 0.5, st.session_state.gap_vertical_m, 0.01)

        st.caption("Per-site pitch and table depth are edited on the Locations page. If Auto is enabled, table depth is computed as: (N_portrait × module_length + gaps) × cos(tilt).")

        st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------ PAGE: Results ------------------------------
else:
    # one button to simulate
    simulate = st.button("Simulate now", type="primary", use_container_width=True)

    def run_simulation():
        ss = st.session_state
        sites_df = ss.sites_df.dropna(subset=["lat","lon"]).copy()
        if sites_df.empty:
            st.error("No valid sites. Add at least one site on the Locations page.")
            return None

        gamma = ss.gamma_pct_per_c / 100.0
        module_w_stc = ss.module_w_stc

        # stringing + capacity
        if ss.override_total_strings:
            total_strings_actual = int(ss.total_strings_manual)
        else:
            total_strings_actual = int(math.ceil((ss.dc_kw_target * 1000.0) / (ss.modules_per_string * module_w_stc)))

        if ss.override_inverter_count:
            inverter_count = int(ss.inverter_count_manual)
        else:
            inverter_count = int(math.ceil(total_strings_actual / max(1, ss.strings_per_inverter)))

        total_modules_actual = total_strings_actual * ss.modules_per_string
        dc_w_actual = total_modules_actual * module_w_stc
        dc_kw_actual = dc_w_actual / 1000.0

        ac_kw_nameplate = inverter_count * ss.inv_kw_each

        kpi_rows, monthly_rows = [], []

        for _, r in sites_df.iterrows():
            name = (str(r.get("name","")).strip() or "Site")
            lat = float(r["lat"]); lon = float(r["lon"])
            tz_field = str(r.get("tz","")).strip()
            tz = tz_field if tz_field else auto_timezone(lat, lon)

            tilt = float(r.get("tilt", 15.0)); azimuth = float(r.get("azimuth", 180.0))
            pitch_m = float(r.get("pitch_m", 6.0))

            # auto or manual depth
            if ss.auto_depth:
                stack = ss.n_portrait * ss.module_length_m + max(0, ss.n_portrait - 1) * ss.gap_vertical_m
                table_depth_m = stack * math.cos(math.radians(tilt))
            else:
                table_depth_m = float(r.get("table_depth_m", 2.1))

            gcr = np.nan if pitch_m <= 0 else max(0.0, min(1.0, table_depth_m / pitch_m))

            # weather
            try:
                tmy, _ = fetch_tmy_and_meta(lat, lon)
            except Exception as e:
                st.error(f"{name}: PVGIS TMY fetch failed: {e}")
                continue

            # time index
            try:
                site = location.Location(lat, lon, tz=tz, altitude=0, name=name)
            except Exception:
                st.error(f"{name}: invalid tz '{tz}'. Skipping.")
                continue
            if tmy.index.tz is None:
                try:
                    tmy.index = tmy.index.tz_localize(tz)
                except Exception:
                    tmy.index = tmy.index.tz_localize("UTC").tz_convert(tz)

            # plane-of-array
            solpos = site.get_solarposition(times=tmy.index)
            poa = irradiance.get_total_irradiance(
                surface_tilt=tilt, surface_azimuth=azimuth,
                dni=tmy["dni"], ghi=tmy["ghi"], dhi=tmy["dhi"],
                solar_zenith=solpos["zenith"], solar_azimuth=solpos["azimuth"],
                albedo=ss.albedo
            )
            poa_global = poa["poa_global"]

            # temperature + power
            wind = tmy["wind_speed"] if "wind_speed" in tmy.columns else None
            t_cell = cell_temperature_noct(poa_global, tmy["temp_air"],
                                           noct_c=ss.noct_c, wind_speed=wind, wind_coeff=ss.wind_coeff)

            pdc0_total_w = dc_w_actual
            pdc = pdc0_total_w * (poa_global / 1000.0) * (1.0 + gamma * (t_cell - 25.0))
            pdc = pdc.clip(lower=0.0)

            pac_nom = pdc * ss.inv_eff
            ac_w = np.minimum(pac_nom, ac_kw_nameplate * 1000.0) * (1 - ss.soiling) * (1 - ss.dc_ohmic)
            ac_w = ac_w.clip(lower=0.0)

            # KPIs
            annual_kwh = ac_w.sum() / 1000.0
            monthly_kwh = ac_w.resample("MS").sum() / 1000.0
            specific_yield = float("nan") if dc_kw_actual <= 0 else annual_kwh / dc_kw_actual
            capacity_factor = float("nan") if ac_kw_nameplate <= 0 else annual_kwh / (ac_kw_nameplate * 8760.0)
            pr = float("nan") if dc_kw_actual <= 0 else annual_kwh / ((poa_global.sum() / 1000.0) * dc_kw_actual)

            slope_deg, aspect = terrain_slope_aspect(lat, lon)
            grade_flag, grade_note = grading_flag(slope_deg)

            kpi_rows.append({
                "Site": name, "TZ": tz,
                "Tilt (°)": tilt, "Azimuth (°)": azimuth,
                "Pitch (m)": round(pitch_m, 2), "Table depth (m)": round(table_depth_m, 2),
                "GCR": None if np.isnan(gcr) else round(gcr, 2),
                "Annual kWh": round(annual_kwh, 0),
                "Specific yield (kWh/kWp)": None if np.isnan(specific_yield) else round(specific_yield, 1),
                "Capacity factor (%)": None if np.isnan(capacity_factor) else round(capacity_factor * 100.0, 1),
                "PR": None if np.isnan(pr) else round(pr, 2),
                "Slope (deg)": None if slope_deg is None else round(slope_deg, 1),
                "Aspect": aspect,
                "Grading": grade_flag,
                "Recommendation": grade_note
            })

            mdf = monthly_kwh.to_frame(name="kWh")
            mdf["Month"] = mdf.index.month
            mdf["Site"]  = name
            monthly_rows.append(mdf[["Month","Site","kWh"]].reset_index(drop=True))

        if not kpi_rows:
            return None

        # aggregate
        kpis = pd.DataFrame(kpi_rows)
        monthly_all = pd.concat(monthly_rows, ignore_index=True)
        return dict(
            kpis=kpis,
            monthly_all=monthly_all,
            capacity= dict(
                total_strings=total_strings_actual,
                modules_per_string=st.session_state.modules_per_string,
                inverter_count=inverter_count,
                dc_kw_actual=dc_kw_actual,
                ac_kw_nameplate=ac_kw_nameplate
            )
        )

    if simulate:
        st.session_state.last_results = run_simulation()

    res = st.session_state.last_results
    if res is None:
        st.info("Adjust inputs on the other pages, then click **Simulate now**.")
    else:
        cap = res["capacity"]
        m = st.columns(5)
        m[0].metric("Total strings", f"{cap['total_strings']}")
        m[1].metric("Modules/string", f"{cap['modules_per_string']}")
        m[2].metric("Inverters", f"{cap['inverter_count']}")
        m[3].metric("DC actual", f"{cap['dc_kw_actual']:,.2f} kWp")
        m[4].metric("AC nameplate", f"{cap['ac_kw_nameplate']:,.2f} kW")

        kpis = res["kpis"]
        st.markdown('<div class="sv-card"><div class="sv-title">Per-site KPIs</div>', unsafe_allow_html=True)
        st.dataframe(kpis, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Specific Yield comparison
        st.markdown('<div class="sv-card"><div class="sv-title">Specific yield comparison (kWh/kWp)</div>', unsafe_allow_html=True)
        kpis_sy = kpis.dropna(subset=["Specific yield (kWh/kWp)"]).sort_values("Specific yield (kWh/kWp)", ascending=True)
        if not kpis_sy.empty:
            fig_sy = px.bar(
                kpis_sy, x="Specific yield (kWh/kWp)", y="Site",
                orientation="h", color="Site", text="Specific yield (kWh/kWp)",
                color_discrete_sequence=px.colors.qualitative.Plotly,
                template="plotly_white", height=420
            )
            fig_sy.update_traces(textposition="outside")
            fig_sy.update_layout(xaxis_title="kWh/kWp", yaxis_title="", showlegend=False,
                                 margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_sy, use_container_width=True)
        else:
            st.info("No specific yield values to plot.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Monthly energy grouped
        st.markdown('<div class="sv-card"><div class="sv-title">Monthly AC energy (grouped)</div>', unsafe_allow_html=True)
        monthly_all = res["monthly_all"]
        month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly_all["MonthName"] = monthly_all["Month"].map({i+1:m for i,m in enumerate(month_labels)})
        fig_month = px.bar(
            monthly_all, x="MonthName", y="kWh", color="Site",
            barmode="group", category_orders={"MonthName": month_labels},
            labels={"kWh":"Energy (kWh)"},
            color_discrete_sequence=px.colors.qualitative.Plotly, template="plotly_white", height=440
        )
        fig_month.update_yaxes(tickformat=",")
        fig_month.update_layout(legend_title="", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_month, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Annual yield comparison
        st.markdown('<div class="sv-card"><div class="sv-title">Annual yield comparison (kWh)</div>', unsafe_allow_html=True)
        ann = kpis[["Site","Annual kWh","Specific yield (kWh/kWp)"]].copy()
        ann_sorted = ann.sort_values("Annual kWh", ascending=True)
        fig_ann = px.bar(
            ann_sorted, x="Annual kWh", y="Site", orientation="h",
            color="Specific yield (kWh/kWp)", color_continuous_scale="Viridis",
            text="Annual kWh", template="plotly_white", height=440
        )
        fig_ann.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_ann.update_layout(coloraxis_colorbar_title="kWh/kWp",
                              xaxis_title="Annual energy (kWh)", yaxis_title="",
                              margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_ann, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Terrain slope and grading
        st.markdown('<div class="sv-card"><div class="sv-title">Terrain — slope and grading</div>', unsafe_allow_html=True)
        slope_df = kpis[["Site","Pitch (m)","Table depth (m)","GCR","Slope (deg)","Aspect","Grading","Recommendation"]].copy()
        slope_sorted = slope_df.sort_values(by=["Slope (deg)"], ascending=True, na_position="last")
        st.dataframe(slope_sorted, use_container_width=True)

        import plotly.express as px
        fig_slope = px.bar(
            slope_sorted, x="Slope (deg)", y="Site", orientation="h",
            color="Grading",
            category_orders={"Grading": ["none","minor","major","unknown"]},
            color_discrete_map={"none":"#16a34a","minor":"#f59e0b","major":"#dc2626","unknown":"#64748b"},
            template="plotly_white", height=420
        )
        fig_slope.update_layout(xaxis_title="Slope (degrees)", yaxis_title="", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_slope, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Downloads
        st.markdown('<div class="sv-card"><div class="sv-title">Download results</div>', unsafe_allow_html=True)
        st.download_button("Per-site KPIs (CSV)",
                           kpis.to_csv(index=False).encode(),
                           file_name="site_kpis.csv", mime="text/csv")
        monthly_wide = monthly_all.pivot_table(index="MonthName", columns="Site", values="kWh", aggfunc="sum")
        monthly_wide = monthly_wide.reindex(month_labels)
        st.download_button("Monthly energy per site (CSV)",
                           monthly_wide.to_csv().encode(),
                           file_name="monthly_energy_per_site.csv", mime="text/csv")

# ------------------------------ Footer ------------------------------
st.markdown('<div class="sv-footer">Solar Versus © Eyob Aga Muleta — Built with pvlib, Plotly, and Streamlit</div>', unsafe_allow_html=True)

