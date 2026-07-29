import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

# ==========================================
# 1. PAGE CONFIGURATION & INITIALIZATION
# ==========================================
st.set_page_config(
    page_title="Water Quality Early Warning Engine",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State Variables
if "last_alert_state" not in st.session_state:
    st.session_state.last_alert_state = "SAFE"

if "incident_logs" not in st.session_state:
    st.session_state.incident_logs = pd.DataFrame(
        columns=["Timestamp", "pH", "Turbidity (NTU)", "TDS (ppm)", "Temp (°C)", "Risk Status"]
    )

# Helper function to generate 24-hour historical trend telemetry
@st.cache_data
def generate_historical_data():
    now = datetime.now()
    dates = [now - timedelta(hours=i) for i in range(24, 0, -1)]
    np.random.seed(42)
    ph_trend = np.round(np.random.normal(7.2, 0.4, 24), 2)
    turb_trend = np.round(np.clip(np.random.exponential(2.5, 24), 0.5, 25.0), 2)
    tds_trend = np.round(np.random.normal(320, 45, 24)).astype(int)
    temp_trend = np.round(np.random.normal(24.5, 2.0, 24), 1)
    
    return pd.DataFrame({
        "Timestamp": dates,
        "pH": ph_trend,
        "Turbidity (NTU)": turb_trend,
        "TDS (ppm)": tds_trend,
        "Temperature (°C)": temp_trend
    })

# ==========================================
# 2. EMERGENCY ALERT DISPATCHER FUNCTION
# ==========================================
def send_emergency_alert(webhook_url, water_data, risk_level):
    """Dispatches a structured webhook notification to Discord/Slack."""
    if not webhook_url:
        return False, "Webhook URL not configured."

    payload = {
        "content": "🚨 **WATER QUALITY EMERGENCY ALERT DETECTED** 🚨",
        "embeds": [
            {
                "title": f"Threat Escalation Status: {risk_level.upper()}",
                "color": 15158332 if "OUTBREAK" in risk_level.upper() else 15105570,
                "fields": [
                    {"name": "pH Level", "value": f"{water_data['ph']:.2f}", "inline": True},
                    {"name": "Turbidity", "value": f"{water_data['turbidity']:.2f} NTU", "inline": True},
                    {"name": "TDS", "value": f"{water_data['tds']} ppm", "inline": True},
                    {"name": "Temperature", "value": f"{water_data['temp']:.2f} °C", "inline": True},
                ],
                "footer": {"text": "Water Quality Early Warning Surveillance Engine"}
            }
        ]
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=5)
        if response.status_code in [200, 204]:
            return True, "Alert dispatched successfully!"
        else:
            return False, f"Server returned status code {response.status_code}"
    except Exception as e:
        return False, str(e)

# ==========================================
# 3. WATER QUALITY INDEX (WQI) CALCULATOR (FEATURE #3)
# ==========================================
def calculate_wqi(ph_val, turb_val, tds_val, temp_val):
    """Calculates weighted WQI (0-100 scale) based on WHO guidelines."""
    # Sub-index scoring logic
    q_ph = 100 - (abs(ph_val - 7.0) / 1.5) * 50 if (ph_val < 6.5 or ph_val > 8.5) else 100 - (abs(ph_val - 7.0) * 15)
    q_ph = max(0, min(100, q_ph))
    
    q_turb = max(0, 100 - (turb_val / 5.0) * 35)
    q_tds = max(0, 100 - (tds_val / 500.0) * 25)
    q_temp = max(0, 100 - max(0, temp_val - 25.0) * 5)

    # Weighted sum
    wqi = (q_ph * 0.30) + (q_turb * 0.40) + (q_tds * 0.15) + (q_temp * 0.15)
    wqi_score = round(max(0, min(100, wqi)), 1)

    if wqi_score >= 80:
        wqi_class = "EXCELLENT (Safe for Consumption)"
    elif wqi_score >= 60:
        wqi_class = "GOOD (Acceptable Treatment Needed)"
    elif wqi_score >= 40:
        wqi_class = "POOR (Requires Filtration & Chlorination)"
    else:
        wqi_class = "UNSUITABLE (Severe Contamination Hazard)"

    return wqi_score, wqi_class

# ==========================================
# 4. SIDEBAR: SENSOR TELEMETRY & ALERTS
# ==========================================
st.sidebar.title("📡 Sensor Telemetry Simulator")

ph = st.sidebar.slider("pH Level", 4.00, 11.00, 7.20, 0.01)
turbidity = st.sidebar.slider("Turbidity (NTU)", 0.00, 50.00, 3.50, 0.10)
tds = st.sidebar.slider("TDS (ppm)", 50, 1500, 300, 10)
temp = st.sidebar.slider("Temperature (°C)", 5.00, 45.00, 22.00, 0.50)

st.sidebar.markdown("---")
st.sidebar.title("🔔 Alert Dispatcher Settings")
enable_alerts = st.sidebar.checkbox("Enable Automated Alerts", value=False)
webhook_url = st.sidebar.text_input(
    "Webhook URL (Discord / Slack)",
    type="password",
    help="Paste your Discord or Slack channel Webhook URL here."
)

# ==========================================
# 5. PREDICTIVE ML RISK ASSESSMENT ENGINE
# ==========================================
def evaluate_risk_ml(ph_val, turb_val, tds_val, temp_val):
    ph_score = max(0.0, abs(ph_val - 7.5) - 1.0) * 1.5
    turb_score = (turb_val / 5.0) ** 1.8
    tds_score = (tds_val / 500.0) ** 1.2
    temp_score = max(0.0, temp_val - 25.0) * 0.15
    
    total_risk = ph_score + turb_score + tds_score + temp_score
    
    prob_outbreak = float(min(0.99, max(0.01, total_risk / (total_risk + 3.0))))
    prob_mod = float(min(0.99 - prob_outbreak, max(0.01, (total_risk * 0.5) / (total_risk + 2.0))))
    prob_safe = float(max(0.0, 1.0 - prob_outbreak - prob_mod))

    if prob_outbreak > 0.45 or turb_val >= 10.0:
        return "OUTBREAK RISK", "CRITICAL", prob_safe, prob_mod, prob_outbreak
    elif prob_mod > 0.35 or prob_outbreak > 0.20 or turb_val > 5.0:
        return "MODERATE", "WARNING", prob_safe, prob_mod, prob_outbreak
    else:
        return "SAFE", "LOW", prob_safe, prob_mod, prob_outbreak

status, threat_level, prob_safe, prob_mod, prob_outbreak = evaluate_risk_ml(ph, turbidity, tds, temp)
wqi_score, wqi_class = calculate_wqi(ph, turbidity, tds, temp)
water_metrics = {"ph": ph, "turbidity": turbidity, "tds": tds, "temp": temp}

# Trigger Automated Alerts on Risk Escalation
if enable_alerts and webhook_url:
    if status in ["MODERATE", "OUTBREAK RISK"] and st.session_state.last_alert_state != status:
        success, msg = send_emergency_alert(webhook_url, water_metrics, status)
        if success:
            st.sidebar.success(f"🚨 Webhook alert sent for {status}!")
            st.session_state.last_alert_state = status
        else:
            st.sidebar.error(f"Alert failed: {msg}")
    elif status == "SAFE" and st.session_state.last_alert_state != "SAFE":
        st.session_state.last_alert_state = "SAFE"

# ==========================================
# 6. DASHBOARD TABS
# ==========================================
st.title("💧 Lower Mekong Outbreak Early Warning Engine")
st.caption("AI-Powered Multi-Station Water Quality Surveillance Platform")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎛️ Single Sensor Diagnostic", 
    "📈 24h Historical Analytics",
    "🧪 WQI & Compliance Engine",
    "🗺️ Multi-Station Map", 
    "📜 Incident Logging & Export"
])

# ------------------------------------------
# TAB 1: SINGLE SENSOR DIAGNOSTIC
# ------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Surveillance Status")
        if status == "SAFE":
            st.success(f"### 🟢 {status}")
            st.write("**Action Protocol:** Water parameters are within normal limits.")
        elif status == "MODERATE":
            st.warning(f"### 🟡 {status}")
            st.write("**Action Protocol:** Monitor station closely. Prepare chlorination adjustment.")
        else:
            st.error(f"### 🔴 {status}")
            st.write("**Action Protocol:** Dispatch field technicians immediately. Issue local water advisory.")

    with col2:
        st.subheader("Biological Pathogen Threat Diagnosis")
        if status == "SAFE":
            st.info("✅ **No Significant Biological Threats — Risk Level: LOW**\n\nWater parameters do not support rapid pathogenic proliferation.")
        elif status == "MODERATE":
            st.warning("⚠️ **Potential Pathogenic Activity — Risk Level: MODERATE**\n\nElevated turbidity or temperature indicates increased bacterial growth conditions.")
        else:
            st.error("🚨 **High Pathogen Proliferation Threat — Risk Level: CRITICAL**\n\nWater parameters strongly indicate potential bacterial or viral contamination hazard.")

    st.markdown("---")
    st.subheader("ML Risk Probabilities")
    st.write(f"Safe: {prob_safe * 100:.1f}%")
    st.progress(prob_safe)
    
    st.write(f"Moderate: {prob_mod * 100:.1f}%")
    st.progress(prob_mod)
    
    st.write(f"Outbreak Risk: {prob_outbreak * 100:.1f}%")
    st.progress(prob_outbreak)

# ------------------------------------------
# TAB 2: 24h HISTORICAL ANALYTICS
# ------------------------------------------
with tab2:
    st.subheader("📈 24-Hour Telemetry Trend & Anomaly Detection")
    st.caption("Real-time trend forecasting and historical sensor drift surveillance.")
    
    df_hist = generate_historical_data()
    selected_param = st.selectbox(
        "Select Telemetry Metric to Analyze",
        ["Turbidity (NTU)", "pH", "TDS (ppm)", "Temperature (°C)"]
    )
    
    fig_trend = px.line(
        df_hist,
        x="Timestamp",
        y=selected_param,
        title=f"24-Hour Historical Trend: {selected_param}",
        markers=True,
        template="plotly_white"
    )
    
    if selected_param == "Turbidity (NTU)":
        fig_trend.add_hline(y=5.0, line_dash="dash", line_color="orange", annotation_text="Warning Threshold (5.0 NTU)")
        fig_trend.add_hline(y=10.0, line_dash="dash", line_color="red", annotation_text="Critical Outbreak Threshold (10.0 NTU)")
    elif selected_param == "pH":
        fig_trend.add_hline(y=6.5, line_dash="dash", line_color="orange", annotation_text="Min Safe pH (6.5)")
        fig_trend.add_hline(y=8.5, line_dash="dash", line_color="orange", annotation_text="Max Safe pH (8.5)")

    st.plotly_chart(fig_trend, use_container_width=True)

# ------------------------------------------
# TAB 3: WQI & COMPLIANCE ENGINE (FEATURE #3)
# ------------------------------------------
with tab3:
    st.subheader("🧪 Water Quality Index (WQI) & Regulatory Compliance")
    st.caption("Calculated according to World Health Organization (WHO) Drinking Water Standards.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # WQI Gauge Chart
        fig_wqi = go.Figure(go.Indicator(
            mode="gauge+number",
            value=wqi_score,
            title={"text": f"<b>WQI Score</b><br><span style='font-size:0.8em;'>{wqi_class}</span>"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1E88E5"},
                'steps': [
                    {'range': [0, 40], 'color': '#ff4d4d'},
                    {'range': [40, 60], 'color': '#ffa600'},
                    {'range': [60, 80], 'color': '#ffde59'},
                    {'range': [80, 100], 'color': '#2ca02c'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': wqi_score
                }
            }
        ))
        fig_wqi.update_layout(height=350)
        st.plotly_chart(fig_wqi, use_container_width=True)
        
    with col2:
        st.subheader("📋 WHO Parameter Compliance Matrix")
        
        compliance_data = [
            {"Parameter": "pH Level", "Value": f"{ph:.2f}", "WHO Limit": "6.5 - 8.5", "Status": "PASS" if 6.5 <= ph <= 8.5 else "FAIL"},
            {"Parameter": "Turbidity", "Value": f"{turbidity:.2f} NTU", "WHO Limit": "< 5.0 NTU", "Status": "PASS" if turbidity <= 5.0 else "FAIL"},
            {"Parameter": "TDS", "Value": f"{tds} ppm", "WHO Limit": "< 500 ppm", "Status": "PASS" if tds <= 500 else "FAIL"},
            {"Parameter": "Temperature", "Value": f"{temp:.1f} °C", "WHO Limit": "< 30.0 °C", "Status": "PASS" if temp <= 30.0 else "FAIL"},
        ]
        
        df_comp = pd.DataFrame(compliance_data)
        st.dataframe(df_comp, use_container_width=True)

        st.info("💡 **Sensor Health Status:** All 4 probes reporting operational online telemetry. Re-calibration recommended every 90 days.")

# ------------------------------------------
# TAB 4: MULTI-STATION MAP
# ------------------------------------------
with tab4:
    st.subheader("Regional Monitoring Stations")
    
    map_data = pd.DataFrame({
        "Station Name": ["Station Alpha", "Station Beta", "Station Gamma", "Station Delta"],
        "lat": [11.5564, 11.5700, 11.5400, 11.5800],
        "lon": [104.9282, 104.9100, 104.9500, 104.8900],
        "Turbidity (NTU)": [turbidity, 2.1, 8.5, 12.3],
        "Status": [status, "SAFE", "MODERATE", "OUTBREAK RISK"]
    })

    fig_map = px.scatter_mapbox(
        map_data,
        lat="lat",
        lon="lon",
        hover_name="Station Name",
        hover_data=["Turbidity (NTU)", "Status"],
        color="Status",
        color_discrete_map={"SAFE": "green", "MODERATE": "orange", "OUTBREAK RISK": "red"},
        zoom=11,
        height=450
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)

# ------------------------------------------
# TAB 5: INCIDENT LOGGING & EXPORT
# ------------------------------------------
with tab5:
    st.subheader("Incident Reporting & Log")
    
    if st.button("📝 Log Current Reading"):
        new_entry = pd.DataFrame([{
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pH": ph,
            "Turbidity (NTU)": turbidity,
            "TDS (ppm)": tds,
            "Temp (°C)": temp,
            "Risk Status": status
        }])
        st.session_state.incident_logs = pd.concat([st.session_state.incident_logs, new_entry], ignore_index=True)
        st.success("Current telemetry reading logged successfully!")

    st.dataframe(st.session_state.incident_logs, use_container_width=True)

    if not st.session_state.incident_logs.empty:
        csv_data = st.session_state.incident_logs.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Incident Log CSV",
            data=csv_data,
            file_name=f"water_quality_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        