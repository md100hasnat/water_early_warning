import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from datetime import datetime

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
# 3. SIDEBAR: SENSOR TELEMETRY & ALERTS
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
# 4. RISK ASSESSMENT ENGINE
# ==========================================
def evaluate_risk(ph_val, turb_val, tds_val, temp_val):
    risk_score = 0
    if ph_val < 6.5 or ph_val > 8.5:
        risk_score += 2
    if turb_val > 5.0:
        risk_score += 3
    if tds_val > 500:
        risk_score += 1
    if temp_val > 30.0:
        risk_score += 2

    if risk_score >= 4:
        return "OUTBREAK RISK", "CRITICAL", 0.10, 0.20, 0.70
    elif risk_score >= 2:
        return "MODERATE", "WARNING", 0.30, 0.60, 0.10
    else:
        return "SAFE", "LOW", 0.98, 0.02, 0.00

status, threat_level, prob_safe, prob_mod, prob_outbreak = evaluate_risk(ph, turbidity, tds, temp)
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
# 5. DASHBOARD TABS
# ==========================================
st.title("💧 Lower Mekong Outbreak Early Warning Engine")
st.caption("AI-Powered Multi-Station Water Quality Surveillance Platform")

tab1, tab2, tab3 = st.tabs([
    "🎛️ Single Sensor Diagnostic", 
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
    st.subheader("Risk Probabilities")
    st.write(f"Safe: {prob_safe * 100:.1f}%")
    st.progress(prob_safe)
    
    st.write(f"Moderate: {prob_mod * 100:.1f}%")
    st.progress(prob_mod)
    
    st.write(f"Outbreak Risk: {prob_outbreak * 100:.1f}%")
    st.progress(prob_outbreak)

# ------------------------------------------
# TAB 2: MULTI-STATION MAP
# ------------------------------------------
with tab2:
    st.subheader("Regional Monitoring Stations")
    
    # Mock geographic data for regional water stations
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
# TAB 3: INCIDENT LOGGING & EXPORT
# ------------------------------------------
with tab3:
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
        