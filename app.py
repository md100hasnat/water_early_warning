import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
import joblib
import os
from datetime import datetime
from engine import train_engine, predict_water_safety, MODEL_FILE

# Page configuration
st.set_page_config(
    page_title="Early Warning Surveillance Engine",
    page_icon="💧",
    layout="wide"
)

# Load ML Model
@st.cache_resource
def load_warning_system():
    if not os.path.exists(MODEL_FILE):
        return train_engine()
    return joblib.load(MODEL_FILE)

model = load_warning_system()

# Helper: Pathogen Risk Diagnostic Logic
def diagnose_pathogens(ph, turbidity, tds, temp):
    suspected = []
    if turbidity > 15.0 and temp > 28.0:
        suspected.append(("Vibrio cholerae (Cholera)", "HIGH", "Bacterial bloom supported by elevated water temperature and high suspended solids."))
        suspected.append(("Escherichia coli (E. coli)", "HIGH", "High organic load/turbidity indicates potential fecal contamination."))
    elif turbidity > 5.0 and temp > 24.0:
        suspected.append(("Giardia lamblia", "MODERATE", "Moderate turbidity increases risk of protozoan cyst survival."))
    if ph < 6.0 or ph > 9.0:
        suspected.append(("Chemical Toxicity / Acid Stress", "HIGH", "pH out of safe physiological range; disrupts aquatic biological stability."))
    if tds > 1000:
        suspected.append(("Heavy Mineral / Saline Intrusion", "HIGH", "High dissolved solid concentrations, potential industrial runoff."))
    
    if not suspected:
        suspected.append(("No Significant Biological Threats", "LOW", "Water parameters do not support rapid pathogenic proliferation."))
    return suspected

st.title("💧 Low-Cost Water Quality & Outbreak Early Warning Engine")
st.caption("AI-Powered Multi-Station Environmental Surveillance & Epidemiological Risk Engine")

# Top Navigation Tabs
tab1, tab2, tab3 = st.tabs(["🎛️ Single Sensor Diagnostic", "🗺️ Multi-Station Map", "📜 Incident Logging & Export"])

# ==========================================
# TAB 1: SINGLE SENSOR DIAGNOSTIC
# ==========================================
with tab1:
    st.sidebar.header("📡 Sensor Telemetry Simulator")
    ph_input = st.sidebar.slider("pH Level", 4.0, 11.0, 7.2, 0.1)
    turbidity_input = st.sidebar.slider("Turbidity (NTU)", 0.0, 40.0, 3.5, 0.5)
    tds_input = st.sidebar.slider("TDS (ppm)", 0, 1500, 300, 10)
    temp_input = st.sidebar.slider("Temperature (°C)", 10.0, 40.0, 22.0, 0.5)

    status, icon, message, probs = predict_water_safety(
        model, ph_input, turbidity_input, tds_input, temp_input
    )

    col_status, col_metrics = st.columns([1, 2])

    with col_status:
        st.subheader("Surveillance Status")
        if status == "SAFE":
            st.success(f"### {icon} {status}")
        elif status == "MODERATE WARNING":
            st.warning(f"### {icon} {status}")
        else:
            st.error(f"### {icon} {status}")
        
        st.write(f"**Action Protocol:** {message}")
        st.markdown("---")
        st.markdown("#### Risk Probabilities")
        st.progress(float(probs[0]), text=f"Safe: {probs[0]*100:.1f}%")
        st.progress(float(probs[1]), text=f"Moderate: {probs[1]*100:.1f}%")
        st.progress(float(probs[2]), text=f"Outbreak Risk: {probs[2]*100:.1f}%")

    with col_metrics:
        st.subheader("Biological Pathogen Threat Diagnosis")
        pathogen_list = diagnose_pathogens(ph_input, turbidity_input, tds_input, temp_input)
        
        for name, threat, desc in pathogen_list:
            if threat == "HIGH":
                st.error(f"⚠️ **{name}** — Risk Level: {threat}\n\n*{desc}*")
            elif threat == "MODERATE":
                st.warning(f"⚡ **{name}** — Risk Level: {threat}\n\n*{desc}*")
            else:
                st.info(f"✅ **{name}** — Risk Level: {threat}\n\n*{desc}*")

# ==========================================
# TAB 2: MULTI-STATION MAP SURVEILLANCE
# ==========================================
with tab2:
    st.subheader("🗺️ Regional Monitoring Nodes")
    st.write("Simulated live network of low-cost IoT nodes deployed across community water points.")
    
    # Generate mock GIS station data
    stations_data = pd.DataFrame({
        'Station Name': ['Node 01 - Community Well', 'Node 02 - River Intake', 'Node 03 - School Tap', 'Node 04 - Reservoir East', 'Node 05 - Village Outlet'],
        'lat': [12.9716, 12.9820, 12.9600, 12.9900, 12.9500],
        'lon': [77.5946, 77.6050, 77.5800, 77.6200, 77.5700],
        'pH': [7.1, 5.8, 7.5, 8.9, 6.2],
        'Turbidity': [3.1, 22.4, 2.8, 18.0, 8.5],
        'TDS': [250, 1100, 310, 850, 520],
        'Temp': [21.0, 31.5, 22.0, 29.0, 26.0],
        'Status': ['SAFE 🟢', 'OUTBREAK ALERT 🔴', 'SAFE 🟢', 'OUTBREAK ALERT 🔴', 'MODERATE WARNING 🟡']
    })
    
    fig_map = px.scatter_mapbox(
        stations_data,
        lat="lat",
        lon="lon",
        hover_name="Station Name",
        hover_data=["Status", "pH", "Turbidity", "TDS", "Temp"],
        color="Status",
        color_discrete_map={
            'SAFE 🟢': 'green',
            'MODERATE WARNING 🟡': 'gold',
            'OUTBREAK ALERT 🔴': 'red'
        },
        zoom=11,
        height=450
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)
    
    st.dataframe(stations_data, use_container_width=True)

# ==========================================
# TAB 3: LOGGING & CSV EXPORT
# ==========================================
with tab3:
    st.subheader("📜 Historical Surveillance Logs")
    
    # Create or update session log
    if 'log_df' not in st.session_state:
        st.session_state.log_df = pd.DataFrame(columns=["Timestamp", "pH", "Turbidity", "TDS", "Temp", "Status"])
    
    if st.button("➕ Log Current Readings"):
        new_entry = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pH": ph_input,
            "Turbidity": turbidity_input,
            "TDS": tds_input,
            "Temp": temp_input,
            "Status": status
        }
        st.session_state.log_df = pd.concat([st.session_state.log_df, pd.DataFrame([new_entry])], ignore_index=True)
        st.success("Reading logged to current session record!")

    st.dataframe(st.session_state.log_df, use_container_width=True)
    
    if not st.session_state.log_df.empty:
        csv_data = st.session_state.log_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Surveillance Report (CSV)",
            data=csv_data,
            file_name=f"water_quality_log_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        