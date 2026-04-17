import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Process Control System", layout="wide")

# -----------------------------
# STYLE
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0b1220;
    color: #e6edf3;
}

h1 {
    text-align: center;
    color: #00E5FF;
    font-weight: 800;
    letter-spacing: 2px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.markdown("""
<h1>RARE EARTH ELEMENTS RECOVERY FROM INDUSTRIAL WASTE</h1>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# DATASET (UNCHANGED)
# -----------------------------
data = [
[80,2.0,120,1,62],[90,1.0,60,1,47],[100,3.0,180,1,76],
[95,2.0,120,1,70],[110,4.0,240,1,79],[85,1.5,90,1,54],
[100,2.5,150,1,73],[105,3.5,180,1,77],[90,2.0,120,1,69],
[80,1.0,60,1,49],[95,3.0,180,1,74],[100,2.0,120,1,72],
[110,3.0,240,1,78],[85,2.0,90,1,57],[90,1.5,60,1,51],

[80,1.0,60,2,44],[90,2.0,120,2,61],[100,3.0,180,2,73],
[95,2.5,150,2,69],[110,4.0,240,2,80],[85,1.5,90,2,51],
[100,2.0,120,2,66],[105,3.0,180,2,71],[90,1.5,90,2,56],

[80,1.0,60,3,39],[90,2.0,120,3,56],[100,3.0,180,3,71],
[95,2.5,150,3,66],[110,4.0,240,3,83],[85,1.5,90,3,46],
[100,2.0,120,3,61],[105,3.5,180,3,77],[90,2.5,120,3,54]
]

df = pd.DataFrame(data, columns=["Temp","Acid","Time","WasteType","Recovery"])

# -----------------------------
# MODEL
# -----------------------------
X = df.drop("Recovery", axis=1)
y = df["Recovery"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=9,
    random_state=42
)

model.fit(X_train, y_train)

importance = model.feature_importances_

# -----------------------------
# INPUTS
# -----------------------------
st.markdown("### PROCESS PARAMETERS")

col1, col2, col3, col4 = st.columns(4)

with col1:
    temp = st.slider("Temperature (°C)", 60, 120, 90)

with col2:
    acid = st.slider("Acid Concentration (M)", 0.5, 4.0, 2.0)

with col3:
    time = st.slider("Time (min)", 30, 240, 120)

with col4:
    waste = st.selectbox("Industrial Waste Type", ["Red Mud","Fly Ash","Phosphogypsum"])

waste_map = {"Red Mud":1,"Fly Ash":2,"Phosphogypsum":3}

input_data = np.array([[temp, acid, time, waste_map[waste]]])
pred = model.predict(input_data)[0]

st.markdown("---")

# -----------------------------
# KPI
# -----------------------------
col1, col2 = st.columns(2)

col1.metric("RECOVERY OUTPUT", f"{pred:.2f} %")
col2.metric("WASTE STREAM", waste)

st.markdown("---")

# -----------------------------
# TAB SYSTEM
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "PROCESS BEHAVIOR",
    "SYSTEM RESPONSE",
    "LIVE INTERPRETATION"
])

# -----------------------------
# TAB 1 - CURVE (DYNAMIC)
# -----------------------------
with tab1:
    temps = np.arange(60, 121, 5)
    preds = []

    for t in temps:
        preds.append(model.predict([[t, acid, time, waste_map[waste]]])[0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=temps,
        y=preds,
        mode="lines+markers",
        line=dict(color="#00E5FF", width=3)
    ))

    fig.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TAB 2 - FEATURE RESPONSE (NOW INPUT-SENSITIVE)
# -----------------------------
with tab2:
    # local sensitivity simulation (dynamic approximation)
    base = model.predict([[temp, acid, time, waste_map[waste]]])[0]

    sensitivity = []
    labels = ["Temp Impact", "Acid Impact", "Time Impact", "Waste Impact"]

    # small perturbation method
    sensitivity.append(abs(model.predict([[temp+5, acid, time, waste_map[waste]]])[0] - base))
    sensitivity.append(abs(model.predict([[temp, acid+0.5, time, waste_map[waste]]])[0] - base))
    sensitivity.append(abs(model.predict([[temp, acid, time+30, waste_map[waste]]])[0] - base))
    sensitivity.append(abs(model.predict([[temp, acid, time, (waste_map[waste]%3)+1]])[0] - base))

    fig2 = go.Figure()
    fig2.add_bar(
        x=labels,
        y=sensitivity,
        marker_color=["#00E5FF","#FFB703","#8ECAE6","#FF006E"]
    )

    fig2.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# TAB 3 - LIVE SYSTEM INTERPRETATION
# -----------------------------
with tab3:
    interpretation = []

    if temp > 100:
        interpretation.append("High temperature regime → accelerated leaching kinetics")

    if acid > 3:
        interpretation.append("Strong acid environment → dissolution dominated regime")

    if time > 180:
        interpretation.append("Extended reaction time → nearing equilibrium state")

    if waste == "Red Mud":
        interpretation.append("Iron-rich matrix → complex leaching behavior")

    if pred > 75:
        interpretation.append("High recovery zone → optimal operating window")

    elif pred < 55:
        interpretation.append("Low efficiency zone → adjust process parameters")

    st.write("\n".join(["• " + i for i in interpretation]))
