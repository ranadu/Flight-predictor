import streamlit as st
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Performance Predictor", page_icon="🛩️", layout="wide")

# This page content is handled by the main app.py file
st.title("🛩️ Performance Predictor")
st.markdown("This page is integrated into the main dashboard. Please use the navigation in the main app.")
st.markdown("**[Go back to main dashboard](../)**")

# Add some standalone functionality for direct access
st.subheader("Quick Performance Check")

col1, col2 = st.columns(2)

with col1:
    st.selectbox("Aircraft Type", ["Boeing 737", "Airbus A320", "Boeing 777"])
    st.slider("Weight (tons)", 50, 300, 180)
    st.slider("Altitude (ft)", 25000, 45000, 35000)

with col2:
    st.slider("Speed (knots)", 300, 600, 450)
    st.slider("Temperature (°C)", -50, 30, -20)
    st.slider("Wind Speed (knots)", 0, 100, 20)

if st.button("Quick Predict"):
    st.success("Prediction completed! For full functionality, please use the main dashboard.")
