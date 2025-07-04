import streamlit as st
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Flight Path Optimizer", page_icon="🛫", layout="wide")

# This page content is handled by the main app.py file
st.title("🛫 Flight Path Optimizer")
st.markdown("This page is integrated into the main dashboard. Please use the navigation in the main app.")
st.markdown("**[Go back to main dashboard](../)**")

# Add some standalone functionality for direct access
st.subheader("Quick Route Planning")

col1, col2 = st.columns(2)

with col1:
    st.selectbox("Origin", ["JFK", "LAX", "ORD", "DFW", "ATL"])
    st.selectbox("Destination", ["LAX", "JFK", "ORD", "DFW", "ATL"])

with col2:
    st.slider("Fuel Priority", 0.0, 1.0, 0.4)
    st.slider("Time Priority", 0.0, 1.0, 0.3)

if st.button("Quick Optimize"):
    st.success("Route optimization completed! For full functionality, please use the main dashboard.")
