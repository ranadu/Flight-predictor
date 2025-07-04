import streamlit as st
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Historical Data", page_icon="📊", layout="wide")

# This page content is handled by the main app.py file
st.title("📊 Historical Data Analysis")
st.markdown("This page is integrated into the main dashboard. Please use the navigation in the main app.")
st.markdown("**[Go back to main dashboard](../)**")

# Add some standalone functionality for direct access
st.subheader("Quick Data Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Flights", "1,247")

with col2:
    st.metric("Avg Efficiency", "85.2%")

with col3:
    st.metric("Data Points", "50,000+")

st.info("For detailed historical analysis and interactive charts, please use the main dashboard.")
