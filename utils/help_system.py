import streamlit as st

def show_help_tooltip(content, key=None):
    """Display a help tooltip for UI elements"""
    if st.button("❓", key=key, help=content):
        st.info(content)

def show_tutorial():
    """Display interactive tutorial for new users"""
    st.markdown("""
    ### 🎓 Welcome to Flight Predictor ML Dashboard
    
    **Quick Start Guide:**
    
    1. **Dashboard Overview**: View real-time flight metrics and system status
    2. **Performance Predictor**: Input aircraft parameters to predict performance
    3. **Flight Path Optimizer**: Optimize routes for fuel efficiency and time
    4. **Historical Analysis**: Analyze past flight data and trends
    
    **Pro Tips:**
    - Use Quick Presets for common scenarios
    - Enable auto-refresh for live monitoring
    - Save favorite configurations for quick access
    - Compare different optimization strategies
    """)

def show_feature_highlights():
    """Show key features and their benefits"""
    features = {
        "🤖 Neural Network Predictions": "Advanced ML models trained on realistic flight data",
        "🗺️ 3D Route Visualization": "Interactive 3D flight path with altitude profiles",
        "🌤️ Real-time Weather Integration": "Live atmospheric conditions affecting performance",
        "📊 Performance Analytics": "Comprehensive efficiency metrics and comparisons",
        "⚡ Quick Presets": "Pre-configured settings for common flight scenarios",
        "💾 Session Management": "Save and recall your favorite configurations"
    }
    
    for feature, description in features.items():
        st.markdown(f"**{feature}**: {description}")

def show_glossary():
    """Display aviation terms glossary"""
    terms = {
        "Fuel Efficiency": "Percentage of optimal fuel consumption achieved",
        "Time Efficiency": "How close actual flight time is to theoretical optimal",
        "Range Efficiency": "Maximum distance achievable with current fuel load",
        "Emission Efficiency": "CO2 emissions compared to industry standards",
        "Ground Speed": "Aircraft speed relative to the ground (affected by wind)",
        "True Airspeed": "Aircraft speed through the air mass",
        "Cruise Altitude": "Optimal altitude for fuel-efficient long-distance flight",
        "Wind Component": "Effect of wind on aircraft performance and route"
    }
    
    for term, definition in terms.items():
        with st.expander(f"📖 {term}"):
            st.write(definition)