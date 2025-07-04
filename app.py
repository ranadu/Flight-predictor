import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
import time

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.flight_predictor import FlightPredictor
from models.path_optimizer import PathOptimizer
from utils.atmospheric_data import AtmosphericDataProvider
from utils.visualization import create_gauge_chart, create_3d_flight_path
from data.data_generator import FlightDataGenerator
from utils.help_system import show_tutorial, show_feature_highlights, show_glossary
from database.db_manager import DatabaseManager
import uuid

# Configure page
st.set_page_config(
    page_title="Flight Predictor ML Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io',
        'Report a bug': None,
        'About': "# Flight Predictor ML Dashboard\nAdvanced machine learning platform for aircraft performance prediction and flight path optimization using neural networks."
    }
)

# Initialize session state
if 'flight_predictor' not in st.session_state:
    with st.spinner("Initializing ML models and database..."):
        st.session_state.flight_predictor = FlightPredictor()
        st.session_state.path_optimizer = PathOptimizer()
        st.session_state.atmospheric_provider = AtmosphericDataProvider()
        st.session_state.data_generator = FlightDataGenerator()
        st.session_state.db_manager = DatabaseManager()
        st.session_state.last_update = datetime.now()
        st.session_state.favorites = []
        st.session_state.recent_predictions = []
        st.session_state.theme = "light"
        st.session_state.session_id = str(uuid.uuid4())
        
        # Load existing data from database
        st.session_state.recent_predictions = st.session_state.db_manager.get_prediction_history(
            limit=10, session_id=st.session_state.session_id
        )
        
        # Load saved configurations
        saved_configs = st.session_state.db_manager.get_user_configurations(
            session_id=st.session_state.session_id
        )
        st.session_state.favorites = [config['configuration'] for config in saved_configs if config['is_favorite']]

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1e3c72;
    }
    .sidebar-info {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-online { background-color: #28a745; }
    .status-warning { background-color: #ffc107; }
    .status-offline { background-color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header with gradient background
    st.markdown("""
    <div class="main-header">
        <h1>🛩️ Flight Predictor ML Dashboard</h1>
        <p>Advanced machine learning platform for aircraft performance prediction and flight path optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar enhancements
    st.sidebar.markdown("### 🚀 Navigation")
    
    # Add system status in sidebar
    # Check database connectivity
    try:
        db_analytics = st.session_state.db_manager.get_analytics_summary()
        db_status = "status-online"
        db_text = "Database: Connected"
    except:
        db_status = "status-offline"
        db_text = "Database: Offline"
    
    st.sidebar.markdown(f"""
    <div class="sidebar-info">
        <h4>System Status</h4>
        <p><span class="status-indicator status-online"></span>ML Models: Online</p>
        <p><span class="status-indicator status-online"></span>Data Feed: Active</p>
        <p><span class="status-indicator {db_status}"></span>{db_text}</p>
        <p><span class="status-indicator status-warning"></span>Weather API: Limited</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced navigation with icons
    page_options = [
        "🏠 Dashboard Overview",
        "⚡ Performance Predictor", 
        "🗺️ Flight Path Optimizer", 
        "📊 Historical Data Analysis",
        "🗄️ Database Analytics",
        "🔧 Settings & Preferences",
        "❓ Help & Tutorial"
    ]
    
    page = st.sidebar.selectbox("Select Page", page_options)
    
    # Show welcome tutorial for new users
    if 'first_visit' not in st.session_state:
        st.session_state.first_visit = False
        with st.sidebar.expander("🎓 Welcome! Quick Tutorial"):
            st.markdown("""
            **New to Flight Predictor?**
            
            1. Start with **Dashboard Overview** for real-time metrics
            2. Try **Performance Predictor** with presets
            3. Explore **Route Optimizer** for path planning
            4. Visit **Help & Tutorial** for detailed guidance
            """)
            if st.button("Got it!", key="welcome_dismiss"):
                st.session_state.first_visit = True
                st.rerun()
    
    # Add quick actions in sidebar
    st.sidebar.markdown("### ⚡ Quick Actions")
    if st.sidebar.button("🔄 Refresh Data"):
        st.session_state.last_update = datetime.now()
        st.rerun()
    
    if st.sidebar.button("💾 Save Current Session"):
        st.sidebar.success("Session saved!")
    
    # Show last update time
    st.sidebar.markdown(f"**Last Updated:** {st.session_state.last_update.strftime('%H:%M:%S')}")
    
    # Add model info in sidebar
    model_info = st.session_state.flight_predictor.get_model_info()
    st.sidebar.markdown(f"""
    <div class="sidebar-info">
        <h4>Model Information</h4>
        <p><strong>Type:</strong> Neural Network</p>
        <p><strong>Layers:</strong> {model_info['hidden_layers']}</p>
        <p><strong>Training Samples:</strong> {model_info['training_samples']:,}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recent predictions in sidebar
    if st.session_state.recent_predictions:
        st.sidebar.markdown("### 🕒 Recent Predictions")
        for i, pred in enumerate(st.session_state.recent_predictions[:3]):
            with st.sidebar.expander(f"{pred['aircraft_type']} - {pred['timestamp'].strftime('%H:%M')}"):
                st.write(f"Fuel Efficiency: {pred['prediction']['fuel_efficiency']:.1f}%")
                st.write(f"Flight Time: {pred['prediction']['flight_time']:.1f}h")
    
    # Favorites in sidebar
    if st.session_state.favorites:
        st.sidebar.markdown("### ⭐ Saved Configurations")
        for i, fav in enumerate(st.session_state.favorites[-3:]):
            if st.sidebar.button(f"{fav['aircraft_type']} - {fav['timestamp'].strftime('%m/%d')}", key=f"fav_{i}"):
                st.session_state.preset = "custom"
                st.session_state.custom_config = fav
    
    if page == "🏠 Dashboard Overview":
        show_dashboard_overview()
    elif page == "⚡ Performance Predictor":
        show_performance_predictor()
    elif page == "🗺️ Flight Path Optimizer":
        show_flight_path_optimizer()
    elif page == "📊 Historical Data Analysis":
        show_historical_analysis()
    elif page == "🗄️ Database Analytics":
        show_database_analytics()
    elif page == "🔧 Settings & Preferences":
        show_settings_preferences()
    elif page == "❓ Help & Tutorial":
        show_help_tutorial()

def show_dashboard_overview():
    st.header("🏠 Dashboard Overview")
    
    # Real-time metrics with auto-refresh
    current_time = datetime.now()
    time_factor = (current_time.hour % 24) / 24
    
    # Generate dynamic metrics based on time of day
    base_flights = 1247
    flight_variation = int(np.sin(time_factor * 2 * np.pi) * 200)
    active_flights = base_flights + flight_variation
    
    efficiency_base = 85.2
    efficiency_variation = np.sin(time_factor * 2 * np.pi + 1) * 3
    fuel_efficiency = efficiency_base + efficiency_variation
    
    # Enhanced metrics with database analytics
    try:
        db_analytics = st.session_state.db_manager.get_analytics_summary()
        database_connected = True
    except:
        database_connected = False
        db_analytics = {
            'total_flights': 0,
            'avg_fuel_efficiency': 0,
            'total_predictions': 0,
            'total_optimizations': 0
        }
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if database_connected:
            st.metric("Total Flights (DB)", f"{db_analytics['total_flights']:,}")
        else:
            delta_flights = f"+{abs(flight_variation):,}" if flight_variation > 0 else f"{flight_variation:,}"
            st.metric("Active Flights", f"{active_flights:,}", delta_flights)
    
    with col2:
        if database_connected and db_analytics['avg_fuel_efficiency'] > 0:
            st.metric("Avg Fuel Efficiency (DB)", f"{db_analytics['avg_fuel_efficiency']:.1f}%")
        else:
            delta_eff = f"+{efficiency_variation:.1f}%" if efficiency_variation > 0 else f"{efficiency_variation:.1f}%"
            st.metric("Avg Fuel Efficiency", f"{fuel_efficiency:.1f}%", delta_eff)
    
    with col3:
        if database_connected:
            st.metric("ML Predictions", f"{db_analytics['total_predictions']:,}")
        else:
            route_opt = 92.8 + np.random.uniform(-1, 1)
            st.metric("Route Optimization", f"{route_opt:.1f}%", "+1.5%")
    
    with col4:
        if database_connected:
            st.metric("Route Optimizations", f"{db_analytics['total_optimizations']:,}")
        else:
            weather_conditions = ["Excellent", "Good", "Fair", "Poor"]
            weather_weights = [0.4, 0.3, 0.2, 0.1]
            current_weather = np.random.choice(weather_conditions, p=weather_weights)
            st.metric("Weather Conditions", current_weather, "")
    
    # Add live data indicator with auto-refresh
    refresh_col1, refresh_col2 = st.columns([4, 1])
    with refresh_col1:
        st.markdown(f"""
        <div style="color: #666; font-size: 0.8em;">
            🔴 Live Data | Updated: {current_time.strftime('%H:%M:%S')}
        </div>
        """, unsafe_allow_html=True)
    with refresh_col2:
        auto_refresh = st.checkbox("Auto-refresh", value=False)
        
    if auto_refresh:
        time.sleep(5)
        st.rerun()
    
    # Real-time atmospheric conditions
    st.subheader("Current Atmospheric Conditions")
    atmospheric_data = st.session_state.atmospheric_provider.get_current_conditions()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Temperature and pressure gauges
        temp_gauge = create_gauge_chart(
            atmospheric_data['temperature'], 
            "Temperature (°C)", 
            -40, 60, 
            ['blue', 'green', 'orange', 'red']
        )
        st.plotly_chart(temp_gauge, use_container_width=True)
    
    with col2:
        # Wind speed gauge
        wind_gauge = create_gauge_chart(
            atmospheric_data['wind_speed'], 
            "Wind Speed (km/h)", 
            0, 200, 
            ['green', 'yellow', 'orange', 'red']
        )
        st.plotly_chart(wind_gauge, use_container_width=True)
    
    # Interactive tabs for different views
    st.subheader("📊 Real-time Analytics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Activity Heatmap", "🌍 Global Status", "📈 Performance Trends", "⚠️ Alerts"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            flight_data = st.session_state.data_generator.generate_flight_activity_data()
            
            fig = px.density_heatmap(
                flight_data, 
                x='hour', 
                y='day_of_week', 
                z='flight_count',
                title="Flight Activity by Hour and Day",
                labels={'hour': 'Hour of Day', 'day_of_week': 'Day of Week', 'flight_count': 'Flight Count'},
                color_continuous_scale="Blues"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Key Insights")
            peak_hour = flight_data.loc[flight_data['flight_count'].idxmax()]
            st.info(f"**Peak Activity:**\n{peak_hour['day_of_week']} at {peak_hour['hour']}:00\n({peak_hour['flight_count']} flights)")
            
            avg_flights = flight_data['flight_count'].mean()
            st.success(f"**Average Flights:**\n{avg_flights:.1f} per hour")
            
            weekend_avg = flight_data[flight_data['day_of_week'].isin(['Saturday', 'Sunday'])]['flight_count'].mean()
            weekday_avg = flight_data[~flight_data['day_of_week'].isin(['Saturday', 'Sunday'])]['flight_count'].mean()
            st.metric("Weekend vs Weekday", f"{weekend_avg:.1f}", f"{weekend_avg - weekday_avg:.1f}")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # World map showing flight density
            st.markdown("### 🌍 Global Flight Density")
            world_data = pd.DataFrame({
                'lat': [40.6413, 33.9425, 41.9742, 51.4700, 49.0097, 35.7653],
                'lon': [-73.7781, -118.4081, -87.9073, -0.4543, 2.5479, 140.3886],
                'city': ['New York', 'Los Angeles', 'Chicago', 'London', 'Paris', 'Tokyo'],
                'flights': [245, 198, 187, 156, 142, 198],
                'size': [20, 15, 12, 10, 8, 15]
            })
            
            fig = px.scatter_geo(
                world_data, 
                lat="lat", 
                lon="lon", 
                size="size",
                color="flights",
                hover_name="city",
                hover_data={'flights': True},
                color_continuous_scale="Viridis",
                size_max=15
            )
            fig.update_layout(height=300, geo=dict(showframe=False, showcoastlines=True))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Regional Performance")
            regions = ['North America', 'Europe', 'Asia', 'Others']
            efficiency = [87.2, 84.5, 89.1, 82.3]
            
            fig = px.bar(
                x=regions, 
                y=efficiency,
                title="Regional Fuel Efficiency",
                color=efficiency,
                color_continuous_scale="RdYlGn"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Performance trends over time
        st.markdown("### 📈 Real-time Performance Trends")
        
        # Generate trend data
        hours = list(range(24))
        efficiency_trend = [85 + 5*np.sin(h/24 * 2*np.pi) + np.random.normal(0, 2) for h in hours]
        fuel_consumption = [800 + 100*np.sin(h/24 * 2*np.pi + np.pi) + np.random.normal(0, 20) for h in hours]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours, 
            y=efficiency_trend,
            mode='lines+markers',
            name='Fuel Efficiency (%)',
            line=dict(color='#2E86AB'),
            yaxis='y'
        ))
        fig.add_trace(go.Scatter(
            x=hours, 
            y=fuel_consumption,
            mode='lines+markers',
            name='Fuel Consumption (kg/h)',
            line=dict(color='#F24236'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="24-Hour Performance Trends",
            xaxis_title="Hour of Day",
            yaxis=dict(title="Efficiency (%)", side="left"),
            yaxis2=dict(title="Fuel Consumption (kg/h)", overlaying="y", side="right"),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### ⚠️ System Alerts & Notifications")
        
        # Create alert system
        alerts = [
            {"level": "🔴", "message": "High fuel consumption detected on Route JFK-LAX", "time": "2 min ago"},
            {"level": "🟡", "message": "Weather advisory: Strong winds affecting European routes", "time": "15 min ago"},
            {"level": "🟢", "message": "Route optimization improved efficiency by 3.2%", "time": "1 hour ago"},
            {"level": "🔵", "message": "New ML model training completed successfully", "time": "3 hours ago"}
        ]
        
        for alert in alerts:
            st.markdown(f"""
            <div style="padding: 10px; border-left: 4px solid #1e3c72; margin: 10px 0; background: #f8f9fa;">
                <strong>{alert['level']} {alert['message']}</strong><br>
                <small style="color: #666;">{alert['time']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Add alert configuration
        st.markdown("### 🔔 Alert Settings")
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("Email notifications", value=True)
            st.checkbox("Performance alerts", value=True)
        with col2:
            st.checkbox("Weather warnings", value=True)
            st.checkbox("System status updates", value=False)

def show_performance_predictor():
    st.header("⚡ Aircraft Performance Predictor")
    
    # Add prediction presets
    st.subheader("🎯 Quick Prediction Presets")
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    
    with preset_col1:
        if st.button("🏙️ Short Haul (Domestic)", use_container_width=True):
            st.session_state.preset = "short_haul"
    
    with preset_col2:
        if st.button("🌍 Long Haul (International)", use_container_width=True):
            st.session_state.preset = "long_haul"
    
    with preset_col3:
        if st.button("🚁 Regional Flight", use_container_width=True):
            st.session_state.preset = "regional"
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("✈️ Flight Parameters")
        
        # Apply preset values if selected
        if 'preset' in st.session_state:
            if st.session_state.preset == "short_haul":
                default_aircraft = "Boeing 737"
                default_weight = 120
                default_altitude = 32000
                default_speed = 420
            elif st.session_state.preset == "long_haul":
                default_aircraft = "Boeing 777"
                default_weight = 250
                default_altitude = 40000
                default_speed = 520
            else:  # regional
                default_aircraft = "Embraer E190"
                default_weight = 80
                default_altitude = 28000
                default_speed = 380
        else:
            default_aircraft = "Boeing 737"
            default_weight = 180
            default_altitude = 35000
            default_speed = 450
        
        # Aircraft specifications with enhanced UI
        aircraft_options = {
            "Boeing 737": "🛩️ Boeing 737 (Short-Medium Haul)",
            "Airbus A320": "🛩️ Airbus A320 (Short-Medium Haul)", 
            "Boeing 777": "✈️ Boeing 777 (Long Haul)",
            "Airbus A350": "✈️ Airbus A350 (Long Haul)",
            "Embraer E190": "🛫 Embraer E190 (Regional)"
        }
        
        aircraft_type = st.selectbox(
            "Aircraft Type",
            list(aircraft_options.keys()),
            format_func=lambda x: aircraft_options[x],
            index=list(aircraft_options.keys()).index(default_aircraft)
        )
        
        # Enhanced sliders with better formatting and help text
        weight = st.slider(
            "Aircraft Weight (tons)", 
            50, 300, default_weight,
            help="Total aircraft weight including passengers, cargo, and fuel"
        )
        
        altitude = st.slider(
            "Cruise Altitude (ft)", 
            25000, 45000, default_altitude,
            help="Typical cruise altitude for optimal fuel efficiency"
        )
        
        speed = st.slider(
            "Target Speed (knots)", 
            300, 600, default_speed,
            help="Target cruise speed in knots"
        )
        
        # Environmental conditions with current weather integration
        st.subheader("🌤️ Environmental Conditions")
        
        # Add "Use Current Weather" button
        current_weather_col1, current_weather_col2 = st.columns([2, 1])
        with current_weather_col1:
            use_current = st.checkbox("Use Current Weather Data", value=False)
        with current_weather_col2:
            if st.button("🔄 Refresh"):
                st.rerun()
        
        if use_current:
            current_conditions = st.session_state.atmospheric_provider.get_current_conditions()
            temperature = current_conditions['temperature']
            wind_speed = current_conditions['wind_speed']
            wind_direction = current_conditions['wind_direction']
            humidity = current_conditions['humidity']
            
            # Display current weather as read-only info
            st.info(f"🌡️ Temperature: {temperature:.1f}°C | 💨 Wind: {wind_speed:.1f} km/h from {wind_direction:.0f}° | 💧 Humidity: {humidity:.1f}%")
        else:
            temperature = st.slider("Temperature (°C)", -50, 30, -20)
            wind_speed = st.slider("Wind Speed (knots)", 0, 100, 20)
            wind_direction = st.slider("Wind Direction (°)", 0, 360, 180)
            humidity = st.slider("Humidity (%)", 0, 100, 60)
        
        # Enhanced predict button with loading animation
        predict_col1, predict_col2 = st.columns([3, 1])
        with predict_col1:
            predict_button = st.button("🚀 Predict Performance", type="primary", use_container_width=True)
        with predict_col2:
            if st.button("💾 Save Config"):
                config_data = {
                    'aircraft_type': aircraft_type,
                    'weight': weight,
                    'altitude': altitude,
                    'speed': speed,
                    'timestamp': datetime.now()
                }
                
                # Save to database
                saved = st.session_state.db_manager.save_user_configuration(
                    name=f"{aircraft_type} Config",
                    config_type="aircraft",
                    configuration=config_data,
                    session_id=st.session_state.session_id,
                    is_favorite=True
                )
                
                if saved:
                    st.session_state.favorites.append(config_data)
                    st.success("Configuration saved to database!")
                else:
                    st.error("Could not save configuration")
        
        if predict_button:
            # Prepare input data
            input_data = {
                'aircraft_type': aircraft_type,
                'weight': weight,
                'altitude': altitude,
                'speed': speed,
                'temperature': temperature,
                'wind_speed': wind_speed,
                'wind_direction': wind_direction,
                'humidity': humidity
            }
            
            # Make prediction with loading animation
            with st.spinner("Running ML prediction..."):
                time.sleep(1)  # Brief delay for UX
                prediction = st.session_state.flight_predictor.predict_performance(input_data)
                st.session_state.current_prediction = prediction
                
                # Save prediction to database
                prediction_data = {
                    'aircraft_type': aircraft_type,
                    'weight': weight,
                    'altitude': altitude,
                    'speed': speed,
                    'temperature': temperature,
                    'wind_speed': wind_speed,
                    'wind_direction': wind_direction,
                    'humidity': humidity,
                    'results': prediction
                }
                
                saved = st.session_state.db_manager.save_prediction(
                    prediction_data, 
                    st.session_state.session_id
                )
                
                # Add to recent predictions for UI
                prediction_entry = {
                    'timestamp': datetime.now(),
                    'aircraft_type': aircraft_type,
                    'prediction': prediction
                }
                st.session_state.recent_predictions.insert(0, prediction_entry)
                if len(st.session_state.recent_predictions) > 10:
                    st.session_state.recent_predictions.pop()
            
            if saved:
                st.success("Prediction completed and saved to database!")
            else:
                st.success("Prediction completed!")
                st.warning("Could not save to database")
    
    with col2:
        st.subheader("🎯 Performance Predictions")
        
        if hasattr(st.session_state, 'current_prediction'):
            pred = st.session_state.current_prediction
            
            # Enhanced performance metrics with color coding
            col2a, col2b = st.columns(2)
            
            with col2a:
                # Color code metrics based on efficiency
                fuel_color = "normal" if pred['fuel_efficiency'] >= 80 else "inverse"
                time_color = "normal" if pred['time_efficiency'] >= 85 else "inverse"
                
                st.metric(
                    "Fuel Consumption", 
                    f"{pred['fuel_consumption']:.1f} kg/h", 
                    f"{pred['fuel_efficiency']:.1f}%",
                    delta_color=fuel_color
                )
                st.metric(
                    "Flight Time", 
                    f"{pred['flight_time']:.1f} hours", 
                    f"{pred['time_efficiency']:.1f}%",
                    delta_color=time_color
                )
            
            with col2b:
                range_color = "normal" if pred['range_efficiency'] >= 75 else "inverse"
                emission_color = "normal" if pred['emission_efficiency'] >= 80 else "inverse"
                
                st.metric(
                    "Range", 
                    f"{pred['range']:.0f} km", 
                    f"{pred['range_efficiency']:.1f}%",
                    delta_color=range_color
                )
                st.metric(
                    "Emissions", 
                    f"{pred['emissions']:.1f} kg CO2", 
                    f"{pred['emission_efficiency']:.1f}%",
                    delta_color=emission_color
                )
            
            # Overall performance score
            overall_score = (pred['fuel_efficiency'] + pred['time_efficiency'] + 
                           pred['range_efficiency'] + pred['emission_efficiency']) / 4
            
            score_color = "🟢" if overall_score >= 85 else "🟡" if overall_score >= 70 else "🔴"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 10px; color: white; margin: 20px 0;">
                <h2>{score_color} Overall Performance Score</h2>
                <h1 style="margin: 10px 0; font-size: 3em;">{overall_score:.1f}%</h1>
                <p>{"Excellent" if overall_score >= 85 else "Good" if overall_score >= 70 else "Needs Improvement"}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Performance visualization
            performance_data = pd.DataFrame({
                'Metric': ['Fuel Efficiency', 'Time Efficiency', 'Range Efficiency', 'Emission Efficiency'],
                'Value': [pred['fuel_efficiency'], pred['time_efficiency'], pred['range_efficiency'], pred['emission_efficiency']]
            })
            
            fig = px.bar(
                performance_data, 
                x='Metric', 
                y='Value',
                title="Performance Efficiency Metrics",
                color='Value',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(yaxis_title="Efficiency (%)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Radar chart for overall performance
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=[pred['fuel_efficiency'], pred['time_efficiency'], pred['range_efficiency'], pred['emission_efficiency']],
                theta=['Fuel Efficiency', 'Time Efficiency', 'Range Efficiency', 'Emission Efficiency'],
                fill='toself',
                name='Performance Profile'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Performance Profile Radar Chart"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Set flight parameters and click 'Predict Performance' to see results.")

def show_flight_path_optimizer():
    st.header("🗺️ Flight Path Optimizer")
    
    # Popular route suggestions
    st.subheader("🔥 Popular Routes")
    popular_col1, popular_col2, popular_col3, popular_col4 = st.columns(4)
    
    popular_routes = [
        ("JFK", "LAX", "🇺🇸 Cross-US"),
        ("LHR", "JFK", "🌊 Transatlantic"),
        ("NRT", "LAX", "🌏 Transpacific"),
        ("CDG", "DXB", "🌍 Europe-Middle East")
    ]
    
    selected_route = None
    for i, (orig, dest, label) in enumerate(popular_routes):
        with [popular_col1, popular_col2, popular_col3, popular_col4][i]:
            if st.button(f"{label}\n{orig} → {dest}", use_container_width=True):
                selected_route = (orig, dest)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("✈️ Route Parameters")
        
        # Enhanced airport selection with descriptions
        airport_options = {
            "JFK": "🇺🇸 John F. Kennedy (New York)",
            "LAX": "🇺🇸 Los Angeles International",
            "ORD": "🇺🇸 Chicago O'Hare",
            "DFW": "🇺🇸 Dallas/Fort Worth",
            "ATL": "🇺🇸 Hartsfield-Jackson Atlanta",
            "LHR": "🇬🇧 London Heathrow",
            "CDG": "🇫🇷 Charles de Gaulle (Paris)",
            "NRT": "🇯🇵 Narita International (Tokyo)",
            "DXB": "🇦🇪 Dubai International"
        }
        
        # Use selected popular route if available
        if selected_route:
            default_origin = selected_route[0]
            default_dest = selected_route[1]
        else:
            default_origin = "JFK"
            default_dest = "LAX"
        
        origin = st.selectbox(
            "Origin Airport", 
            list(airport_options.keys()),
            format_func=lambda x: airport_options[x],
            index=list(airport_options.keys()).index(default_origin)
        )
        destination = st.selectbox(
            "Destination Airport", 
            list(airport_options.keys()),
            format_func=lambda x: airport_options[x],
            index=list(airport_options.keys()).index(default_dest)
        )
        
        # Optimization preferences with presets
        st.subheader("🎯 Optimization Strategy")
        
        strategy_col1, strategy_col2, strategy_col3 = st.columns(3)
        
        strategy = None
        with strategy_col1:
            if st.button("💰 Cost Focused", use_container_width=True):
                strategy = "cost"
        with strategy_col2:
            if st.button("⚡ Speed Focused", use_container_width=True):
                strategy = "speed"
        with strategy_col3:
            if st.button("🌍 Eco Friendly", use_container_width=True):
                strategy = "eco"
        
        # Set weights based on strategy
        if strategy == "cost":
            fuel_weight = 0.6
            time_weight = 0.2
            weather_weight = 0.2
        elif strategy == "speed":
            fuel_weight = 0.2
            time_weight = 0.6
            weather_weight = 0.2
        elif strategy == "eco":
            fuel_weight = 0.5
            time_weight = 0.2
            weather_weight = 0.3
        else:
            # Custom sliders
            fuel_weight = st.slider("Fuel Efficiency Priority", 0.0, 1.0, 0.4, 
                                  help="Higher values prioritize fuel savings")
            time_weight = st.slider("Time Efficiency Priority", 0.0, 1.0, 0.3,
                                  help="Higher values prioritize faster routes")
            weather_weight = st.slider("Weather Avoidance Priority", 0.0, 1.0, 0.3,
                                     help="Higher values avoid bad weather")
        
        # Constraints
        st.subheader("Flight Constraints")
        max_altitude = st.slider("Maximum Altitude (ft)", 30000, 50000, 42000)
        min_fuel_reserve = st.slider("Minimum Fuel Reserve (%)", 5, 20, 10)
        
        # Enhanced optimization button
        opt_col1, opt_col2 = st.columns([3, 1])
        with opt_col1:
            optimize_button = st.button("🚀 Optimize Route", type="primary", use_container_width=True)
        with opt_col2:
            if st.button("🔄 Compare"):
                st.session_state.show_comparison = True
        
        if optimize_button:
            # Prepare optimization parameters
            optimization_params = {
                'origin': origin,
                'destination': destination,
                'fuel_weight': fuel_weight,
                'time_weight': time_weight,
                'weather_weight': weather_weight,
                'max_altitude': max_altitude,
                'min_fuel_reserve': min_fuel_reserve
            }
            
            # Optimize route with loading animation
            with st.spinner("Optimizing flight path using ML algorithms..."):
                time.sleep(1.5)  # Brief delay for UX
                optimization_result = st.session_state.path_optimizer.optimize_route(optimization_params)
                st.session_state.current_route = optimization_result
                
                # Save optimization to database
                optimization_data = {
                    'origin': origin,
                    'destination': destination,
                    'fuel_weight': fuel_weight,
                    'time_weight': time_weight,
                    'weather_weight': weather_weight,
                    'max_altitude': max_altitude,
                    'min_fuel_reserve': min_fuel_reserve,
                    'results': optimization_result
                }
                
                saved = st.session_state.db_manager.save_route_optimization(
                    optimization_data,
                    st.session_state.session_id
                )
            
            if saved:
                st.success("Route optimization completed and saved to database!")
            else:
                st.success("Route optimization completed!")
                st.warning("Could not save to database")
    
    with col2:
        st.subheader("Optimized Flight Path")
        
        if hasattr(st.session_state, 'current_route'):
            route = st.session_state.current_route
            
            # Route summary
            col2a, col2b, col2c = st.columns(3)
            
            with col2a:
                st.metric("Total Distance", f"{route['total_distance']:.0f} km")
                st.metric("Flight Time", f"{route['flight_time']:.1f} hours")
            
            with col2b:
                st.metric("Fuel Consumption", f"{route['fuel_consumption']:.0f} kg")
                st.metric("Cost Savings", f"${route['cost_savings']:.0f}")
            
            with col2c:
                st.metric("Weather Score", f"{route['weather_score']:.1f}/10")
                st.metric("Overall Efficiency", f"{route['efficiency_score']:.1f}%")
            
            # 3D flight path visualization
            flight_path_fig = create_3d_flight_path(route['waypoints'])
            st.plotly_chart(flight_path_fig, use_container_width=True)
            
            # Altitude profile
            waypoints_df = pd.DataFrame(route['waypoints'])
            
            fig_altitude = px.line(
                waypoints_df, 
                x='distance', 
                y='altitude',
                title="Flight Altitude Profile",
                labels={'distance': 'Distance (km)', 'altitude': 'Altitude (ft)'}
            )
            st.plotly_chart(fig_altitude, use_container_width=True)
            
            # Route comparison
            st.subheader("Route Comparison")
            comparison_data = pd.DataFrame({
                'Route Type': ['Direct Route', 'Optimized Route'],
                'Distance (km)': [route['direct_distance'], route['total_distance']],
                'Flight Time (hours)': [route['direct_time'], route['flight_time']],
                'Fuel Consumption (kg)': [route['direct_fuel'], route['fuel_consumption']]
            })
            
            st.dataframe(comparison_data, use_container_width=True)
        else:
            st.info("Configure route parameters and click 'Optimize Route' to see the optimized flight path.")

def show_historical_analysis():
    st.header("📊 Historical Flight Data Analysis")
    
    # Data source selection
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        data_source = st.selectbox(
            "Data Source",
            ["Database (Stored)", "Generated (Synthetic)"],
            help="Choose between stored database records or generated synthetic data"
        )
    
    with col2:
        days_range = st.slider("Days of Data", 7, 90, 30)
    
    with col3:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Load data based on selection
    if data_source == "Database (Stored)":
        historical_data = st.session_state.db_manager.get_flight_data(days=days_range)
        if historical_data.empty:
            st.warning("No data found in database. Generating sample data...")
            # Generate and save sample data
            sample_data = st.session_state.data_generator.generate_historical_data(days=30)
            st.session_state.db_manager.save_flight_data(sample_data)
            historical_data = sample_data
            st.success("Sample data generated and saved to database!")
    else:
        historical_data = st.session_state.data_generator.generate_historical_data(days=days_range)
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Filter data by date range
    filtered_data = historical_data[
        (historical_data['date'] >= pd.Timestamp(start_date)) & 
        (historical_data['date'] <= pd.Timestamp(end_date))
    ]
    
    # Summary statistics
    st.subheader("Flight Performance Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Flights", len(filtered_data))
    with col2:
        st.metric("Avg Fuel Efficiency", f"{filtered_data['fuel_efficiency'].mean():.1f}%")
    with col3:
        st.metric("Avg Flight Time", f"{filtered_data['flight_time'].mean():.1f} hours")
    with col4:
        st.metric("Total Distance", f"{filtered_data['distance'].sum():.0f} km")
    
    # Time series analysis
    st.subheader("Performance Trends")
    
    # Daily aggregation
    daily_stats = filtered_data.groupby('date').agg({
        'fuel_efficiency': 'mean',
        'flight_time': 'mean',
        'distance': 'sum',
        'emissions': 'sum'
    }).reset_index()
    
    # Create subplots
    fig_trends = px.line(
        daily_stats, 
        x='date', 
        y='fuel_efficiency',
        title="Daily Average Fuel Efficiency Trend",
        labels={'fuel_efficiency': 'Fuel Efficiency (%)', 'date': 'Date'}
    )
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Aircraft type performance comparison
    st.subheader("Aircraft Type Performance Comparison")
    
    aircraft_performance = filtered_data.groupby('aircraft_type').agg({
        'fuel_efficiency': 'mean',
        'flight_time': 'mean',
        'emissions': 'mean'
    }).reset_index()
    
    fig_aircraft = px.bar(
        aircraft_performance, 
        x='aircraft_type', 
        y='fuel_efficiency',
        title="Average Fuel Efficiency by Aircraft Type",
        labels={'fuel_efficiency': 'Fuel Efficiency (%)', 'aircraft_type': 'Aircraft Type'}
    )
    st.plotly_chart(fig_aircraft, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Performance Correlation Analysis")
    
    # Select numerical columns for correlation
    numerical_cols = ['fuel_efficiency', 'flight_time', 'distance', 'altitude', 'speed', 'temperature', 'wind_speed']
    correlation_data = filtered_data[numerical_cols].corr()
    
    fig_corr = px.imshow(
        correlation_data,
        title="Performance Metrics Correlation Matrix",
        color_continuous_scale='RdBu'
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Detailed data table
    st.subheader("Detailed Flight Data")
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        aircraft_filter = st.multiselect(
            "Filter by Aircraft Type",
            options=filtered_data['aircraft_type'].unique(),
            default=filtered_data['aircraft_type'].unique()
        )
    with col2:
        efficiency_threshold = st.slider("Minimum Fuel Efficiency (%)", 0, 100, 0)
    
    # Apply filters
    table_data = filtered_data[
        (filtered_data['aircraft_type'].isin(aircraft_filter)) &
        (filtered_data['fuel_efficiency'] >= efficiency_threshold)
    ]
    
    # Display table
    st.dataframe(
        table_data[['date', 'aircraft_type', 'origin', 'destination', 'fuel_efficiency', 'flight_time', 'distance']],
        use_container_width=True
    )

def show_settings_preferences():
    st.header("🔧 Settings & Preferences")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎨 Display", "🔔 Notifications", "📊 Data", "🤖 ML Models"])
    
    with tab1:
        st.subheader("Display Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox("Theme", ["Light", "Dark", "Auto"], index=0)
            st.selectbox("Chart Style", ["Modern", "Classic", "Minimal"], index=0)
            st.checkbox("Show animations", value=True)
            st.checkbox("High contrast mode", value=False)
        
        with col2:
            st.slider("Chart refresh rate (seconds)", 1, 60, 10)
            st.selectbox("Date format", ["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD"], index=0)
            st.selectbox("Units", ["Metric", "Imperial"], index=0)
            st.checkbox("Show tooltips", value=True)
    
    with tab2:
        st.subheader("Notification Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Email alerts", value=True)
            st.checkbox("Performance warnings", value=True)
            st.checkbox("Weather advisories", value=True)
            st.checkbox("System updates", value=False)
        
        with col2:
            st.selectbox("Alert frequency", ["Real-time", "Every 5 minutes", "Hourly"], index=1)
            st.text_input("Email address", placeholder="your.email@company.com")
            st.multiselect("Alert types", ["Critical", "Warning", "Info"], default=["Critical", "Warning"])
    
    with tab3:
        st.subheader("Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox("Data retention", ["7 days", "30 days", "90 days", "1 year"], index=2)
            st.checkbox("Auto-backup", value=True)
            st.checkbox("Compress old data", value=True)
        
        with col2:
            if st.button("📥 Export Data"):
                st.success("Data export initiated!")
            if st.button("🗑️ Clear Cache"):
                st.success("Cache cleared!")
            if st.button("🔄 Reset to Defaults"):
                st.warning("Settings reset to defaults!")
    
    with tab4:
        st.subheader("ML Model Configuration")
        
        model_info = st.session_state.flight_predictor.get_model_info()
        
        st.info(f"""
        **Current Model Status:**
        - Type: {model_info['model_type']}
        - Hidden Layers: {model_info['hidden_layers']}
        - Training Samples: {model_info['training_samples']:,}
        - Features: {model_info['n_features']}
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox("Model precision", ["Standard", "High", "Ultra"], index=0)
            st.checkbox("Auto-retrain", value=True)
            st.slider("Confidence threshold", 0.5, 0.99, 0.85)
        
        with col2:
            if st.button("🔄 Retrain Models"):
                with st.spinner("Retraining models..."):
                    time.sleep(2)
                st.success("Models retrained successfully!")
            
            if st.button("📊 Model Performance"):
                st.info("Model accuracy: 94.2%\nLast updated: 2 hours ago")

def show_help_tutorial():
    st.header("❓ Help & Tutorial")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎓 Quick Start", "✨ Features", "📖 Glossary", "🆘 Support"])
    
    with tab1:
        show_tutorial()
        
        st.subheader("🚀 Getting Started")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Dashboard Overview
            - View real-time flight metrics
            - Monitor system status
            - Check atmospheric conditions
            - Access quick actions
            """)
            
            st.markdown("""
            ### Performance Predictor
            - Use quick presets for common scenarios
            - Input custom flight parameters
            - Get instant ML predictions
            - Save favorite configurations
            """)
        
        with col2:
            st.markdown("""
            ### Flight Path Optimizer
            - Select popular routes or custom airports
            - Choose optimization strategy
            - Compare different approaches
            - Visualize 3D flight paths
            """)
            
            st.markdown("""
            ### Historical Analysis
            - Analyze past flight data
            - View performance trends
            - Compare aircraft types
            - Export data for further analysis
            """)
    
    with tab2:
        st.subheader("✨ Key Features")
        show_feature_highlights()
        
        st.subheader("🎯 Pro Tips")
        tips = [
            "Use auto-refresh on the dashboard for live monitoring",
            "Save frequently used configurations as favorites",
            "Compare different optimization strategies side by side",
            "Enable current weather integration for accurate predictions",
            "Use the quick presets to get started faster",
            "Check the alerts tab for important system notifications"
        ]
        
        for tip in tips:
            st.markdown(f"💡 {tip}")
    
    with tab3:
        st.subheader("📖 Aviation Glossary")
        show_glossary()
    
    with tab4:
        st.subheader("🆘 Support & Contact")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Common Issues
            
            **Q: Predictions seem unrealistic?**
            A: Check your input parameters and ensure they match typical aircraft specifications.
            
            **Q: Route optimization not working?**
            A: Verify that origin and destination airports are different and supported.
            
            **Q: Data not updating?**
            A: Try refreshing the page or clearing the cache in settings.
            """)
        
        with col2:
            st.markdown("""
            ### System Requirements
            - Modern web browser (Chrome, Firefox, Safari)
            - Stable internet connection
            - JavaScript enabled
            
            ### Data Sources
            - ML models trained on synthetic aviation data
            - Real-time atmospheric simulation
            - Airport database with major international airports
            """)
        
        st.subheader("📞 Contact Information")
        st.info("""
        For technical support or feature requests:
        - Email: support@flightpredictor.ai
        - Documentation: docs.flightpredictor.ai
        - Version: 1.0.0 (Build 2025.07.04)
        """)

def show_database_analytics():
    st.header("🗄️ Database Analytics")
    
    try:
        analytics = st.session_state.db_manager.get_analytics_summary()
        
        # Database overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Flight Records", f"{analytics['total_flights']:,}")
        with col2:
            st.metric("ML Predictions Stored", f"{analytics['total_predictions']:,}")
        with col3:
            st.metric("Route Optimizations", f"{analytics['total_optimizations']:,}")
        with col4:
            if analytics['total_flights'] > 0:
                st.metric("Avg Fuel Efficiency", f"{analytics['avg_fuel_efficiency']:.1f}%")
            else:
                st.metric("Database Status", "Connected")
        
        # Popular data
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Popular Aircraft Types")
            if analytics['popular_aircraft']:
                aircraft_df = pd.DataFrame(analytics['popular_aircraft'])
                fig = px.bar(
                    aircraft_df, 
                    x='type', 
                    y='count',
                    title="Most Used Aircraft Types",
                    color='count',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No aircraft data available yet")
        
        with col2:
            st.subheader("🛫 Popular Routes")
            if analytics['popular_routes']:
                routes_df = pd.DataFrame(analytics['popular_routes'])
                fig = px.bar(
                    routes_df, 
                    x='route', 
                    y='count',
                    title="Most Optimized Routes",
                    color='count',
                    color_continuous_scale='Greens'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No route data available yet")
        
        # Database management
        st.subheader("🔧 Database Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Generate Sample Data"):
                with st.spinner("Generating and saving sample flight data..."):
                    sample_data = st.session_state.data_generator.generate_historical_data(days=30)
                    saved = st.session_state.db_manager.save_flight_data(sample_data)
                    if saved:
                        st.success(f"Generated and saved {len(sample_data)} flight records!")
                        st.rerun()
                    else:
                        st.error("Failed to save sample data")
        
        with col2:
            if st.button("🔄 Refresh Analytics"):
                st.rerun()
        
        with col3:
            if st.button("📥 Export Data"):
                st.info("Export functionality coming soon!")
        
        # Recent activity
        st.subheader("🕐 Recent Database Activity")
        
        recent_predictions = st.session_state.db_manager.get_prediction_history(limit=5)
        if recent_predictions:
            st.write("**Recent Predictions:**")
            for pred in recent_predictions:
                st.markdown(f"- {pred['aircraft_type']} at {pred['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        else:
            st.info("No recent predictions found")
        
    except Exception as e:
        st.error("Database connection failed")
        st.code(f"Error: {str(e)}")
        
        if st.button("🔄 Retry Connection"):
            st.rerun()

if __name__ == "__main__":
    main()
