import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

def create_gauge_chart(value, title, min_val, max_val, color_ranges=None):
    """Create a gauge chart for displaying metrics"""
    if color_ranges is None:
        color_ranges = ['green', 'yellow', 'orange', 'red']
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        delta={'reference': (min_val + max_val) / 2},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [min_val, min_val + (max_val - min_val) * 0.25], 'color': color_ranges[0]},
                {'range': [min_val + (max_val - min_val) * 0.25, min_val + (max_val - min_val) * 0.5], 'color': color_ranges[1]},
                {'range': [min_val + (max_val - min_val) * 0.5, min_val + (max_val - min_val) * 0.75], 'color': color_ranges[2]},
                {'range': [min_val + (max_val - min_val) * 0.75, max_val], 'color': color_ranges[3]}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        font={'color': "darkblue", 'family': "Arial"},
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_3d_flight_path(waypoints):
    """Create 3D visualization of flight path"""
    # Extract coordinates
    lats = [wp['lat'] for wp in waypoints]
    lons = [wp['lon'] for wp in waypoints]
    alts = [wp['altitude'] for wp in waypoints]
    distances = [wp['distance'] for wp in waypoints]
    
    # Create 3D line plot
    fig = go.Figure()
    
    # Add flight path
    fig.add_trace(go.Scatter3d(
        x=lons,
        y=lats,
        z=alts,
        mode='lines+markers',
        line=dict(
            color='blue',
            width=6
        ),
        marker=dict(
            size=8,
            color=alts,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Altitude (ft)")
        ),
        name='Flight Path',
        hovertemplate='<b>Lat:</b> %{y:.2f}<br>' +
                      '<b>Lon:</b> %{x:.2f}<br>' +
                      '<b>Altitude:</b> %{z:.0f} ft<br>' +
                      '<extra></extra>'
    ))
    
    # Add waypoints with labels
    for i, wp in enumerate(waypoints):
        if i == 0:
            label = "Origin"
            color = "green"
        elif i == len(waypoints) - 1:
            label = "Destination"
            color = "red"
        else:
            label = f"Waypoint {i}"
            color = "orange"
        
        fig.add_trace(go.Scatter3d(
            x=[wp['lon']],
            y=[wp['lat']],
            z=[wp['altitude']],
            mode='markers+text',
            marker=dict(
                size=12,
                color=color,
                symbol='diamond'
            ),
            text=[label],
            textposition="top center",
            showlegend=False,
            hovertemplate=f'<b>{label}</b><br>' +
                          f'<b>Lat:</b> {wp["lat"]:.2f}<br>' +
                          f'<b>Lon:</b> {wp["lon"]:.2f}<br>' +
                          f'<b>Altitude:</b> {wp["altitude"]:.0f} ft<br>' +
                          '<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title='3D Flight Path Visualization',
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Altitude (ft)',
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=0.8)
            )
        ),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig

def create_performance_radar_chart(metrics):
    """Create radar chart for performance metrics"""
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Performance',
        line_color='rgb(0, 102, 204)',
        fillcolor='rgba(0, 102, 204, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                ticktext=['20%', '40%', '60%', '80%', '100%']
            )
        ),
        showlegend=True,
        title="Performance Metrics Radar Chart",
        height=400
    )
    
    return fig

def create_flight_activity_heatmap(activity_data):
    """Create heatmap for flight activity"""
    # Pivot data for heatmap
    pivot_data = activity_data.pivot(index='day_of_week', columns='hour', values='flight_count')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='Blues',
        hoverongaps=False,
        hovertemplate='<b>Day:</b> %{y}<br>' +
                      '<b>Hour:</b> %{x}<br>' +
                      '<b>Flights:</b> %{z}<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title='Flight Activity Heatmap',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        height=400
    )
    
    return fig

def create_fuel_efficiency_timeline(data):
    """Create timeline chart for fuel efficiency"""
    fig = go.Figure()
    
    # Add main efficiency line
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['fuel_efficiency'],
        mode='lines+markers',
        name='Fuel Efficiency',
        line=dict(color='blue', width=2),
        marker=dict(size=6),
        hovertemplate='<b>Date:</b> %{x}<br>' +
                      '<b>Efficiency:</b> %{y:.1f}%<br>' +
                      '<extra></extra>'
    ))
    
    # Add trend line
    z = np.polyfit(range(len(data)), data['fuel_efficiency'], 1)
    p = np.poly1d(z)
    trend_line = p(range(len(data)))
    
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=trend_line,
        mode='lines',
        name='Trend',
        line=dict(color='red', width=2, dash='dash'),
        hovertemplate='<b>Trend:</b> %{y:.1f}%<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title='Fuel Efficiency Over Time',
        xaxis_title='Date',
        yaxis_title='Fuel Efficiency (%)',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_aircraft_comparison_chart(aircraft_data):
    """Create comparison chart for different aircraft types"""
    fig = go.Figure()
    
    # Add bars for each metric
    metrics = ['fuel_efficiency', 'time_efficiency', 'range_efficiency']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            x=aircraft_data['aircraft_type'],
            y=aircraft_data[metric],
            name=metric.replace('_', ' ').title(),
            marker_color=colors[i],
            hovertemplate='<b>Aircraft:</b> %{x}<br>' +
                          f'<b>{metric.replace("_", " ").title()}:</b> %{{y:.1f}}%<br>' +
                          '<extra></extra>'
        ))
    
    fig.update_layout(
        title='Aircraft Performance Comparison',
        xaxis_title='Aircraft Type',
        yaxis_title='Efficiency (%)',
        barmode='group',
        height=400
    )
    
    return fig

def create_weather_impact_chart(weather_data):
    """Create chart showing weather impact on flight performance"""
    fig = go.Figure()
    
    # Create scatter plot
    fig.add_trace(go.Scatter(
        x=weather_data['wind_speed'],
        y=weather_data['fuel_efficiency'],
        mode='markers',
        marker=dict(
            size=10,
            color=weather_data['temperature'],
            colorscale='RdYlBu',
            showscale=True,
            colorbar=dict(title="Temperature (°C)")
        ),
        text=weather_data['weather_type'],
        hovertemplate='<b>Wind Speed:</b> %{x:.1f} km/h<br>' +
                      '<b>Fuel Efficiency:</b> %{y:.1f}%<br>' +
                      '<b>Temperature:</b> %{marker.color:.1f}°C<br>' +
                      '<b>Weather:</b> %{text}<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title='Weather Impact on Fuel Efficiency',
        xaxis_title='Wind Speed (km/h)',
        yaxis_title='Fuel Efficiency (%)',
        height=400
    )
    
    return fig

def create_altitude_profile_chart(waypoints):
    """Create altitude profile chart for flight path"""
    distances = [wp['distance'] for wp in waypoints]
    altitudes = [wp['altitude'] for wp in waypoints]
    
    fig = go.Figure()
    
    # Add altitude profile
    fig.add_trace(go.Scatter(
        x=distances,
        y=altitudes,
        mode='lines+markers',
        name='Altitude Profile',
        line=dict(color='blue', width=3),
        marker=dict(size=8),
        fill='tonexty',
        fillcolor='rgba(0, 100, 255, 0.1)',
        hovertemplate='<b>Distance:</b> %{x:.0f} km<br>' +
                      '<b>Altitude:</b> %{y:.0f} ft<br>' +
                      '<extra></extra>'
    ))
    
    # Add ground reference
    fig.add_trace(go.Scatter(
        x=distances,
        y=[0] * len(distances),
        mode='lines',
        name='Ground Level',
        line=dict(color='brown', width=2),
        showlegend=False
    ))
    
    fig.update_layout(
        title='Flight Altitude Profile',
        xaxis_title='Distance (km)',
        yaxis_title='Altitude (ft)',
        height=400,
        yaxis=dict(range=[0, max(altitudes) * 1.1])
    )
    
    return fig
