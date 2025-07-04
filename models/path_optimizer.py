import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import math

class PathOptimizer:
    def __init__(self):
        self.airports = {
            'JFK': {'lat': 40.6413, 'lon': -73.7781, 'name': 'John F. Kennedy International Airport'},
            'LAX': {'lat': 33.9425, 'lon': -118.4081, 'name': 'Los Angeles International Airport'},
            'ORD': {'lat': 41.9742, 'lon': -87.9073, 'name': 'Chicago O\'Hare International Airport'},
            'DFW': {'lat': 32.8968, 'lon': -97.0380, 'name': 'Dallas/Fort Worth International Airport'},
            'ATL': {'lat': 33.6407, 'lon': -84.4277, 'name': 'Hartsfield-Jackson Atlanta International Airport'},
            'LHR': {'lat': 51.4700, 'lon': -0.4543, 'name': 'London Heathrow Airport'},
            'CDG': {'lat': 49.0097, 'lon': 2.5479, 'name': 'Charles de Gaulle Airport'},
            'NRT': {'lat': 35.7653, 'lon': 140.3886, 'name': 'Narita International Airport'}
        }
        
        # Initialize optimization models
        self._initialize_optimization_models()
    
    def _initialize_optimization_models(self):
        """Initialize neural network models for path optimization"""
        self.fuel_optimizer = MLPRegressor(
            hidden_layer_sizes=(50, 25),
            activation='relu',
            solver='adam',
            alpha=0.01,
            max_iter=500,
            random_state=42
        )
        
        self.time_optimizer = MLPRegressor(
            hidden_layer_sizes=(50, 25),
            activation='relu',
            solver='adam',
            alpha=0.01,
            max_iter=500,
            random_state=42
        )
        
        self.weather_optimizer = MLPRegressor(
            hidden_layer_sizes=(30, 15),
            activation='relu',
            solver='adam',
            alpha=0.01,
            max_iter=500,
            random_state=42
        )
        
        # Train the optimization models
        self._train_optimization_models()
    
    def _train_optimization_models(self):
        """Train the optimization models with synthetic data"""
        # Generate training data for path optimization
        np.random.seed(42)
        n_samples = 2000
        
        # Generate route parameters
        distances = np.random.uniform(500, 8000, n_samples)
        altitudes = np.random.uniform(30000, 45000, n_samples)
        wind_speeds = np.random.uniform(0, 100, n_samples)
        wind_directions = np.random.uniform(0, 360, n_samples)
        temperatures = np.random.uniform(-60, 30, n_samples)
        weather_severity = np.random.uniform(0, 10, n_samples)
        
        # Create feature matrix
        X = np.column_stack([distances, altitudes, wind_speeds, wind_directions, temperatures, weather_severity])
        
        # Generate target values based on physics
        fuel_efficiency = self._calculate_fuel_efficiency(X)
        time_efficiency = self._calculate_time_efficiency(X)
        weather_scores = self._calculate_weather_scores(X)
        
        # Train models
        self.fuel_optimizer.fit(X, fuel_efficiency)
        self.time_optimizer.fit(X, time_efficiency)
        self.weather_optimizer.fit(X, weather_scores)
    
    def _calculate_fuel_efficiency(self, X):
        """Calculate fuel efficiency based on route parameters"""
        distances, altitudes, wind_speeds, wind_directions, temperatures, weather_severity = X.T
        
        # Base efficiency
        base_efficiency = 80 + np.random.normal(0, 5, len(distances))
        
        # Distance effect
        distance_factor = np.where(distances > 3000, -0.002 * (distances - 3000), 0)
        
        # Altitude effect (optimal around 37000 ft)
        altitude_factor = -0.0001 * (altitudes - 37000) ** 2
        
        # Wind effect
        wind_factor = wind_speeds * np.cos(np.radians(wind_directions)) * 0.1
        
        # Temperature effect
        temp_factor = -0.1 * np.abs(temperatures + 25)
        
        # Weather effect
        weather_factor = -weather_severity * 2
        
        efficiency = base_efficiency + distance_factor + altitude_factor + wind_factor + temp_factor + weather_factor
        return np.clip(efficiency, 50, 100)
    
    def _calculate_time_efficiency(self, X):
        """Calculate time efficiency based on route parameters"""
        distances, altitudes, wind_speeds, wind_directions, temperatures, weather_severity = X.T
        
        # Base time efficiency
        base_efficiency = 85 + np.random.normal(0, 3, len(distances))
        
        # Wind effect on time
        headwind_factor = -np.maximum(0, wind_speeds * np.cos(np.radians(wind_directions))) * 0.2
        
        # Weather delays
        weather_factor = -weather_severity * 1.5
        
        # Altitude effect on cruise efficiency
        altitude_factor = -0.00005 * (altitudes - 38000) ** 2
        
        efficiency = base_efficiency + headwind_factor + weather_factor + altitude_factor
        return np.clip(efficiency, 60, 100)
    
    def _calculate_weather_scores(self, X):
        """Calculate weather impact scores"""
        distances, altitudes, wind_speeds, wind_directions, temperatures, weather_severity = X.T
        
        # Weather score (10 = perfect, 0 = severe)
        base_score = 8 + np.random.normal(0, 1, len(distances))
        
        # Severe weather penalty
        weather_penalty = weather_severity * 0.8
        
        # Wind severity
        wind_penalty = np.maximum(0, wind_speeds - 50) * 0.05
        
        # Extreme temperature penalty
        temp_penalty = np.maximum(0, np.abs(temperatures) - 40) * 0.1
        
        score = base_score - weather_penalty - wind_penalty - temp_penalty
        return np.clip(score, 0, 10)
    
    def optimize_route(self, optimization_params):
        """Optimize flight route based on given parameters"""
        origin = optimization_params['origin']
        destination = optimization_params['destination']
        
        # Calculate base route parameters
        base_route = self._calculate_base_route(origin, destination)
        
        # Generate alternative waypoints
        waypoints = self._generate_waypoints(
            self.airports[origin], 
            self.airports[destination],
            optimization_params
        )
        
        # Optimize each segment
        optimized_waypoints = self._optimize_waypoints(waypoints, optimization_params)
        
        # Calculate route metrics
        route_metrics = self._calculate_route_metrics(optimized_waypoints, base_route, optimization_params)
        
        return {
            'waypoints': optimized_waypoints,
            'total_distance': route_metrics['total_distance'],
            'flight_time': route_metrics['flight_time'],
            'fuel_consumption': route_metrics['fuel_consumption'],
            'cost_savings': route_metrics['cost_savings'],
            'weather_score': route_metrics['weather_score'],
            'efficiency_score': route_metrics['efficiency_score'],
            'direct_distance': base_route['distance'],
            'direct_time': base_route['time'],
            'direct_fuel': base_route['fuel']
        }
    
    def _calculate_base_route(self, origin, destination):
        """Calculate direct route parameters"""
        origin_coords = self.airports[origin]
        dest_coords = self.airports[destination]
        
        # Calculate great circle distance
        distance = self._haversine_distance(
            origin_coords['lat'], origin_coords['lon'],
            dest_coords['lat'], dest_coords['lon']
        )
        
        # Estimate direct flight time and fuel
        average_speed = 850  # km/h
        flight_time = distance / average_speed
        fuel_consumption = distance * 0.8 + 500  # kg
        
        return {
            'distance': distance,
            'time': flight_time,
            'fuel': fuel_consumption
        }
    
    def _generate_waypoints(self, origin, destination, params):
        """Generate optimized waypoints for the route"""
        # Calculate number of waypoints based on distance
        total_distance = self._haversine_distance(
            origin['lat'], origin['lon'],
            destination['lat'], destination['lon']
        )
        
        num_waypoints = min(8, max(3, int(total_distance / 1000)))
        
        waypoints = []
        
        # Add origin
        waypoints.append({
            'lat': origin['lat'],
            'lon': origin['lon'],
            'altitude': 0,
            'distance': 0
        })
        
        # Generate intermediate waypoints
        for i in range(1, num_waypoints - 1):
            # Linear interpolation with optimization adjustments
            ratio = i / (num_waypoints - 1)
            
            # Base position
            lat = origin['lat'] + (destination['lat'] - origin['lat']) * ratio
            lon = origin['lon'] + (destination['lon'] - origin['lon']) * ratio
            
            # Add optimization-based adjustments
            lat_offset = np.random.uniform(-2, 2) * (1 - params['fuel_weight'])
            lon_offset = np.random.uniform(-2, 2) * (1 - params['time_weight'])
            
            lat += lat_offset
            lon += lon_offset
            
            # Calculate altitude based on distance and optimization
            altitude = min(params['max_altitude'], 35000 + (total_distance * ratio) / 100)
            
            # Calculate cumulative distance
            if waypoints:
                prev_waypoint = waypoints[-1]
                segment_distance = self._haversine_distance(
                    prev_waypoint['lat'], prev_waypoint['lon'], lat, lon
                )
                cumulative_distance = prev_waypoint['distance'] + segment_distance
            else:
                cumulative_distance = 0
            
            waypoints.append({
                'lat': lat,
                'lon': lon,
                'altitude': altitude,
                'distance': cumulative_distance
            })
        
        # Add destination
        if waypoints:
            prev_waypoint = waypoints[-1]
            final_distance = self._haversine_distance(
                prev_waypoint['lat'], prev_waypoint['lon'],
                destination['lat'], destination['lon']
            )
            total_distance = prev_waypoint['distance'] + final_distance
        else:
            total_distance = 0
        
        waypoints.append({
            'lat': destination['lat'],
            'lon': destination['lon'],
            'altitude': 0,
            'distance': total_distance
        })
        
        return waypoints
    
    def _optimize_waypoints(self, waypoints, params):
        """Optimize waypoints using neural network models"""
        optimized_waypoints = []
        
        for i, waypoint in enumerate(waypoints):
            # Generate environmental conditions for this waypoint
            wind_speed = np.random.uniform(10, 80)
            wind_direction = np.random.uniform(0, 360)
            temperature = np.random.uniform(-50, 20)
            weather_severity = np.random.uniform(0, 5)
            
            # Prepare features for optimization models
            features = np.array([[
                waypoint['distance'],
                waypoint['altitude'],
                wind_speed,
                wind_direction,
                temperature,
                weather_severity
            ]])
            
            # Get optimization predictions
            fuel_efficiency = self.fuel_optimizer.predict(features)[0]
            time_efficiency = self.time_optimizer.predict(features)[0]
            weather_score = self.weather_optimizer.predict(features)[0]
            
            # Calculate optimization adjustments
            optimization_score = (
                fuel_efficiency * params['fuel_weight'] +
                time_efficiency * params['time_weight'] +
                weather_score * params['weather_weight']
            )
            
            # Add environmental data to waypoint
            optimized_waypoint = waypoint.copy()
            optimized_waypoint.update({
                'wind_speed': wind_speed,
                'wind_direction': wind_direction,
                'temperature': temperature,
                'weather_severity': weather_severity,
                'fuel_efficiency': fuel_efficiency,
                'time_efficiency': time_efficiency,
                'weather_score': weather_score,
                'optimization_score': optimization_score
            })
            
            optimized_waypoints.append(optimized_waypoint)
        
        return optimized_waypoints
    
    def _calculate_route_metrics(self, waypoints, base_route, params):
        """Calculate comprehensive route metrics"""
        # Calculate total distance
        total_distance = waypoints[-1]['distance']
        
        # Calculate average efficiencies
        avg_fuel_efficiency = np.mean([wp['fuel_efficiency'] for wp in waypoints[1:-1]])
        avg_time_efficiency = np.mean([wp['time_efficiency'] for wp in waypoints[1:-1]])
        avg_weather_score = np.mean([wp['weather_score'] for wp in waypoints[1:-1]])
        
        # Calculate flight time
        average_speed = 850 * (avg_time_efficiency / 100)
        flight_time = total_distance / average_speed
        
        # Calculate fuel consumption
        base_fuel = total_distance * 0.8 + 500
        fuel_efficiency_factor = avg_fuel_efficiency / 100
        fuel_consumption = base_fuel / fuel_efficiency_factor
        
        # Calculate cost savings
        fuel_cost_per_kg = 0.8  # USD per kg
        direct_fuel_cost = base_route['fuel'] * fuel_cost_per_kg
        optimized_fuel_cost = fuel_consumption * fuel_cost_per_kg
        cost_savings = direct_fuel_cost - optimized_fuel_cost
        
        # Calculate overall efficiency score
        efficiency_score = (
            avg_fuel_efficiency * params['fuel_weight'] +
            avg_time_efficiency * params['time_weight'] +
            (avg_weather_score * 10) * params['weather_weight']
        )
        
        return {
            'total_distance': total_distance,
            'flight_time': flight_time,
            'fuel_consumption': fuel_consumption,
            'cost_savings': cost_savings,
            'weather_score': avg_weather_score,
            'efficiency_score': efficiency_score
        }
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points on Earth"""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
