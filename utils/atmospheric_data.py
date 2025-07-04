import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class AtmosphericDataProvider:
    def __init__(self):
        """Initialize atmospheric data provider with realistic patterns"""
        self.weather_patterns = {
            'clear': {'temp_range': (-20, 25), 'wind_range': (5, 30), 'humidity_range': (30, 70)},
            'cloudy': {'temp_range': (-15, 20), 'wind_range': (10, 45), 'humidity_range': (50, 85)},
            'stormy': {'temp_range': (-10, 15), 'wind_range': (25, 80), 'humidity_range': (70, 95)},
            'windy': {'temp_range': (-25, 20), 'wind_range': (40, 100), 'humidity_range': (25, 75)}
        }
        
        self.altitude_layers = {
            'surface': 0,
            'low': 10000,
            'mid': 25000,
            'high': 35000,
            'very_high': 45000
        }
        
        np.random.seed(int(datetime.now().timestamp()) % 1000)
    
    def get_current_conditions(self, location='global'):
        """Get current atmospheric conditions"""
        # Select weather pattern based on time and randomness
        current_hour = datetime.now().hour
        
        # More stable weather during certain hours
        if 6 <= current_hour <= 18:
            weather_type = np.random.choice(['clear', 'cloudy', 'windy'], p=[0.5, 0.3, 0.2])
        else:
            weather_type = np.random.choice(['clear', 'cloudy', 'stormy', 'windy'], p=[0.3, 0.3, 0.2, 0.2])
        
        pattern = self.weather_patterns[weather_type]
        
        # Generate atmospheric conditions
        conditions = {
            'timestamp': datetime.now(),
            'location': location,
            'weather_type': weather_type,
            'temperature': np.random.uniform(*pattern['temp_range']),
            'wind_speed': np.random.uniform(*pattern['wind_range']),
            'wind_direction': np.random.uniform(0, 360),
            'humidity': np.random.uniform(*pattern['humidity_range']),
            'pressure': np.random.uniform(950, 1050),
            'visibility': self._calculate_visibility(weather_type),
            'turbulence_level': self._calculate_turbulence(weather_type),
            'precipitation': self._calculate_precipitation(weather_type)
        }
        
        return conditions
    
    def get_atmospheric_profile(self, location='global', max_altitude=45000):
        """Get atmospheric conditions at different altitudes"""
        profile = []
        
        for altitude in range(0, max_altitude + 1, 5000):
            conditions = self._get_conditions_at_altitude(altitude, location)
            profile.append(conditions)
        
        return pd.DataFrame(profile)
    
    def _get_conditions_at_altitude(self, altitude, location):
        """Get atmospheric conditions at specific altitude"""
        # Temperature decreases with altitude (lapse rate)
        surface_temp = np.random.uniform(-10, 30)
        
        if altitude <= 11000:  # Troposphere
            temp_lapse_rate = 6.5  # °C per 1000m
            temperature = surface_temp - (altitude * 3.28084 / 1000) * temp_lapse_rate
        elif altitude <= 20000:  # Lower stratosphere
            temperature = -56.5  # Constant temperature
        else:  # Upper stratosphere
            temperature = -56.5 + (altitude - 20000) * 3.28084 / 1000 * 1.0
        
        # Wind speed generally increases with altitude
        wind_speed = 20 + (altitude / 1000) * 2 + np.random.uniform(-10, 15)
        wind_speed = max(0, wind_speed)
        
        # Pressure decreases exponentially with altitude
        pressure = 1013.25 * np.exp(-altitude * 3.28084 / 29000)
        
        # Humidity decreases with altitude
        humidity = max(5, 80 - (altitude / 1000) * 3 + np.random.uniform(-10, 10))
        
        return {
            'altitude': altitude,
            'temperature': temperature,
            'wind_speed': wind_speed,
            'wind_direction': np.random.uniform(0, 360),
            'pressure': pressure,
            'humidity': humidity,
            'air_density': self._calculate_air_density(temperature, pressure),
            'location': location
        }
    
    def _calculate_visibility(self, weather_type):
        """Calculate visibility based on weather type"""
        visibility_ranges = {
            'clear': (15, 25),
            'cloudy': (8, 15),
            'stormy': (1, 8),
            'windy': (10, 20)
        }
        
        return np.random.uniform(*visibility_ranges[weather_type])
    
    def _calculate_turbulence(self, weather_type):
        """Calculate turbulence level based on weather type"""
        turbulence_levels = {
            'clear': np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1]),
            'cloudy': np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2]),
            'stormy': np.random.choice([2, 3, 4], p=[0.2, 0.5, 0.3]),
            'windy': np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
        }
        
        return turbulence_levels[weather_type]
    
    def _calculate_precipitation(self, weather_type):
        """Calculate precipitation based on weather type"""
        precipitation_prob = {
            'clear': 0.05,
            'cloudy': 0.3,
            'stormy': 0.8,
            'windy': 0.2
        }
        
        if np.random.random() < precipitation_prob[weather_type]:
            return np.random.uniform(0.1, 10.0)  # mm/hour
        else:
            return 0.0
    
    def _calculate_air_density(self, temperature, pressure):
        """Calculate air density based on temperature and pressure"""
        # Using ideal gas law: ρ = P / (R * T)
        R = 287.058  # Specific gas constant for dry air (J/(kg·K))
        T_kelvin = temperature + 273.15
        pressure_pa = pressure * 100  # Convert hPa to Pa
        
        density = pressure_pa / (R * T_kelvin)
        return density
    
    def get_weather_forecast(self, hours=24, location='global'):
        """Get weather forecast for specified hours"""
        forecast = []
        current_time = datetime.now()
        
        for hour in range(hours):
            forecast_time = current_time + timedelta(hours=hour)
            
            # Add some trend to make forecast realistic
            if hour == 0:
                base_conditions = self.get_current_conditions(location)
            else:
                # Evolve conditions from previous hour
                prev_conditions = forecast[-1]
                base_conditions = self._evolve_conditions(prev_conditions)
            
            base_conditions['forecast_time'] = forecast_time
            base_conditions['forecast_hour'] = hour
            
            forecast.append(base_conditions)
        
        return pd.DataFrame(forecast)
    
    def _evolve_conditions(self, prev_conditions):
        """Evolve atmospheric conditions from previous state"""
        # Small random changes to create realistic weather evolution
        evolved = prev_conditions.copy()
        
        # Temperature evolution
        temp_change = np.random.uniform(-2, 2)
        evolved['temperature'] += temp_change
        
        # Wind evolution
        wind_change = np.random.uniform(-5, 5)
        evolved['wind_speed'] = max(0, evolved['wind_speed'] + wind_change)
        
        # Wind direction evolution
        direction_change = np.random.uniform(-30, 30)
        evolved['wind_direction'] = (evolved['wind_direction'] + direction_change) % 360
        
        # Humidity evolution
        humidity_change = np.random.uniform(-5, 5)
        evolved['humidity'] = np.clip(evolved['humidity'] + humidity_change, 0, 100)
        
        # Pressure evolution
        pressure_change = np.random.uniform(-2, 2)
        evolved['pressure'] += pressure_change
        
        # Update derived values
        evolved['visibility'] = self._calculate_visibility(evolved['weather_type'])
        evolved['turbulence_level'] = self._calculate_turbulence(evolved['weather_type'])
        evolved['precipitation'] = self._calculate_precipitation(evolved['weather_type'])
        
        return evolved
    
    def get_wind_aloft_data(self, altitudes=[3000, 6000, 9000, 12000, 18000, 24000, 30000, 34000, 39000]):
        """Get wind aloft data for specified altitudes"""
        wind_aloft = []
        
        for altitude in altitudes:
            # Generate realistic wind patterns
            base_wind_speed = 20 + (altitude / 1000) * 1.5
            wind_speed = base_wind_speed + np.random.uniform(-15, 20)
            wind_speed = max(0, wind_speed)
            
            # Wind direction tends to be more westerly at higher altitudes
            if altitude > 20000:
                wind_direction = np.random.uniform(240, 300)  # Generally westerly
            else:
                wind_direction = np.random.uniform(0, 360)
            
            wind_aloft.append({
                'altitude': altitude,
                'wind_speed': wind_speed,
                'wind_direction': wind_direction,
                'temperature': self._get_conditions_at_altitude(altitude, 'global')['temperature']
            })
        
        return pd.DataFrame(wind_aloft)
