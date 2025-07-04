import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class FlightDataGenerator:
    def __init__(self):
        self.aircraft_types = ['Boeing 737', 'Airbus A320', 'Boeing 777', 'Airbus A350', 'Embraer E190']
        self.airports = ['JFK', 'LAX', 'ORD', 'DFW', 'ATL', 'LHR', 'CDG', 'NRT', 'SYD', 'DXB']
        self.routes = [
            ('JFK', 'LAX'), ('LAX', 'NRT'), ('LHR', 'JFK'), ('CDG', 'DXB'),
            ('ATL', 'LHR'), ('ORD', 'LAX'), ('DFW', 'CDG'), ('SYD', 'LAX'),
            ('DXB', 'JFK'), ('NRT', 'SYD'), ('LAX', 'JFK'), ('LHR', 'CDG'),
            ('JFK', 'LHR'), ('LAX', 'SYD'), ('ORD', 'NRT')
        ]
        
        np.random.seed(42)
        random.seed(42)
    
    def generate_historical_data(self, days=30):
        """Generate historical flight data for the specified number of days"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate flights for each day
        flights_per_day = np.random.poisson(50, days)  # Average 50 flights per day
        
        historical_data = []
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            num_flights = flights_per_day[day]
            
            for flight in range(num_flights):
                flight_data = self._generate_single_flight_data(current_date)
                historical_data.append(flight_data)
        
        return pd.DataFrame(historical_data)
    
    def _generate_single_flight_data(self, date):
        """Generate data for a single flight"""
        # Select aircraft and route
        aircraft_type = np.random.choice(self.aircraft_types)
        origin, destination = random.choice(self.routes)
        
        # Generate flight parameters based on aircraft type
        if aircraft_type in ['Boeing 737', 'Airbus A320', 'Embraer E190']:
            weight = np.random.uniform(60, 140)
            altitude = np.random.uniform(28000, 38000)
            speed = np.random.uniform(350, 480)
            distance = np.random.uniform(800, 3500)
        else:  # Larger aircraft
            weight = np.random.uniform(180, 280)
            altitude = np.random.uniform(35000, 43000)
            speed = np.random.uniform(450, 580)
            distance = np.random.uniform(2000, 8000)
        
        # Environmental conditions
        temperature = np.random.uniform(-55, 25)
        wind_speed = np.random.uniform(5, 95)
        wind_direction = np.random.uniform(0, 360)
        humidity = np.random.uniform(20, 90)
        
        # Calculate performance metrics
        fuel_consumption = self._calculate_fuel_consumption(
            aircraft_type, weight, altitude, speed, distance, temperature, wind_speed, wind_direction
        )
        
        flight_time = self._calculate_flight_time(distance, speed, wind_speed, wind_direction)
        
        emissions = fuel_consumption * 3.16  # CO2 emissions factor
        
        # Calculate efficiency metrics
        fuel_efficiency = self._calculate_fuel_efficiency(fuel_consumption, distance, weight)
        time_efficiency = self._calculate_time_efficiency(flight_time, distance, speed)
        
        # Add some seasonal and time-based variations
        seasonal_factor = self._get_seasonal_factor(date)
        time_factor = self._get_time_factor(date)
        
        # Apply factors to efficiency metrics
        fuel_efficiency *= seasonal_factor * time_factor
        fuel_efficiency = np.clip(fuel_efficiency, 60, 100)
        
        return {
            'date': date,
            'aircraft_type': aircraft_type,
            'origin': origin,
            'destination': destination,
            'weight': weight,
            'altitude': altitude,
            'speed': speed,
            'distance': distance,
            'temperature': temperature,
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'humidity': humidity,
            'fuel_consumption': fuel_consumption,
            'flight_time': flight_time,
            'emissions': emissions,
            'fuel_efficiency': fuel_efficiency,
            'time_efficiency': time_efficiency
        }
    
    def _calculate_fuel_consumption(self, aircraft_type, weight, altitude, speed, distance, temperature, wind_speed, wind_direction):
        """Calculate fuel consumption based on flight parameters"""
        # Base consumption rates by aircraft type
        base_rates = {
            'Boeing 737': 2.5,
            'Airbus A320': 2.3,
            'Boeing 777': 4.2,
            'Airbus A350': 3.8,
            'Embraer E190': 2.1
        }
        
        base_rate = base_rates[aircraft_type]
        
        # Base fuel consumption
        base_fuel = base_rate * distance + weight * 0.8
        
        # Altitude efficiency (optimal around 35000-38000 ft)
        if aircraft_type in ['Boeing 737', 'Airbus A320', 'Embraer E190']:
            optimal_altitude = 35000
        else:
            optimal_altitude = 38000
        
        altitude_factor = 1 + abs(altitude - optimal_altitude) / 50000
        
        # Speed factor
        if aircraft_type in ['Boeing 737', 'Airbus A320', 'Embraer E190']:
            optimal_speed = 420
        else:
            optimal_speed = 500
        
        speed_factor = 1 + (speed - optimal_speed) ** 2 / 100000
        
        # Temperature factor
        temp_factor = 1 + abs(temperature + 25) / 80
        
        # Wind factor
        headwind_component = max(0, wind_speed * np.cos(np.radians(wind_direction)))
        wind_factor = 1 + headwind_component / 200
        
        total_fuel = base_fuel * altitude_factor * speed_factor * temp_factor * wind_factor
        
        return max(100, total_fuel)
    
    def _calculate_flight_time(self, distance, speed, wind_speed, wind_direction):
        """Calculate flight time considering wind effects"""
        # Wind component (positive for headwind, negative for tailwind)
        wind_component = wind_speed * np.cos(np.radians(wind_direction))
        
        # Effective ground speed
        ground_speed = speed - wind_component
        ground_speed = max(ground_speed, 200)  # Minimum ground speed
        
        # Flight time in hours
        flight_time = distance / ground_speed
        
        return max(0.5, flight_time)
    
    def _calculate_fuel_efficiency(self, fuel_consumption, distance, weight):
        """Calculate fuel efficiency percentage"""
        # Theoretical optimal fuel consumption
        optimal_fuel = distance * 1.8 + weight * 0.5
        
        # Efficiency as percentage
        efficiency = (optimal_fuel / fuel_consumption) * 100
        
        return np.clip(efficiency, 50, 100)
    
    def _calculate_time_efficiency(self, flight_time, distance, speed):
        """Calculate time efficiency percentage"""
        # Theoretical optimal time
        optimal_time = distance / speed
        
        # Efficiency as percentage
        efficiency = (optimal_time / flight_time) * 100
        
        return np.clip(efficiency, 60, 100)
    
    def _get_seasonal_factor(self, date):
        """Get seasonal factor affecting flight efficiency"""
        # Winter months typically have better fuel efficiency due to cold air
        month = date.month
        if month in [12, 1, 2]:  # Winter
            return np.random.uniform(1.02, 1.08)
        elif month in [6, 7, 8]:  # Summer
            return np.random.uniform(0.95, 1.02)
        else:  # Spring/Fall
            return np.random.uniform(0.98, 1.05)
    
    def _get_time_factor(self, date):
        """Get time-based factor affecting flight efficiency"""
        # Rush hours and peak travel times
        hour = date.hour
        if 6 <= hour <= 9 or 17 <= hour <= 20:  # Rush hours
            return np.random.uniform(0.92, 0.98)
        elif 23 <= hour or hour <= 5:  # Night flights
            return np.random.uniform(1.02, 1.08)
        else:  # Off-peak hours
            return np.random.uniform(0.98, 1.05)
    
    def generate_flight_activity_data(self):
        """Generate flight activity heatmap data"""
        hours = list(range(24))
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        activity_data = []
        
        for day in days:
            for hour in hours:
                # Generate realistic flight activity patterns
                if day in ['Saturday', 'Sunday']:
                    # Weekend pattern
                    if 6 <= hour <= 22:
                        base_activity = np.random.poisson(25)
                    else:
                        base_activity = np.random.poisson(8)
                else:
                    # Weekday pattern
                    if 6 <= hour <= 9 or 17 <= hour <= 21:
                        base_activity = np.random.poisson(45)
                    elif 10 <= hour <= 16:
                        base_activity = np.random.poisson(35)
                    else:
                        base_activity = np.random.poisson(15)
                
                activity_data.append({
                    'hour': hour,
                    'day_of_week': day,
                    'flight_count': base_activity
                })
        
        return pd.DataFrame(activity_data)
