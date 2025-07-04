import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

class FlightPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_columns = [
            'weight', 'altitude', 'speed', 'temperature', 
            'wind_speed', 'wind_direction', 'humidity', 'aircraft_type_encoded'
        ]
        
        # Initialize and train the model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the neural network model with optimal hyperparameters"""
        self.model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.01,
            max_iter=1000,
            random_state=42
        )
        
        # Generate training data and train the model
        self._generate_training_data()
        self._train_model()
    
    def _generate_training_data(self):
        """Generate synthetic training data based on realistic flight physics"""
        np.random.seed(42)
        n_samples = 5000
        
        # Generate realistic flight parameters
        aircraft_types = ['Boeing 737', 'Airbus A320', 'Boeing 777', 'Airbus A350', 'Embraer E190']
        
        data = []
        for _ in range(n_samples):
            aircraft_type = np.random.choice(aircraft_types)
            
            # Aircraft-specific parameter ranges
            if aircraft_type in ['Boeing 737', 'Airbus A320', 'Embraer E190']:
                weight_range = (50, 150)
                altitude_range = (25000, 40000)
                speed_range = (300, 500)
            else:  # Larger aircraft
                weight_range = (150, 300)
                altitude_range = (30000, 45000)
                speed_range = (400, 600)
            
            weight = np.random.uniform(*weight_range)
            altitude = np.random.uniform(*altitude_range)
            speed = np.random.uniform(*speed_range)
            temperature = np.random.uniform(-50, 30)
            wind_speed = np.random.uniform(0, 100)
            wind_direction = np.random.uniform(0, 360)
            humidity = np.random.uniform(0, 100)
            
            # Calculate performance metrics based on physics
            fuel_consumption = self._calculate_fuel_consumption(
                weight, altitude, speed, temperature, wind_speed, wind_direction
            )
            flight_time = self._calculate_flight_time(speed, wind_speed, wind_direction)
            aircraft_range = self._calculate_range(fuel_consumption, weight)
            emissions = fuel_consumption * 3.16  # CO2 emissions factor
            
            # Calculate efficiency metrics
            fuel_efficiency = max(0, min(100, 100 - (fuel_consumption - 500) / 10))
            time_efficiency = max(0, min(100, 100 - (flight_time - 5) * 5))
            range_efficiency = max(0, min(100, aircraft_range / 100))
            emission_efficiency = max(0, min(100, 100 - (emissions - 1500) / 50))
            
            data.append({
                'aircraft_type': aircraft_type,
                'weight': weight,
                'altitude': altitude,
                'speed': speed,
                'temperature': temperature,
                'wind_speed': wind_speed,
                'wind_direction': wind_direction,
                'humidity': humidity,
                'fuel_consumption': fuel_consumption,
                'flight_time': flight_time,
                'range': aircraft_range,
                'emissions': emissions,
                'fuel_efficiency': fuel_efficiency,
                'time_efficiency': time_efficiency,
                'range_efficiency': range_efficiency,
                'emission_efficiency': emission_efficiency
            })
        
        self.training_data = pd.DataFrame(data)
    
    def _calculate_fuel_consumption(self, weight, altitude, speed, temperature, wind_speed, wind_direction):
        """Calculate fuel consumption based on flight parameters"""
        # Base fuel consumption
        base_consumption = 400 + weight * 1.5
        
        # Altitude effect (optimal at 35000 ft)
        altitude_factor = 1 + abs(altitude - 35000) / 100000
        
        # Speed effect
        speed_factor = 1 + (speed - 450) ** 2 / 100000
        
        # Temperature effect
        temp_factor = 1 + abs(temperature + 20) / 100
        
        # Wind effect (headwind increases consumption)
        wind_factor = 1 + max(0, wind_speed * np.cos(np.radians(wind_direction))) / 200
        
        fuel_consumption = base_consumption * altitude_factor * speed_factor * temp_factor * wind_factor
        return max(200, fuel_consumption)
    
    def _calculate_flight_time(self, speed, wind_speed, wind_direction):
        """Calculate flight time considering wind effects"""
        # Base flight time for average route (1000 km)
        base_distance = 1000
        
        # Effective ground speed
        wind_component = wind_speed * np.cos(np.radians(wind_direction))
        ground_speed = speed - wind_component
        
        flight_time = base_distance / max(ground_speed, 200)  # Ensure minimum ground speed
        return max(1, flight_time)
    
    def _calculate_range(self, fuel_consumption, weight):
        """Calculate aircraft range based on fuel consumption and weight"""
        # Base range calculation
        fuel_capacity = weight * 0.3  # Approximate fuel capacity
        range_km = (fuel_capacity / fuel_consumption) * 1000
        return max(500, min(range_km, 15000))
    
    def _train_model(self):
        """Train the neural network model"""
        # Prepare features and targets
        X = self.training_data[['weight', 'altitude', 'speed', 'temperature', 
                               'wind_speed', 'wind_direction', 'humidity', 'aircraft_type']].copy()
        
        # Encode categorical variables
        self.label_encoder.fit(X['aircraft_type'])
        X['aircraft_type_encoded'] = self.label_encoder.transform(X['aircraft_type'])
        X = X.drop('aircraft_type', axis=1)
        
        # Target variables
        y = self.training_data[['fuel_consumption', 'flight_time', 'range', 'emissions',
                               'fuel_efficiency', 'time_efficiency', 'range_efficiency', 
                               'emission_efficiency']]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model trained successfully. MSE: {mse:.2f}, R2: {r2:.2f}")
        self.is_trained = True
    
    def predict_performance(self, input_data):
        """Predict aircraft performance based on input parameters"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
        
        # Prepare input data
        input_df = pd.DataFrame([input_data])
        
        # Encode aircraft type
        aircraft_type_encoded = self.label_encoder.transform([input_data['aircraft_type']])[0]
        
        # Create feature vector
        features = np.array([[
            input_data['weight'],
            input_data['altitude'],
            input_data['speed'],
            input_data['temperature'],
            input_data['wind_speed'],
            input_data['wind_direction'],
            input_data['humidity'],
            aircraft_type_encoded
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        
        # Return structured prediction
        return {
            'fuel_consumption': prediction[0],
            'flight_time': prediction[1],
            'range': prediction[2],
            'emissions': prediction[3],
            'fuel_efficiency': prediction[4],
            'time_efficiency': prediction[5],
            'range_efficiency': prediction[6],
            'emission_efficiency': prediction[7]
        }
    
    def get_model_info(self):
        """Get information about the trained model"""
        if not self.is_trained:
            return "Model is not trained yet"
        
        return {
            'model_type': 'Multi-layer Perceptron (Neural Network)',
            'hidden_layers': self.model.hidden_layer_sizes,
            'activation': self.model.activation,
            'solver': self.model.solver,
            'n_features': len(self.feature_columns),
            'n_outputs': 8,
            'training_samples': len(self.training_data)
        }
