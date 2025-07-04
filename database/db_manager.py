import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from datetime import datetime
import json

Base = declarative_base()

class FlightData(Base):
    __tablename__ = 'flight_data'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    aircraft_type = Column(String(50), nullable=False)
    origin = Column(String(10), nullable=False)
    destination = Column(String(10), nullable=False)
    weight = Column(Float, nullable=False)
    altitude = Column(Float, nullable=False)
    speed = Column(Float, nullable=False)
    distance = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False)
    wind_speed = Column(Float, nullable=False)
    wind_direction = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)
    fuel_consumption = Column(Float, nullable=False)
    flight_time = Column(Float, nullable=False)
    emissions = Column(Float, nullable=False)
    fuel_efficiency = Column(Float, nullable=False)
    time_efficiency = Column(Float, nullable=False)
    created_at = Column(DateTime, default=func.now())

class PredictionHistory(Base):
    __tablename__ = 'prediction_history'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=func.now())
    aircraft_type = Column(String(50), nullable=False)
    weight = Column(Float, nullable=False)
    altitude = Column(Float, nullable=False)
    speed = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False)
    wind_speed = Column(Float, nullable=False)
    wind_direction = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)
    prediction_results = Column(Text, nullable=False)  # JSON string
    session_id = Column(String(100))

class RouteOptimization(Base):
    __tablename__ = 'route_optimizations'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=func.now())
    origin = Column(String(10), nullable=False)
    destination = Column(String(10), nullable=False)
    fuel_weight = Column(Float, nullable=False)
    time_weight = Column(Float, nullable=False)
    weather_weight = Column(Float, nullable=False)
    max_altitude = Column(Float, nullable=False)
    min_fuel_reserve = Column(Float, nullable=False)
    optimization_results = Column(Text, nullable=False)  # JSON string
    session_id = Column(String(100))

class UserConfigurations(Base):
    __tablename__ = 'user_configurations'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    config_type = Column(String(50), nullable=False)  # 'aircraft' or 'route'
    configuration = Column(Text, nullable=False)  # JSON string
    created_at = Column(DateTime, default=func.now())
    is_favorite = Column(Boolean, default=False)
    session_id = Column(String(100))

class DatabaseManager:
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def save_flight_data(self, flight_data_df):
        """Save flight data to database"""
        try:
            with self.get_session() as session:
                for _, row in flight_data_df.iterrows():
                    flight_record = FlightData(
                        date=row['date'],
                        aircraft_type=row['aircraft_type'],
                        origin=row['origin'],
                        destination=row['destination'],
                        weight=row['weight'],
                        altitude=row['altitude'],
                        speed=row['speed'],
                        distance=row['distance'],
                        temperature=row['temperature'],
                        wind_speed=row['wind_speed'],
                        wind_direction=row['wind_direction'],
                        humidity=row['humidity'],
                        fuel_consumption=row['fuel_consumption'],
                        flight_time=row['flight_time'],
                        emissions=row['emissions'],
                        fuel_efficiency=row['fuel_efficiency'],
                        time_efficiency=row['time_efficiency']
                    )
                    session.add(flight_record)
                session.commit()
                return True
        except Exception as e:
            print(f"Error saving flight data: {e}")
            return False
    
    def get_flight_data(self, days=30, limit=1000):
        """Retrieve flight data from database"""
        try:
            with self.get_session() as session:
                cutoff_date = datetime.now() - pd.Timedelta(days=days)
                records = session.query(FlightData).filter(
                    FlightData.date >= cutoff_date
                ).order_by(FlightData.date.desc()).limit(limit).all()
                
                data = []
                for record in records:
                    data.append({
                        'date': record.date,
                        'aircraft_type': record.aircraft_type,
                        'origin': record.origin,
                        'destination': record.destination,
                        'weight': record.weight,
                        'altitude': record.altitude,
                        'speed': record.speed,
                        'distance': record.distance,
                        'temperature': record.temperature,
                        'wind_speed': record.wind_speed,
                        'wind_direction': record.wind_direction,
                        'humidity': record.humidity,
                        'fuel_consumption': record.fuel_consumption,
                        'flight_time': record.flight_time,
                        'emissions': record.emissions,
                        'fuel_efficiency': record.fuel_efficiency,
                        'time_efficiency': record.time_efficiency
                    })
                
                return pd.DataFrame(data)
        except Exception as e:
            print(f"Error retrieving flight data: {e}")
            return pd.DataFrame()
    
    def save_prediction(self, prediction_data, session_id=None):
        """Save prediction history"""
        try:
            with self.get_session() as session:
                prediction_record = PredictionHistory(
                    aircraft_type=prediction_data['aircraft_type'],
                    weight=prediction_data['weight'],
                    altitude=prediction_data['altitude'],
                    speed=prediction_data['speed'],
                    temperature=prediction_data['temperature'],
                    wind_speed=prediction_data['wind_speed'],
                    wind_direction=prediction_data['wind_direction'],
                    humidity=prediction_data['humidity'],
                    prediction_results=json.dumps(prediction_data['results']),
                    session_id=session_id
                )
                session.add(prediction_record)
                session.commit()
                return True
        except Exception as e:
            print(f"Error saving prediction: {e}")
            return False
    
    def get_prediction_history(self, limit=50, session_id=None):
        """Retrieve prediction history"""
        try:
            with self.get_session() as session:
                query = session.query(PredictionHistory)
                if session_id:
                    query = query.filter(PredictionHistory.session_id == session_id)
                
                records = query.order_by(PredictionHistory.timestamp.desc()).limit(limit).all()
                
                data = []
                for record in records:
                    data.append({
                        'timestamp': record.timestamp,
                        'aircraft_type': record.aircraft_type,
                        'prediction': json.loads(record.prediction_results)
                    })
                
                return data
        except Exception as e:
            print(f"Error retrieving prediction history: {e}")
            return []
    
    def save_route_optimization(self, optimization_data, session_id=None):
        """Save route optimization results"""
        try:
            with self.get_session() as session:
                route_record = RouteOptimization(
                    origin=optimization_data['origin'],
                    destination=optimization_data['destination'],
                    fuel_weight=optimization_data['fuel_weight'],
                    time_weight=optimization_data['time_weight'],
                    weather_weight=optimization_data['weather_weight'],
                    max_altitude=optimization_data['max_altitude'],
                    min_fuel_reserve=optimization_data['min_fuel_reserve'],
                    optimization_results=json.dumps(optimization_data['results']),
                    session_id=session_id
                )
                session.add(route_record)
                session.commit()
                return True
        except Exception as e:
            print(f"Error saving route optimization: {e}")
            return False
    
    def save_user_configuration(self, name, config_type, configuration, session_id=None, is_favorite=False):
        """Save user configuration"""
        try:
            with self.get_session() as session:
                config_record = UserConfigurations(
                    name=name,
                    config_type=config_type,
                    configuration=json.dumps(configuration),
                    session_id=session_id,
                    is_favorite=is_favorite
                )
                session.add(config_record)
                session.commit()
                return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def get_user_configurations(self, config_type=None, session_id=None, favorites_only=False):
        """Retrieve user configurations"""
        try:
            with self.get_session() as session:
                query = session.query(UserConfigurations)
                
                if config_type:
                    query = query.filter(UserConfigurations.config_type == config_type)
                if session_id:
                    query = query.filter(UserConfigurations.session_id == session_id)
                if favorites_only:
                    query = query.filter(UserConfigurations.is_favorite == True)
                
                records = query.order_by(UserConfigurations.created_at.desc()).all()
                
                data = []
                for record in records:
                    data.append({
                        'id': record.id,
                        'name': record.name,
                        'config_type': record.config_type,
                        'configuration': json.loads(record.configuration),
                        'created_at': record.created_at,
                        'is_favorite': record.is_favorite
                    })
                
                return data
        except Exception as e:
            print(f"Error retrieving configurations: {e}")
            return []
    
    def get_analytics_summary(self):
        """Get analytics summary from database"""
        try:
            with self.get_session() as session:
                # Flight statistics
                total_flights = session.query(FlightData).count()
                avg_fuel_efficiency = session.query(func.avg(FlightData.fuel_efficiency)).scalar() or 0
                total_predictions = session.query(PredictionHistory).count()
                total_optimizations = session.query(RouteOptimization).count()
                
                # Popular aircraft types
                popular_aircraft = session.query(
                    FlightData.aircraft_type,
                    func.count(FlightData.aircraft_type).label('count')
                ).group_by(FlightData.aircraft_type).order_by(func.count(FlightData.aircraft_type).desc()).limit(5).all()
                
                # Popular routes
                popular_routes = session.query(
                    FlightData.origin,
                    FlightData.destination,
                    func.count().label('count')
                ).group_by(FlightData.origin, FlightData.destination).order_by(func.count().desc()).limit(5).all()
                
                return {
                    'total_flights': total_flights,
                    'avg_fuel_efficiency': round(avg_fuel_efficiency, 1),
                    'total_predictions': total_predictions,
                    'total_optimizations': total_optimizations,
                    'popular_aircraft': [{'type': aircraft, 'count': count} for aircraft, count in popular_aircraft],
                    'popular_routes': [{'route': f"{origin}-{dest}", 'count': count} for origin, dest, count in popular_routes]
                }
        except Exception as e:
            print(f"Error getting analytics: {e}")
            return {
                'total_flights': 0,
                'avg_fuel_efficiency': 0,
                'total_predictions': 0,
                'total_optimizations': 0,
                'popular_aircraft': [],
                'popular_routes': []
            }