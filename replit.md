# Flight Predictor ML Dashboard

## Overview

This is a Streamlit-based machine learning dashboard for aircraft performance prediction and flight path optimization. The application uses neural networks to predict flight performance metrics and optimize flight paths based on various atmospheric and aircraft parameters.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with enhanced UX features
- **Layout**: Multi-page dashboard with sidebar navigation and interactive elements
- **Visualization**: Plotly for interactive charts and 3D visualizations
- **Pages**: 
  - Dashboard Overview (with real-time metrics)
  - Performance Predictor (with quick presets)
  - Flight Path Optimizer (with strategy selection)
  - Historical Data Analysis (database integration)
  - Database Analytics (data management)
  - Settings & Preferences
  - Help & Tutorial

### Backend Architecture
- **Core Engine**: Python-based ML pipeline
- **Machine Learning**: Scikit-learn MLPRegressor (Multi-layer Perceptron)
- **Data Processing**: Pandas and NumPy for data manipulation
- **Model Architecture**: Neural networks with multiple hidden layers
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Data Persistence**: Flight data, predictions, and user configurations stored in database

## Key Components

### 1. Flight Predictor (`models/flight_predictor.py`)
- **Purpose**: Predicts aircraft performance metrics using ML
- **Model**: 3-layer neural network (100, 50, 25 neurons)
- **Features**: Weight, altitude, speed, temperature, wind conditions, humidity, aircraft type
- **Training**: Synthetic data generation based on flight physics

### 2. Path Optimizer (`models/path_optimizer.py`)
- **Purpose**: Optimizes flight routes for fuel efficiency and time
- **Components**: 
  - Fuel optimizer neural network
  - Time optimizer neural network
  - Weather optimizer neural network
- **Airport Database**: Predefined coordinates for major international airports

### 3. Database Manager (`database/db_manager.py`)
- **Purpose**: PostgreSQL database integration for data persistence
- **Tables**: 
  - Flight data storage (historical records)
  - Prediction history (ML results)
  - Route optimizations (optimization results)
  - User configurations (saved settings)
- **Features**: SQLAlchemy ORM, analytics queries, data management

### 4. Data Generator (`data/data_generator.py`)
- **Purpose**: Generates synthetic flight data for training and testing
- **Features**: 
  - Realistic aircraft types (Boeing 737, Airbus A320, etc.)
  - Major airport routes
  - Time-based flight patterns
  - Poisson distribution for daily flight counts

### 5. Atmospheric Data Provider (`utils/atmospheric_data.py`)
- **Purpose**: Simulates real-time atmospheric conditions
- **Weather Patterns**: Clear, cloudy, stormy, windy conditions
- **Altitude Layers**: Surface to very high altitude conditions
- **Time-based Patterns**: Different weather probabilities by time of day

### 6. Visualization Utilities (`utils/visualization.py`)
- **Gauge Charts**: Performance metrics visualization
- **3D Flight Paths**: Interactive flight route visualization
- **Plotly Integration**: Modern, interactive charts

### 7. Help System (`utils/help_system.py`)
- **Interactive Tutorial**: Step-by-step user guidance
- **Feature Highlights**: Key functionality explanations
- **Aviation Glossary**: Technical terms and definitions

## Data Flow

1. **Data Generation**: Synthetic flight data is generated based on realistic aviation parameters
2. **Model Training**: Neural networks are trained on synthetic data with aviation physics constraints
3. **Real-time Prediction**: User inputs are processed through trained models
4. **Visualization**: Results are displayed through interactive Plotly charts
5. **Optimization**: Flight paths are optimized based on multiple objective functions

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms
- **plotly**: Interactive visualization

### Database Dependencies
- **psycopg2-binary**: PostgreSQL database adapter
- **sqlalchemy**: ORM for database operations
- **uuid**: Session and unique identifier generation

### ML Dependencies
- **MLPRegressor**: Neural network implementation
- **StandardScaler**: Feature normalization
- **LabelEncoder**: Categorical variable encoding

## Deployment Strategy

### Local Development
- **Entry Point**: `app.py` - main Streamlit application
- **Page Structure**: Multi-page app with sidebar navigation
- **Session State**: Persistent model instances across user sessions

### Production Considerations
- Models are initialized once per session for performance
- Synthetic data generation allows offline operation
- Modular architecture supports easy scaling and feature addition

## Changelog

- July 04, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.