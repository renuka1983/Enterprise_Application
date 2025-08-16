"""
Machine Learning Forecasting Module for Energy Optimization
Energy Consumption Prediction using Scikit-learn Regression Models
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class EnergyConsumptionForecaster:
    """
    Machine learning models for forecasting energy consumption
    with multiple algorithms and performance comparison.
    """
    
    def __init__(self):
        """Initialize the energy consumption forecaster."""
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.predictions = {}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different ML models for energy forecasting."""
        
        # Linear models
        self.models['Linear Regression'] = LinearRegression()
        self.models['Ridge Regression'] = Ridge(alpha=1.0)
        self.models['Lasso Regression'] = Lasso(alpha=0.1)
        
        # Ensemble models
        self.models['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
        self.models['Gradient Boosting'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        # Initialize scalers
        for name in self.models.keys():
            self.scalers[name] = StandardScaler()
    
    def prepare_features(self, energy_df, production_df, forecast_days=30):
        """
        Prepare features for energy consumption forecasting.
        
        Args:
            energy_df (pd.DataFrame): Energy consumption data
            production_df (pd.DataFrame): Production data
            forecast_days (int): Number of days to forecast
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_names, scaler)
        """
        
        # Merge energy and production data
        df = energy_df.merge(production_df, on=['Date', 'Plant', 'PlantType'])
        
        # Create time-based features
        df['Date'] = pd.to_datetime(df['Date'])
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        
        # Create lag features
        df['EnergyConsumption_Lag1'] = df.groupby('Plant')['EnergyConsumption_kWh'].shift(1)
        df['EnergyConsumption_Lag7'] = df.groupby('Plant')['EnergyConsumption_kWh'].shift(7)
        df['Production_Lag1'] = df.groupby('Plant')['Production_Units'].shift(1)
        
        # Create rolling features
        df['EnergyConsumption_MA7'] = df.groupby('Plant')['EnergyConsumption_kWh'].rolling(7).mean().reset_index(0, drop=True)
        df['EnergyConsumption_MA30'] = df.groupby('Plant')['EnergyConsumption_kWh'].rolling(30).mean().reset_index(0, drop=True)
        
        # Create interaction features
        df['Production_Energy_Ratio'] = df['Production_Units'] / df['EnergyConsumption_kWh']
        df['Efficiency_Trend'] = df['EnergyEfficiency'] * df['ProductionEfficiency']
        
        # Create seasonal features
        df['Seasonal_Sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['Seasonal_Cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        
        # Drop rows with NaN values (from lag features)
        df = df.dropna()
        
        # Define feature columns
        feature_cols = [
            'DayOfYear', 'Month', 'DayOfWeek', 'WeekOfYear',
            'Production_Units', 'Downtime_Hours', 'ProductionEfficiency',
            'EnergyConsumption_Lag1', 'EnergyConsumption_Lag7',
            'Production_Lag1', 'EnergyConsumption_MA7', 'EnergyConsumption_MA30',
            'Production_Energy_Ratio', 'Efficiency_Trend',
            'Seasonal_Sin', 'Seasonal_Cos'
        ]
        
        # Target variable
        target_col = 'EnergyConsumption_kWh'
        
        # Prepare X and y
        X = df[feature_cols]
        y = df[target_col]
        
        # Split data (last 30 days for testing)
        split_idx = len(df) - forecast_days
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test, feature_cols, df
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """
        Train all models and evaluate performance.
        
        Args:
            X_train, X_test, y_train, y_test: Training and testing data
            
        Returns:
            dict: Model performance metrics
        """
        
        for model_name, model in self.models.items():
            # Create pipeline with scaling
            pipeline = Pipeline([
                ('scaler', self.scalers[model_name]),
                ('model', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = pipeline.predict(X_train)
            y_pred_test = pipeline.predict(X_test)
            
            # Store predictions
            self.predictions[model_name] = {
                'train': y_pred_train,
                'test': y_pred_test
            }
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Cross-validation score
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store performance metrics
            self.model_performance[model_name] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_mean': cv_mean,
                'cv_std': cv_std
            }
            
            # Store feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = pipeline.named_steps['model'].feature_importances_
            elif hasattr(model, 'coef_'):
                self.feature_importance[model_name] = pipeline.named_steps['model'].coef_
        
        return self.model_performance
    
    def create_model_comparison_chart(self):
        """Create comparison visualization of model performance."""
        
        # Prepare data for plotting
        model_names = list(self.model_performance.keys())
        metrics = ['test_rmse', 'test_mae', 'test_r2', 'cv_mean']
        metric_labels = ['Test RMSE', 'Test MAE', 'Test R²', 'CV Score']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metric_labels,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            values = [self.model_performance[name][metric] for name in model_names]
            
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=values,
                    name=label,
                    marker_color=colors[:len(model_names)],
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add value labels on bars
            for j, value in enumerate(values):
                fig.add_annotation(
                    x=model_names[j],
                    y=value,
                    text=f"{value:.3f}",
                    showarrow=False,
                    yshift=10,
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Model Performance Comparison",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_feature_importance_chart(self, feature_names, model_name='Random Forest'):
        """Create feature importance visualization."""
        
        if model_name not in self.feature_importance:
            return None
        
        importance = self.feature_importance[model_name]
        
        # Create DataFrame for sorting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': abs(importance)
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"Feature Importance - {model_name}",
            color='Importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=500,
            xaxis_title="Feature Importance",
            yaxis_title="Features"
        )
        
        return fig
    
    def create_forecast_visualization(self, y_train, y_test, model_name='Random Forest'):
        """Create visualization of actual vs predicted values."""
        
        if model_name not in self.predictions:
            return None
        
        predictions = self.predictions[model_name]
        
        # Create date range for x-axis
        train_dates = pd.date_range(start='2023-01-01', periods=len(y_train), freq='D')
        test_dates = pd.date_range(start=train_dates[-1] + pd.Timedelta(days=1), periods=len(y_test), freq='D')
        
        fig = go.Figure()
        
        # Add training data
        fig.add_trace(go.Scatter(
            x=train_dates,
            y=y_train,
            mode='lines',
            name='Actual (Training)',
            line=dict(color='blue', width=2)
        ))
        
        # Add training predictions
        fig.add_trace(go.Scatter(
            x=train_dates,
            y=predictions['train'],
            mode='lines',
            name='Predicted (Training)',
            line=dict(color='lightblue', width=2, dash='dash')
        ))
        
        # Add test data
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=y_test,
            mode='lines',
            name='Actual (Test)',
            line=dict(color='red', width=2)
        ))
        
        # Add test predictions
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=predictions['test'],
            mode='lines',
            name='Predicted (Test)',
            line=dict(color='orange', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f"Energy Consumption Forecast - {model_name}",
            xaxis_title="Date",
            yaxis_title="Energy Consumption (kWh)",
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def get_model_summary(self):
        """Return DataFrame of model performance summary."""
        
        summary_data = []
        for model_name, metrics in self.model_performance.items():
            summary_data.append({
                'Model': model_name,
                'Test RMSE': f"{metrics['test_rmse']:.2f}",
                'Test MAE': f"{metrics['test_mae']:.2f}",
                'Test R²': f"{metrics['test_r2']:.3f}",
                'CV Score': f"{metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}"
            })
        
        return pd.DataFrame(summary_data)
    
    def forecast_future_consumption(self, energy_df, production_df, days_ahead=30, model_name='Random Forest'):
        """
        Forecast energy consumption for future days.
        
        Args:
            energy_df (pd.DataFrame): Historical energy data
            production_df (pd.DataFrame): Historical production data
            days_ahead (int): Number of days to forecast
            model_name (str): Name of the model to use
            
        Returns:
            pd.DataFrame: Forecasted energy consumption
        """
        
        # Prepare features for the last available data
        X_train, X_test, y_train, y_test, feature_names, df = self.prepare_features(energy_df, production_df)
        
        # Train models if not already trained
        if not self.model_performance:
            self.train_models(X_train, X_test, y_train, y_test)
        
        # Get the trained pipeline
        pipeline = Pipeline([
            ('scaler', self.scalers[model_name]),
            ('model', self.models[model_name])
        ])
        pipeline.fit(X_train, y_train)
        
        # Get the last row of features for forecasting
        last_features = X_test.iloc[-1:].copy()
        
        # Generate future dates
        last_date = df['Date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='D')
        
        forecasts = []
        current_features = last_features.copy()
        
        for i, future_date in enumerate(future_dates):
            # Update time-based features
            current_features['DayOfYear'] = future_date.dayofyear
            current_features['Month'] = future_date.month
            current_features['DayOfWeek'] = future_date.dayofweek
            current_features['WeekOfYear'] = future_date.isocalendar().week
            current_features['Seasonal_Sin'] = np.sin(2 * np.pi * future_date.dayofyear / 365)
            current_features['Seasonal_Cos'] = np.cos(2 * np.pi * future_date.dayofyear / 365)
            
            # Make prediction
            prediction = pipeline.predict(current_features)[0]
            
            # Update lag features for next iteration
            if i > 0:
                current_features['EnergyConsumption_Lag1'] = forecasts[-1]['EnergyConsumption_kWh']
                current_features['Production_Lag1'] = forecasts[-1]['Production_Units']
            
            forecasts.append({
                'Date': future_date,
                'EnergyConsumption_kWh': max(prediction, 0),  # Ensure non-negative
                'Production_Units': current_features['Production_Units'].iloc[0],
                'Forecast_Type': 'ML Prediction'
            })
        
        return pd.DataFrame(forecasts)
