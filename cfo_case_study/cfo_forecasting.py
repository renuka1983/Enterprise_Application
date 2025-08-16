import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import warnings
warnings.filterwarnings('ignore')

def traditional_forecasting(df, forecast_months=3):
    """
    Traditional Excel-style calculation for forecasting Cash On Hand.
    """
    # Get the last row for starting values
    last_row = df.iloc[-1]
    
    # Simple assumptions for forecasting
    revenue_growth_rate = 0.02  # 2% monthly growth
    cost_ratio = last_row['Cost'] / last_row['Revenue']  # Maintain cost ratio
    opex_ratio = last_row['Operating_Expenses'] / last_row['Revenue']  # Maintain opex ratio
    capex_ratio = last_row['Capex'] / last_row['Revenue']  # Maintain capex ratio
    
    forecast_data = []
    current_cash = last_row['Cash_On_Hand']
    
    for i in range(1, forecast_months + 1):
        # Project revenue with growth
        projected_revenue = last_row['Revenue'] * (1 + revenue_growth_rate) ** i
        
        # Project costs maintaining ratios
        projected_cost = projected_revenue * cost_ratio
        projected_opex = projected_revenue * opex_ratio
        projected_capex = projected_revenue * capex_ratio
        
        # Calculate cash flow
        cash_flow = projected_revenue - projected_cost - projected_opex - projected_capex
        current_cash += cash_flow
        
        forecast_data.append({
            'Month': f'Forecast_{i}',
            'Revenue': round(projected_revenue, 2),
            'Cost': round(projected_cost, 2),
            'Operating_Expenses': round(projected_opex, 2),
            'Capex': round(projected_capex, 2),
            'Cash_On_Hand': round(current_cash, 2),
            'Net_Cash_Flow': round(cash_flow, 2),
            'Method': 'Traditional'
        })
    
    return pd.DataFrame(forecast_data)

def ml_forecasting(df, forecast_months=3):
    """
    Machine Learning approach using Random Forest for forecasting.
    """
    # Prepare features
    df_ml = df.copy()
    df_ml['Month_Num'] = range(len(df_ml))
    df_ml['Revenue_Lag1'] = df_ml['Revenue'].shift(1)
    df_ml['Cost_Lag1'] = df_ml['Cost'].shift(1)
    df_ml['Opex_Lag1'] = df_ml['Operating_Expenses'].shift(1)
    df_ml['Capex_Lag1'] = df_ml['Capex'].shift(1)
    df_ml['Cash_Lag1'] = df_ml['Cash_On_Hand'].shift(1)
    
    # Add seasonal features
    df_ml['Month_of_Year'] = pd.to_datetime(df_ml['Month'] + '-01').dt.month
    df_ml['Quarter'] = pd.to_datetime(df_ml['Month'] + '-01').dt.quarter
    
    # Drop NaN rows
    df_ml = df_ml.dropna()
    
    # Features for prediction
    feature_cols = ['Month_Num', 'Revenue_Lag1', 'Cost_Lag1', 'Opex_Lag1', 
                   'Capex_Lag1', 'Cash_Lag1', 'Month_of_Year', 'Quarter']
    
    X = df_ml[feature_cols]
    y = df_ml['Cash_On_Hand']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Prepare forecast data
    last_row = df_ml.iloc[-1]
    forecast_data = []
    current_cash = last_row['Cash_On_Hand']
    
    for i in range(1, forecast_months + 1):
        # Create feature vector for prediction
        next_month = last_row['Month_Num'] + i
        next_month_of_year = ((last_row['Month_of_Year'] + i - 1) % 12) + 1
        next_quarter = ((next_month_of_year - 1) // 3) + 1
        
        features = np.array([[
            next_month,
            last_row['Revenue'],
            last_row['Cost'],
            last_row['Operating_Expenses'],
            last_row['Capex'],
            current_cash,
            next_month_of_year,
            next_quarter
        ]])
        
        # Predict cash on hand
        predicted_cash = rf_model.predict(features)[0]
        
        # Calculate other metrics (simplified)
        revenue_growth = 0.02
        projected_revenue = last_row['Revenue'] * (1 + revenue_growth) ** i
        projected_cost = projected_revenue * 0.72  # Assume 72% cost ratio
        projected_opex = projected_revenue * 0.16  # Assume 16% opex ratio
        projected_capex = projected_revenue * 0.06  # Assume 6% capex ratio
        
        cash_flow = projected_revenue - projected_cost - projected_opex - projected_capex
        current_cash = predicted_cash
        
        forecast_data.append({
            'Month': f'Forecast_{i}',
            'Revenue': round(projected_revenue, 2),
            'Cost': round(projected_cost, 2),
            'Operating_Expenses': round(projected_opex, 2),
            'Capex': round(projected_capex, 2),
            'Cash_On_Hand': round(current_cash, 2),
            'Net_Cash_Flow': round(cash_flow, 2),
            'Method': 'ML_RandomForest'
        })
        
        # Update last row for next iteration (preserve engineered features)
        last_row = df_ml.iloc[-1].copy()
        last_row['Revenue'] = projected_revenue
        last_row['Cost'] = projected_cost
        last_row['Operating_Expenses'] = projected_opex
        last_row['Capex'] = projected_capex
        last_row['Cash_On_Hand'] = current_cash
    
    return pd.DataFrame(forecast_data), rf_model, feature_cols, {'MAE': mae, 'R2': r2}

def ai_forecasting_with_shap(df, forecast_months=3):
    """
    AI approach using sklearn pipeline with SHAP explainability.
    """
    # Prepare features (more sophisticated than ML approach)
    df_ai = df.copy()
    df_ai['Month_Num'] = range(len(df_ai))
    
    # Create lag features
    for lag in [1, 2, 3]:
        df_ai[f'Revenue_Lag{lag}'] = df_ai['Revenue'].shift(lag)
        df_ai[f'Cost_Lag{lag}'] = df_ai['Cost'].shift(lag)
        df_ai[f'Cash_Lag{lag}'] = df_ai['Cash_On_Hand'].shift(lag)
    
    # Add rolling statistics
    df_ai['Revenue_MA3'] = df_ai['Revenue'].rolling(3).mean()
    df_ai['Cost_MA3'] = df_ai['Cost'].rolling(3).mean()
    df_ai['Cash_MA3'] = df_ai['Cash_On_Hand'].rolling(3).mean()
    
    # Add seasonal and trend features
    df_ai['Month_of_Year'] = pd.to_datetime(df_ai['Month'] + '-01').dt.month
    df_ai['Quarter'] = pd.to_datetime(df_ai['Month'] + '-01').dt.quarter
    df_ai['Revenue_Trend'] = df_ai['Revenue'].pct_change()
    df_ai['Cost_Trend'] = df_ai['Cost'].pct_change()
    
    # Add financial ratios
    df_ai['Gross_Margin_Ratio'] = (df_ai['Revenue'] - df_ai['Cost']) / df_ai['Revenue']
    df_ai['Opex_Ratio'] = df_ai['Operating_Expenses'] / df_ai['Revenue']
    df_ai['Capex_Ratio'] = df_ai['Capex'] / df_ai['Revenue']
    
    # Drop NaN rows
    df_ai = df_ai.dropna()
    
    # Features for prediction
    feature_cols = [col for col in df_ai.columns if col not in ['Month', 'Cash_On_Hand', 'Method']]
    
    X = df_ai[feature_cols]
    y = df_ai['Cash_On_Hand']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create pipeline with scaling and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10))
    ])
    
    # Train pipeline
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # SHAP explainability
    explainer = shap.TreeExplainer(pipeline.named_steps['regressor'])
    X_scaled = pipeline.named_steps['scaler'].transform(X_test)
    shap_values = explainer.shap_values(X_scaled)
    
    # Prepare forecast data
    last_row = df_ai.iloc[-1]
    forecast_data = []
    current_cash = last_row['Cash_On_Hand']
    
    for i in range(1, forecast_months + 1):
        # Create feature vector for prediction
        features = {}
        for col in feature_cols:
            if 'Lag' in col:
                if 'Revenue' in col:
                    features[col] = last_row['Revenue']
                elif 'Cost' in col:
                    features[col] = last_row['Cost']
                elif 'Cash' in col:
                    features[col] = current_cash
            elif 'MA' in col:
                features[col] = last_row[col]
            elif 'Trend' in col:
                features[col] = 0.02  # Assume 2% trend
            elif 'Ratio' in col:
                features[col] = last_row[col]
            elif col == 'Month_Num':
                features[col] = last_row[col] + i
            elif col == 'Month_of_Year':
                features[col] = ((last_row[col] + i - 1) % 12) + 1
            elif col == 'Quarter':
                features[col] = ((features['Month_of_Year'] - 1) // 3) + 1
            else:
                features[col] = last_row[col]
        
        # Create feature array
        feature_array = np.array([[features[col] for col in feature_cols]])
        
        # Predict cash on hand
        predicted_cash = pipeline.predict(feature_array)[0]
        
        # Calculate other metrics
        revenue_growth = 0.02
        projected_revenue = last_row['Revenue'] * (1 + revenue_growth) ** i
        projected_cost = projected_revenue * 0.72
        projected_opex = projected_revenue * 0.16
        projected_capex = projected_revenue * 0.06
        
        cash_flow = projected_revenue - projected_cost - projected_opex - projected_capex
        current_cash = predicted_cash
        
        forecast_data.append({
            'Month': f'Forecast_{i}',
            'Revenue': round(projected_revenue, 2),
            'Cost': round(projected_cost, 2),
            'Operating_Expenses': round(projected_opex, 2),
            'Capex': round(projected_capex, 2),
            'Cash_On_Hand': round(current_cash, 2),
            'Net_Cash_Flow': round(cash_flow, 2),
            'Method': 'AI_Pipeline'
        })
        
        # Update last row for next iteration (preserve engineered features)
        last_row = df_ai.iloc[-1].copy()
        last_row['Revenue'] = projected_revenue
        last_row['Cost'] = projected_cost
        last_row['Operating_Expenses'] = projected_opex
        last_row['Capex'] = projected_capex
        last_row['Cash_On_Hand'] = current_cash
    
    return pd.DataFrame(forecast_data), pipeline, feature_cols, {'MAE': mae, 'R2': r2}, shap_values, X_test

def create_forecast_comparison(df_original, df_traditional, df_ml, df_ai):
    """
    Create comparison visualization of different forecasting methods.
    """
    # Combine all data for plotting
    df_combined = pd.concat([
        df_original[['Month', 'Cash_On_Hand']].assign(Method='Historical'),
        df_traditional[['Month', 'Cash_On_Hand']].assign(Method='Traditional'),
        df_ml[['Month', 'Cash_On_Hand']].assign(Method='ML'),
        df_ai[['Month', 'Cash_On_Hand']].assign(Method='AI')
    ], ignore_index=True)
    
    # Create comparison plot
    fig = go.Figure()
    
    methods = ['Historical', 'Traditional', 'ML', 'AI']
    colors = ['blue', 'red', 'green', 'purple']
    
    for method, color in zip(methods, colors):
        method_data = df_combined[df_combined['Method'] == method]
        fig.add_trace(
            go.Scatter(
                x=method_data['Month'],
                y=method_data['Cash_On_Hand'],
                mode='lines+markers',
                name=method,
                line=dict(color=color, width=2),
                marker=dict(size=6)
            )
        )
    
    fig.update_layout(
        title="Cash On Hand Forecast Comparison - All Methods",
        xaxis_title="Month",
        yaxis_title="Cash On Hand ($)",
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig
