import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

def generate_cfo_financial_data(months=24, seed=42):
    """
    Generate synthetic CFO financial data with realistic trends, seasonality, and noise.
    
    Parameters:
    - months: Number of months to generate (default: 24)
    - seed: Random seed for reproducibility (default: 42)
    
    Returns:
    - pandas DataFrame with financial data
    """
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate date range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=30*i) for i in range(months)]
    
    # Base values and trends
    base_revenue = 2500000  # $2.5M base monthly revenue
    revenue_trend = 0.02    # 2% monthly growth trend
    revenue_seasonality = 0.15  # 15% seasonal variation
    
    base_cost = 1800000     # $1.8M base monthly cost (72% of revenue)
    cost_trend = 0.018      # 1.8% monthly growth (slightly slower than revenue)
    cost_seasonality = 0.12 # 12% seasonal variation
    
    base_opex = 400000      # $400K base monthly operating expenses
    opex_trend = 0.015      # 1.5% monthly growth
    opex_seasonality = 0.08 # 8% seasonal variation
    
    base_capex = 150000     # $150K base monthly capital expenditure
    capex_trend = 0.025     # 2.5% monthly growth
    capex_seasonality = 0.25 # 25% seasonal variation (higher for capex)
    
    # Initialize data storage
    data = []
    cash_on_hand = 500000   # Starting cash position
    
    for i, date in enumerate(dates):
        # Calculate month number (0-23)
        month_num = i
        
        # Revenue generation with trend, seasonality, and noise
        trend_factor = (1 + revenue_trend) ** month_num
        seasonal_factor = 1 + revenue_seasonality * np.sin(2 * np.pi * month_num / 12)
        noise = np.random.normal(0, 0.05)  # 5% random noise
        revenue = base_revenue * trend_factor * seasonal_factor * (1 + noise)
        
        # Cost generation (correlated with revenue but with different patterns)
        cost_trend_factor = (1 + cost_trend) ** month_num
        cost_seasonal = 1 + cost_seasonality * np.sin(2 * np.pi * month_num / 12 + np.pi/6)  # Phase shift
        cost_noise = np.random.normal(0, 0.04)  # 4% random noise
        cost = base_cost * cost_trend_factor * cost_seasonal * (1 + cost_noise)
        
        # Operating expenses
        opex_trend_factor = (1 + opex_trend) ** month_num
        opex_seasonal = 1 + opex_seasonality * np.sin(2 * np.pi * month_num / 12 + np.pi/3)
        opex_noise = np.random.normal(0, 0.03)  # 3% random noise
        operating_expenses = base_opex * opex_trend_factor * opex_seasonal * (1 + opex_noise)
        
        # Capital expenditure (more volatile)
        capex_trend_factor = (1 + capex_trend) ** month_num
        capex_seasonal = 1 + capex_seasonality * np.sin(2 * np.pi * month_num / 12 + np.pi/2)
        capex_noise = np.random.normal(0, 0.08)  # 8% random noise
        capex = base_capex * capex_trend_factor * capex_seasonal * (1 + capex_noise)
        
        # Ensure capex doesn't go negative
        capex = max(0, capex)
        
        # Calculate cash flow and update cash on hand
        cash_flow = revenue - cost - operating_expenses - capex
        cash_on_hand += cash_flow
        
        # Add some realistic constraints
        if cash_on_hand < 100000:  # If cash gets too low, reduce capex
            capex = max(0, capex * 0.5)
            cash_flow = revenue - cost - operating_expenses - capex
            cash_on_hand = cash_on_hand - (capex * 0.5) + cash_flow
        
        # Store the data
        data.append({
            'Month': date.strftime('%Y-%m'),
            'Revenue': round(revenue, 2),
            'Cost': round(cost, 2),
            'Operating_Expenses': round(operating_expenses, 2),
            'Capex': round(capex, 2),
            'Cash_On_Hand': round(cash_on_hand, 2),
            'Gross_Margin': round(revenue - cost, 2),
            'Operating_Income': round(revenue - cost - operating_expenses, 2),
            'Net_Cash_Flow': round(cash_flow, 2)
        })
    
    return pd.DataFrame(data)

def create_financial_dashboard(df):
    """
    Create an interactive financial dashboard using Plotly.
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Revenue vs Cost Trends', 'Cash Position Over Time', 
                       'Monthly Cash Flow', 'Financial Ratios'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Revenue and Cost Trends
    fig.add_trace(
        go.Scatter(x=df['Month'], y=df['Revenue'], name='Revenue', 
                  line=dict(color='blue', width=2), mode='lines+markers'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['Month'], y=df['Cost'], name='Cost', 
                  line=dict(color='red', width=2), mode='lines+markers'),
        row=1, col=1
    )
    
    # 2. Cash Position Over Time
    fig.add_trace(
        go.Scatter(x=df['Month'], y=df['Cash_On_Hand'], name='Cash on Hand', 
                  line=dict(color='green', width=2), mode='lines+markers'),
        row=1, col=2
    )
    
    # 3. Monthly Cash Flow
    colors = ['green' if x > 0 else 'red' for x in df['Net_Cash_Flow']]
    fig.add_trace(
        go.Bar(x=df['Month'], y=df['Net_Cash_Flow'], name='Net Cash Flow', 
               marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    
    # 4. Financial Ratios (Gross Margin %)
    gross_margin_pct = (df['Gross_Margin'] / df['Revenue']) * 100
    fig.add_trace(
        go.Scatter(x=df['Month'], y=gross_margin_pct, name='Gross Margin %', 
                  line=dict(color='purple', width=2), mode='lines+markers'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="CFO Financial Dashboard - 24 Month Analysis",
        showlegend=True
    )
    
    # Update x-axis labels
    fig.update_xaxes(tickangle=45)
    
    return fig

def save_to_csv(df, filename='cfo_financial_data.csv'):
    """
    Save the DataFrame to CSV file.
    """
    csv_data = df.to_csv(index=False)
    return csv_data, filename
