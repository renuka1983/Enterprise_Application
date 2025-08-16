import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

def get_openai_api_key():
    """
    Get OpenAI API key from environment variables or Streamlit secrets.
    """
    # Try to get from environment variables first
    api_key = os.getenv('OPENAI_API_KEY')
    
    # If not found, try Streamlit secrets
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except:
            api_key = None
    
    return api_key

def analyze_financial_health(df):
    """
    Analyze financial health metrics from the data.
    """
    analysis = {}
    
    # Revenue trends
    revenue_trend = df['Revenue'].pct_change().mean()
    revenue_volatility = df['Revenue'].pct_change().std()
    
    # Cost structure
    avg_cost_ratio = (df['Cost'] / df['Revenue']).mean()
    avg_opex_ratio = (df['Operating_Expenses'] / df['Revenue']).mean()
    avg_capex_ratio = (df['Capex'] / df['Revenue']).mean()
    
    # Cash flow analysis
    positive_cash_flow_months = len(df[df['Net_Cash_Flow'] > 0])
    cash_flow_ratio = positive_cash_flow_months / len(df)
    
    # Gross margin analysis
    gross_margin = ((df['Revenue'] - df['Cost']) / df['Revenue']).mean()
    
    analysis = {
        'revenue_trend': revenue_trend,
        'revenue_volatility': revenue_volatility,
        'avg_cost_ratio': avg_cost_ratio,
        'avg_opex_ratio': avg_opex_ratio,
        'avg_capex_ratio': avg_capex_ratio,
        'cash_flow_ratio': cash_flow_ratio,
        'gross_margin': gross_margin,
        'total_months': len(df)
    }
    
    return analysis

def generate_cfo_insights(df, forecast_results, analysis):
    """
    Generate CFO insights based on historical data and forecasts.
    """
    insights = []
    
    # Revenue insights
    if analysis['revenue_trend'] > 0.01:
        insights.append("📈 Revenue shows strong positive growth trend")
    elif analysis['revenue_trend'] > 0:
        insights.append("📊 Revenue shows moderate growth")
    else:
        insights.append("⚠️ Revenue shows declining trend")
    
    # Cost structure insights
    if analysis['avg_cost_ratio'] > 0.75:
        insights.append("⚠️ High cost structure - consider cost optimization")
    elif analysis['avg_cost_ratio'] < 0.65:
        insights.append("✅ Efficient cost structure maintained")
    
    # Cash flow insights
    if analysis['cash_flow_ratio'] > 0.7:
        insights.append("✅ Strong cash flow generation")
    elif analysis['cash_flow_ratio'] < 0.5:
        insights.append("⚠️ Cash flow challenges - review operations")
    
    # Gross margin insights
    if analysis['gross_margin'] > 0.3:
        insights.append("✅ Healthy gross margins")
    elif analysis['gross_margin'] < 0.2:
        insights.append("⚠️ Low gross margins - pricing review needed")
    
    return insights

def generate_cfo_recommendations(df, forecast_results, analysis):
    """
    Generate actionable CFO recommendations.
    """
    recommendations = []
    
    # Revenue recommendations
    if analysis['revenue_trend'] < 0.015:
        recommendations.append("🚀 **Revenue Growth**: Implement pricing optimization and market expansion strategies")
    
    # Cost management
    if analysis['avg_cost_ratio'] > 0.75:
        recommendations.append("💰 **Cost Control**: Review supplier contracts and operational efficiency")
    
    # Cash flow management
    if analysis['cash_flow_ratio'] < 0.6:
        recommendations.append("💵 **Cash Flow**: Optimize working capital and payment terms")
    
    # Capital allocation
    if analysis['avg_capex_ratio'] > 0.1:
        recommendations.append("🏗️ **Capital Allocation**: Review capex ROI and prioritize high-impact projects")
    
    # Risk management
    if analysis['revenue_volatility'] > 0.1:
        recommendations.append("🛡️ **Risk Management**: Diversify revenue streams and implement hedging strategies")
    
    return recommendations

def create_executive_summary(df, forecast_results, analysis):
    """
    Create an executive summary for CFO presentation.
    """
    summary = f"""
## 📊 Executive Summary - CFO Financial Analysis

### 🎯 **Key Performance Indicators (24 Months)**
- **Total Revenue**: ${df['Revenue'].sum():,.0f}
- **Average Monthly Revenue**: ${df['Revenue'].mean():,.0f}
- **Revenue Growth Trend**: {analysis['revenue_trend']*100:.1f}% monthly
- **Gross Margin**: {analysis['gross_margin']*100:.1f}%
- **Cash Flow Positive Months**: {analysis['cash_flow_ratio']*100:.1f}%

### 📈 **Forecast Outlook (Next 3 Months)**
- **Traditional Method**: ${forecast_results['traditional']['Cash_On_Hand'].iloc[-1]:,.0f}
- **ML Method**: ${forecast_results['ml']['Cash_On_Hand'].iloc[-1]:,.0f}
- **AI Method**: ${forecast_results['ai']['Cash_On_Hand'].iloc[-1]:,.0f}

### 🔍 **Financial Health Assessment**
- **Cost Structure**: {'Efficient' if analysis['avg_cost_ratio'] < 0.7 else 'Needs Review'}
- **Cash Flow**: {'Strong' if analysis['cash_flow_ratio'] > 0.7 else 'Moderate' if analysis['cash_flow_ratio'] > 0.5 else 'Challenged'}
- **Profitability**: {'High' if analysis['gross_margin'] > 0.3 else 'Moderate' if analysis['gross_margin'] > 0.2 else 'Low'}

### ⚠️ **Risk Factors**
- Revenue volatility: {analysis['revenue_volatility']*100:.1f}%
- Cash flow consistency: {analysis['cash_flow_ratio']*100:.1f}%
- Cost structure efficiency: {analysis['avg_cost_ratio']*100:.1f}%
"""
    
    return summary

def create_ai_narrative(df, forecast_results, analysis):
    """
    Create AI-generated narrative explanation of financial results.
    """
    # This would normally call OpenAI API, but for demo purposes we'll create a structured narrative
    narrative = f"""
## 🤖 AI-Generated Financial Analysis Narrative

### 📊 **Current Financial Position**
Based on the analysis of your 24-month financial data, your organization demonstrates a **{analysis['revenue_trend']*100:.1f}% monthly revenue growth rate** with a **{analysis['gross_margin']*100:.1f}% gross margin**. 

The cash flow analysis reveals that **{analysis['cash_flow_ratio']*100:.1f}% of months** generated positive cash flow, indicating {'strong operational efficiency' if analysis['cash_flow_ratio'] > 0.7 else 'moderate cash generation' if analysis['cash_flow_ratio'] > 0.5 else 'cash flow challenges that require attention'}.

### 🔮 **Forecast Insights**
Our AI models project the following cash positions for the next 3 months:

1. **Traditional Excel Method**: ${forecast_results['traditional']['Cash_On_Hand'].iloc[-1]:,.0f}
2. **Machine Learning Model**: ${forecast_results['ml']['Cash_On_Hand'].iloc[-1]:,.0f}
3. **Advanced AI Pipeline**: ${forecast_results['ai']['Cash_On_Hand'].iloc[-1]:,.0f}

The **{abs(forecast_results['ai']['Cash_On_Hand'].iloc[-1] - forecast_results['traditional']['Cash_On_Hand'].iloc[-1]):,.0f} difference** between traditional and AI methods highlights the value of incorporating machine learning insights into financial planning.

### 💡 **Strategic Recommendations**
- **Immediate Actions**: {'Focus on cost optimization' if analysis['avg_cost_ratio'] > 0.75 else 'Maintain current operational efficiency'}
- **Medium-term**: {'Implement revenue diversification strategies' if analysis['revenue_volatility'] > 0.1 else 'Continue current growth trajectory'}
- **Long-term**: {'Review capital allocation strategy' if analysis['avg_capex_ratio'] > 0.1 else 'Optimize capital expenditure for growth'}

### 🎯 **Key Success Factors**
Your financial performance is driven by:
- Consistent revenue growth patterns
- {'Efficient' if analysis['avg_cost_ratio'] < 0.7 else 'Optimizable'} cost structure
- {'Strong' if analysis['cash_flow_ratio'] > 0.7 else 'Moderate'} cash flow generation
- {'Balanced' if analysis['avg_capex_ratio'] < 0.1 else 'High'} capital investment levels
"""
    
    return narrative

def simulate_openai_analysis(df, forecast_results, analysis):
    """
    Simulate OpenAI API analysis for demo purposes.
    In production, this would call the actual OpenAI API.
    """
    st.info("🤖 **GenAI Integration Note**: This is a simulated AI analysis. In production, this would use OpenAI's GPT-4 API to generate dynamic insights and recommendations.")
    
    # Generate insights and recommendations
    insights = generate_cfo_insights(df, forecast_results, analysis)
    recommendations = generate_cfo_recommendations(df, forecast_results, analysis)
    executive_summary = create_executive_summary(df, forecast_results, analysis)
    ai_narrative = create_ai_narrative(df, forecast_results, analysis)
    
    return {
        'insights': insights,
        'recommendations': recommendations,
        'executive_summary': executive_summary,
        'ai_narrative': ai_narrative
    }
