import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import utility modules (now in same directory)
try:
    from .cfo_synthetic_data import generate_cfo_financial_data, create_financial_dashboard, save_to_csv
    from .cfo_forecasting import traditional_forecasting, ml_forecasting, ai_forecasting_with_shap, create_forecast_comparison
    from .cfo_genai import simulate_openai_analysis, analyze_financial_health
except ImportError:
    # Fallback for direct execution or when imported from page wrapper
    try:
        from cfo_synthetic_data import generate_cfo_financial_data, create_financial_dashboard, save_to_csv
        from cfo_forecasting import traditional_forecasting, ml_forecasting, ai_forecasting_with_shap, create_forecast_comparison
        from cfo_genai import simulate_openai_analysis, analyze_financial_health
    except ImportError:
        # Final fallback: try importing from the current directory
        import os
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from cfo_synthetic_data import generate_cfo_financial_data, create_financial_dashboard, save_to_csv
        from cfo_forecasting import traditional_forecasting, ml_forecasting, ai_forecasting_with_shap, create_forecast_comparison
        from cfo_genai import simulate_openai_analysis, analyze_financial_health

# Page configuration
st.set_page_config(
    page_title="CFO Financial Case Study",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .method-card {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .method-card:hover {
        border-color: #667eea;
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">💰 CFO Financial Case Study</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Comprehensive Financial Analysis: Traditional vs ML vs AI Approaches
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Info section
    if st.session_state.get('show_info', False):
        with st.expander("ℹ️ Module Information & Use Cases", expanded=True):
            st.markdown("""
            ## 🎯 **Use Case: CFO Financial Case Study & Cash Flow Forecasting**
            
            This module addresses the critical challenge of **financial planning and cash flow management** in manufacturing operations, 
            providing AI-powered insights to forecast cash on hand, optimize financial performance, and ensure business sustainability.
            
            ### 🏭 **Target Industries**
            - **Steel Manufacturing**: Capital-intensive operations with complex cash flow cycles
            - **Automotive Assembly**: High-volume production with seasonal cash flow variations
            - **Chemical Processing**: Long-term investment cycles and working capital management
            - **Electronics Manufacturing**: R&D-intensive operations with innovation funding needs
            - **Food Processing**: Seasonal cash flow patterns and inventory management
            - **Textile Manufacturing**: Working capital optimization and seasonal demand
            - **Pharmaceutical Production**: Long development cycles and regulatory compliance costs
            - **Paper Manufacturing**: Capital-intensive operations with environmental compliance
            - **Cement Production**: Infrastructure investment cycles and market demand
            
            ### 🚀 **Key Functionalities**
            
            #### **📈 Traditional Analysis**
            - **Financial Statements**: Revenue, costs, operating expenses, capex analysis
            - **Cash Flow Calculation**: Simple Excel-style cash on hand forecasting
            - **Historical Trends**: Financial performance patterns over time
            - **Summary Statistics**: Key financial metrics and ratios
            - **Basic Forecasting**: Linear trend-based projections
            
            #### **🤖 Machine Learning Prediction**
            - **Cash Flow Forecasting**: Predict cash on hand up to 6 months ahead
            - **Multiple ML Models**: Random Forest regression for financial prediction
            - **Feature Engineering**: Time-based and financial feature creation
            - **Model Performance**: RMSE, MAE, and R² evaluation metrics
            - **Production Correlation**: Financial vs operational performance analysis
            
            #### **🧠 AI-Powered Analysis**
            - **Advanced Forecasting**: Sklearn pipeline with multiple features
            - **SHAP Explainability**: Understand what drives financial performance
            - **Pattern Recognition**: Identify financial trends and anomalies
            - **Feature Importance**: Understand key financial drivers
            - **Predictive Analytics**: Forecast financial performance and risks
            
            #### **🌟 GenAI Integration**
            - **AI-Generated Reports**: Comprehensive financial analysis reports
            - **Financial Health Analysis**: AI-powered financial performance insights
            - **Strategic Recommendations**: AI-generated improvement suggestions
            - **Executive Summaries**: Professional financial reporting
            - **Stakeholder Communication**: Enhanced financial transparency
            
            ### 💰 **Business Value**
            - **Cash Flow Optimization**: 15-25% improvement in cash management
            - **Financial Planning**: Better forecasting for strategic decisions
            - **Risk Management**: Proactive identification of financial risks
            - **Cost Control**: Optimize operating expenses and capital investments
            - **Stakeholder Confidence**: Enhanced financial transparency and reporting
            
            ### 🔧 **Technical Features**
            - **Synthetic Data Generation**: Realistic financial datasets with trends
            - **Interactive Visualizations**: Dynamic charts and financial dashboards
            - **Real-time Analysis**: On-demand financial insights and recommendations
            - **Scalable Architecture**: Support for multiple time periods and scenarios
            - **API Integration Ready**: Prepared for real ERP and financial systems
            
            ### 📊 **Data Sources**
            - **Revenue Data**: Sales, pricing, market demand patterns
            - **Cost Structure**: Operating expenses, cost of goods sold
            - **Capital Expenditure**: Investment in equipment and infrastructure
            - **Operational Metrics**: Production volumes, efficiency indicators
            - **Market Data**: Economic indicators, industry trends
            
            ### 🎯 **User Personas**
            - **CFOs**: Financial strategy, cash flow management, strategic planning
            - **Financial Controllers**: Financial reporting, budget management, analysis
            - **Operations Managers**: Cost optimization, investment planning
            - **Business Leaders**: Strategic financial decisions, risk assessment
            - **Investors**: Financial performance analysis, transparency
            """)
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # Data generation parameters
    st.sidebar.subheader("Data Parameters")
    num_months = st.sidebar.slider("Number of Months", 12, 36, 24)
    random_seed = st.sidebar.number_input("Random Seed", value=42, help="For reproducible results")
    
    # Forecasting parameters
    st.sidebar.subheader("Forecasting Parameters")
    forecast_months = st.sidebar.slider("Forecast Months", 1, 6, 3)
    
    st.sidebar.markdown("---")
    
    # Info button
    if st.sidebar.button("ℹ️ Module Info", type="secondary"):
        st.session_state.show_info = not st.session_state.get('show_info', False)
    
    # Generate or load data
    if st.sidebar.button("🔄 Generate New Dataset") or 'cfo_data' not in st.session_state:
        with st.spinner("Generating synthetic CFO financial data..."):
            st.session_state.cfo_data = generate_cfo_financial_data(months=num_months, seed=random_seed)
        st.success(f"✅ Generated {num_months} months of synthetic financial data!")
    
    # Check if data exists
    if 'cfo_data' not in st.session_state:
        st.warning("Please generate data first using the sidebar button.")
        return
    
    df = st.session_state.cfo_data
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Overview", "🔮 Traditional Forecasting", "🤖 ML Forecasting", 
        "🧠 AI Forecasting", "📈 Comparison", "🤖 GenAI Insights"
    ])
    
    with tab1:
        st.header("📊 Financial Data Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Total Revenue</h3>
                <h2>${:,.0f}</h2>
            </div>
            """.format(df['Revenue'].sum()), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Avg Monthly Revenue</h3>
                <h2>${:,.0f}</h2>
            </div>
            """.format(df['Revenue'].mean()), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>Gross Margin</h3>
                <h2>{:.1f}%</h2>
            </div>
            """.format(((df['Revenue'] - df['Cost']) / df['Revenue']).mean() * 100), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>Ending Cash</h3>
                <h2>${:,.0f}</h2>
            </div>
            """.format(df['Cash_On_Hand'].iloc[-1]), unsafe_allow_html=True)
        
        # Data preview
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Financial dashboard
        st.subheader("📈 Interactive Financial Dashboard")
        dashboard_fig = create_financial_dashboard(df)
        st.plotly_chart(dashboard_fig, use_container_width=True)
        
        # Download data
        st.subheader("💾 Download Data")
        csv_data, filename = save_to_csv(df, f'cfo_financial_data_{num_months}months.csv')
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=filename,
            mime="text/csv"
        )
    
    with tab2:
        st.header("🔮 Traditional Forecasting (Excel-Style)")
        
        st.markdown("""
        <div class="method-card">
            <h3>📊 Traditional Approach</h3>
            <p>Simple Excel-style calculations using historical ratios and growth assumptions.</p>
            <ul>
                <li>Maintains historical cost ratios</li>
                <li>Applies simple growth rates</li>
                <li>Basic cash flow calculations</li>
                <li>No machine learning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Run Traditional Forecast"):
            with st.spinner("Running traditional forecasting..."):
                df_traditional = traditional_forecasting(df, forecast_months)
                st.session_state.df_traditional = df_traditional
                
                st.success("✅ Traditional forecasting completed!")
                
                # Display results
                st.subheader("📊 Traditional Forecast Results")
                st.dataframe(df_traditional, use_container_width=True)
                
                # Show calculation logic
                st.subheader("🧮 Calculation Logic")
                st.markdown("""
                **Cash On Hand = Previous Cash + (Revenue - Cost - Operating Expenses - Capex)**
                
                - Revenue: Previous month × (1 + growth rate)ⁿ
                - Cost: Revenue × historical cost ratio
                - Operating Expenses: Revenue × historical opex ratio
                - Capex: Revenue × historical capex ratio
                """)
        
        if 'df_traditional' in st.session_state:
            st.subheader("📈 Traditional Forecast Visualization")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Month'], y=df['Cash_On_Hand'], 
                                   name='Historical', mode='lines+markers'))
            fig.add_trace(go.Scatter(x=st.session_state.df_traditional['Month'], 
                                   y=st.session_state.df_traditional['Cash_On_Hand'], 
                                   name='Traditional Forecast', mode='lines+markers'))
            fig.update_layout(title="Traditional Forecasting Results", height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("🤖 Machine Learning Forecasting")
        
        st.markdown("""
        <div class="method-card">
            <h3>🤖 ML Approach</h3>
            <p>Random Forest regression using engineered features and historical patterns.</p>
            <ul>
                <li>Lag features (previous month values)</li>
                <li>Seasonal indicators</li>
                <li>Machine learning model training</li>
                <li>Performance metrics evaluation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Run ML Forecast"):
            with st.spinner("Training ML model and generating forecast..."):
                df_ml, rf_model, feature_cols, metrics = ml_forecasting(df, forecast_months)
                st.session_state.df_ml = df_ml
                st.session_state.rf_model = rf_model
                st.session_state.ml_metrics = metrics
                
                st.success("✅ ML forecasting completed!")
                
                # Display results
                st.subheader("📊 ML Forecast Results")
                st.dataframe(df_ml, use_container_width=True)
                
                # Show model performance
                st.subheader("📈 Model Performance")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Absolute Error", f"${metrics['MAE']:,.0f}")
                with col2:
                    st.metric("R² Score", f"{metrics['R2']:.3f}")
                
                # Show features used
                st.subheader("🔍 Features Used")
                st.write(f"**{len(feature_cols)} features**: {', '.join(feature_cols)}")
        
        if 'df_ml' in st.session_state:
            st.subheader("📈 ML Forecast Visualization")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Month'], y=df['Cash_On_Hand'], 
                                   name='Historical', mode='lines+markers'))
            fig.add_trace(go.Scatter(x=st.session_state.df_ml['Month'], 
                                   y=st.session_state.df_ml['Cash_On_Hand'], 
                                   name='ML Forecast', mode='lines+markers'))
            fig.update_layout(title="ML Forecasting Results", height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("🧠 AI Forecasting with SHAP Explainability")
        
        st.markdown("""
        <div class="method-card">
            <h3>🧠 AI Pipeline Approach</h3>
            <p>Advanced sklearn pipeline with feature engineering, scaling, and SHAP explainability.</p>
            <ul>
                <li>Advanced feature engineering</li>
                <li>Pipeline with scaling</li>
                <li>SHAP explainability plots</li>
                <li>Comprehensive feature analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Run AI Forecast"):
            with st.spinner("Running AI pipeline with SHAP analysis..."):
                try:
                    df_ai, pipeline, feature_cols, metrics, shap_values, X_test, feature_importance = ai_forecasting_with_shap(df, forecast_months)
                    st.session_state.df_ai = df_ai
                    st.session_state.pipeline = pipeline
                    st.session_state.ai_metrics = metrics
                    st.session_state.shap_values = shap_values
                    st.session_state.X_test = X_test
                    st.session_state.feature_importance = feature_importance
                except ValueError as e:
                    if "not enough values to unpack" in str(e):
                        st.error("❌ Error in AI forecasting: Function returned fewer values than expected. This might be due to a compatibility issue.")
                        st.info("💡 Try running the ML forecast first, or check if all required libraries are installed.")
                        return
                    else:
                        st.error(f"❌ Error in AI forecasting: {str(e)}")
                        return
                except Exception as e:
                    st.error(f"❌ Unexpected error in AI forecasting: {str(e)}")
                    return
                
                st.success("✅ AI forecasting completed!")
                
                # Display results
                st.subheader("📊 AI Forecast Results")
                st.dataframe(df_ai, use_container_width=True)
                
                # Show model performance
                st.subheader("📈 AI Model Performance")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Absolute Error", f"${metrics['MAE']:,.0f}")
                with col2:
                    st.metric("R² Score", f"{metrics['R2']:.3f}")
                
                # Show features used
                st.subheader("🔍 AI Features Used")
                st.write(f"**{len(feature_cols)} advanced features**: {', '.join(feature_cols[:10])}...")
                
                # Show feature importance or SHAP info
                if st.session_state.get('feature_importance') is not None:
                    st.subheader("📊 Feature Importance Analysis")
                    importance_df = pd.DataFrame([
                        {'Feature': feature, 'Importance': importance}
                        for feature, importance in st.session_state.feature_importance.items()
                    ]).sort_values('Importance', ascending=False)
                    
                    # Create feature importance bar chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=importance_df['Feature'],
                            y=importance_df['Importance'],
                            marker_color='lightblue'
                        )
                    ])
                    fig.update_layout(
                        title="Feature Importance for Cash On Hand Prediction",
                        xaxis_title="Features",
                        yaxis_title="Importance Score",
                        height=400,
                        showlegend=False
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show top features table
                    st.subheader("🏆 Top 10 Most Important Features")
                    st.dataframe(importance_df.head(10), use_container_width=True)
                    
                elif st.session_state.get('shap_values') is not None:
                    st.info("✅ SHAP explainability available for detailed feature analysis")
                else:
                    st.warning("⚠️ Feature importance analysis not available")
        
        if 'df_ai' in st.session_state:
            st.subheader("📈 AI Forecast Visualization")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Month'], y=df['Cash_On_Hand'], 
                                   name='Historical', mode='lines+markers'))
            fig.add_trace(go.Scatter(x=st.session_state.df_ai['Month'], 
                                   y=st.session_state.df_ai['Cash_On_Hand'], 
                                   name='AI Forecast', mode='lines+markers'))
            fig.update_layout(title="AI Forecasting Results", height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("📈 Forecasting Method Comparison")
        
        if all(key in st.session_state for key in ['df_traditional', 'df_ml', 'df_ai']):
            st.subheader("🔄 Method Comparison")
            
            # Create comparison table
            comparison_data = []
            for i in range(forecast_months):
                comparison_data.append({
                    'Forecast Month': f'Month {i+1}',
                    'Traditional': f"${st.session_state.df_traditional.iloc[i]['Cash_On_Hand']:,.0f}",
                    'ML': f"${st.session_state.df_ml.iloc[i]['Cash_On_Hand']:,.0f}",
                    'AI': f"${st.session_state.df_ai.iloc[i]['Cash_On_Hand']:,.0f}",
                    'ML vs Traditional': f"${st.session_state.df_ml.iloc[i]['Cash_On_Hand'] - st.session_state.df_traditional.iloc[i]['Cash_On_Hand']:,.0f}",
                    'AI vs Traditional': f"${st.session_state.df_ai.iloc[i]['Cash_On_Hand'] - st.session_state.df_traditional.iloc[i]['Cash_On_Hand']:,.0f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Create comparison visualization
            st.subheader("📊 Visual Comparison")
            fig = create_forecast_comparison(df, st.session_state.df_traditional, 
                                          st.session_state.df_ml, st.session_state.df_ai)
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance comparison
            st.subheader("📈 Performance Metrics Comparison")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Traditional", "Baseline", delta=None)
            
            with col2:
                ml_improvement = st.session_state.ml_metrics['R2'] - 0.5  # Assume baseline R² = 0.5
                st.metric("ML Model", f"R²: {st.session_state.ml_metrics['R2']:.3f}", 
                         delta=f"{ml_improvement:.3f}")
            
            with col3:
                ai_improvement = st.session_state.ai_metrics['R2'] - 0.5
                st.metric("AI Pipeline", f"R²: {st.session_state.ai_metrics['R2']:.3f}", 
                         delta=f"{ai_improvement:.3f}")
        
        else:
            st.warning("⚠️ Please run all forecasting methods first to see comparison.")
    
    with tab6:
        st.header("🤖 GenAI Financial Insights")
        
        st.markdown("""
        <div class="method-card">
            <h3>🤖 Generative AI Analysis</h3>
            <p>AI-powered financial analysis, insights, and strategic recommendations.</p>
            <ul>
                <li>Financial health assessment</li>
                <li>Strategic insights</li>
                <li>Actionable recommendations</li>
                <li>Executive summary</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Generate AI Insights"):
            if all(key in st.session_state for key in ['df_traditional', 'df_ml', 'df_ai']):
                with st.spinner("Generating AI insights and recommendations..."):
                    # Prepare forecast results
                    forecast_results = {
                        'traditional': st.session_state.df_traditional,
                        'ml': st.session_state.df_ml,
                        'ai': st.session_state.df_ai
                    }
                    
                    # Analyze financial health
                    analysis = analyze_financial_health(df)
                    
                    # Generate AI insights
                    ai_results = simulate_openai_analysis(df, forecast_results, analysis)
                    
                    st.session_state.ai_results = ai_results
                    st.success("✅ AI insights generated!")
                    
                    # Display insights
                    st.subheader("💡 Key Insights")
                    for insight in ai_results['insights']:
                        st.markdown(f"- {insight}")
                    
                    # Display recommendations
                    st.subheader("🚀 Strategic Recommendations")
                    for rec in ai_results['recommendations']:
                        st.markdown(f"- {rec}")
                    
                    # Display executive summary
                    st.subheader("📊 Executive Summary")
                    st.markdown(ai_results['executive_summary'])
                    
                    # Display AI narrative
                    st.subheader("🤖 AI-Generated Narrative")
                    st.markdown(ai_results['ai_narrative'])
            else:
                st.warning("⚠️ Please run all forecasting methods first to generate AI insights.")
        
        if 'ai_results' in st.session_state:
            st.subheader("📊 AI Analysis Results")
            
            # Display insights
            st.markdown("### 💡 Key Insights")
            for insight in st.session_state.ai_results['insights']:
                st.markdown(f"- {insight}")
            
            # Display recommendations
            st.markdown("### 🚀 Strategic Recommendations")
            for rec in st.session_state.ai_results['recommendations']:
                st.markdown(f"- {rec}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>💰 CFO Financial Case Study | Traditional vs ML vs AI Approaches</p>
        <p>Built with Streamlit, Python, and Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
