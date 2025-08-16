import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import utility modules
from hr_synthetic_data import generate_synthetic_hr_data, create_hr_dashboard, create_attrition_heatmap, save_to_csv
from hr_ml_prediction import (prepare_features_for_ml, train_logistic_regression, train_random_forest,
                             create_model_comparison, create_feature_importance_comparison,
                             create_confusion_matrix_heatmap, predict_attrition_for_employee,
                             generate_retention_recommendations)
from hr_ai_agents import MultiAgentSystem
from hr_genai import simulate_openai_analysis, create_hr_chatbot_interface

# Page configuration
st.set_page_config(
    page_title="CHRO Attrition Prediction",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .risk-high { color: #d62728; font-weight: bold; }
    .risk-medium { color: #ff7f0e; font-weight: bold; }
    .risk-low { color: #2ca02c; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main function for the CHRO Attrition Prediction app."""
    
    # Header
    st.title("👥 CHRO Attrition Prediction System")
    st.markdown("### AI-Powered Workforce Retention Analysis & Prediction")
    
    # Info section
    if st.session_state.get('show_info', False):
        with st.expander("ℹ️ Module Information & Use Cases", expanded=True):
            st.markdown("""
            ## 🎯 **Use Case: CHRO Attrition Prediction & Workforce Retention**
            
            This module addresses the critical challenge of **employee retention and workforce planning** in manufacturing operations, 
            providing AI-powered insights to predict attrition, understand employee satisfaction, and implement retention strategies.
            
            ### 🏭 **Target Industries**
            - **Steel Manufacturing**: High-skill workforce retention and safety management
            - **Automotive Assembly**: Production line employee satisfaction and retention
            - **Chemical Processing**: Technical expertise retention and safety compliance
            - **Electronics Manufacturing**: Engineering talent retention and innovation
            - **Food Processing**: Operational staff retention and quality focus
            - **Textile Manufacturing**: Skilled labor retention and productivity
            - **Pharmaceutical Production**: Research talent retention and compliance
            - **Paper Manufacturing**: Technical expertise retention and efficiency
            - **Cement Production**: Safety-focused workforce retention
            
            ### 🚀 **Key Functionalities**
            
            #### **📈 Traditional Analysis**
            - **Employee Demographics**: Age, department, job role distribution
            - **Attrition Patterns**: Historical attrition rates and trends
            - **Summary Statistics**: Comprehensive workforce metrics
            - **Department Analysis**: Performance across organizational units
            - **Tenure Analysis**: Employee retention patterns over time
            
            #### **🤖 Machine Learning Prediction**
            - **Attrition Prediction**: Forecast employee departure likelihood
            - **Multiple ML Models**: Logistic Regression and Random Forest classifiers
            - **Feature Importance**: Understand what drives employee retention
            - **Risk Scoring**: Individual employee attrition risk assessment
            - **Model Performance**: Evaluate and optimize prediction accuracy
            
            #### **🧠 AI-Powered Analysis**
            - **Multi-Agent System**: Coordinated AI agents for comprehensive analysis
            - **HR Metrics Agent**: Quantitative workforce performance analysis
            - **Employee Feedback Agent**: Sentiment analysis and qualitative insights
            - **Pattern Recognition**: Identify retention risk factors
            - **Predictive Analytics**: Forecast workforce trends and needs
            
            #### **🌟 GenAI Integration**
            - **AI-Generated Insights**: Comprehensive workforce analysis reports
            - **Retention Strategies**: AI-powered improvement recommendations
            - **Feedback Summarization**: Automated employee feedback analysis
            - **Strategic Planning**: Data-driven workforce planning insights
            - **Stakeholder Communication**: Professional HR reporting and insights
            
            ### 💰 **Business Value**
            - **Cost Reduction**: 20-30% reduction in recruitment and training costs
            - **Productivity Improvement**: Higher retention leads to better performance
            - **Knowledge Retention**: Preserve institutional knowledge and expertise
            - **Safety Enhancement**: Retain experienced safety-conscious employees
            - **Competitive Advantage**: Stable workforce for consistent operations
            
            ### 🔧 **Technical Features**
            - **Synthetic Data Generation**: Realistic HR datasets with attrition patterns
            - **Interactive Visualizations**: Dynamic charts and dashboards
            - **Real-time Analysis**: On-demand insights and recommendations
            - **Scalable Architecture**: Support for large workforce datasets
            - **API Integration Ready**: Prepared for real HRIS and ATS systems
            
            ### 📊 **Data Sources**
            - **Employee Demographics**: Age, gender, department, job role
            - **Performance Metrics**: Performance ratings, satisfaction scores
            - **Employment History**: Tenure, promotions, training hours
            - **Behavioral Data**: Work patterns, engagement indicators
            - **Feedback Data**: Employee surveys, exit interviews, sentiment
            
            ### 🎯 **User Personas**
            - **CHROs**: Workforce strategy, retention planning, organizational development
            - **HR Managers**: Employee relations, performance management, retention programs
            - **Operations Managers**: Workforce planning, productivity optimization
            - **Business Leaders**: Strategic workforce decisions, cost management
            - **Safety Managers**: Workforce stability and safety performance
            """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        num_employees = st.slider("Number of Employees", min_value=500, max_value=2000, value=1000, step=100)
        random_seed = st.number_input("Random Seed", min_value=1, max_value=1000, value=42, step=1)
        
        if st.button("🔄 Generate New Dataset", type="primary"):
            st.session_state.hr_data = None
            st.rerun()
        
        st.markdown("---")
        
        # Info button
        if st.button("ℹ️ Module Info", type="secondary"):
            st.session_state.show_info = not st.session_state.get('show_info', False)
    
    # Initialize session state
    if 'hr_data' not in st.session_state:
        st.session_state.hr_data = None
    if 'ml_models' not in st.session_state:
        st.session_state.ml_models = None
    if 'ai_insights' not in st.session_state:
        st.session_state.ai_insights = None
    if 'multi_agent' not in st.session_state:
        st.session_state.multi_agent = None
    if 'genai_results' not in st.session_state:
        st.session_state.genai_results = None
    
    # Generate or load data
    if st.session_state.hr_data is None:
        with st.spinner("🔄 Generating synthetic HR dataset..."):
            st.session_state.hr_data = generate_synthetic_hr_data(num_employees, random_seed)
        st.success(f"✅ Generated dataset with {len(st.session_state.hr_data)} employees")
    
    df = st.session_state.hr_data
    
    # Key metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Employees", f"{len(df):,}")
    
    with col2:
        attrition_rate = df['Attrition'].mean() * 100
        st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
    
    with col3:
        avg_satisfaction = df['SatisfactionScore'].mean()
        st.metric("Avg Satisfaction", f"{avg_satisfaction:.1f}/10")
    
    with col4:
        avg_performance = df['PerformanceRating'].mean()
        st.metric("Avg Performance", f"{avg_performance:.1f}/5")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Traditional Analysis", 
        "🤖 ML Prediction", 
        "🧠 AI Agents", 
        "🌟 GenAI Insights"
    ])
    
    # Tab 1: Traditional Analysis
    with tab1:
        st.header("📈 Traditional HR Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 HR Attrition Dashboard")
            dashboard_fig = create_hr_dashboard(df)
            st.plotly_chart(dashboard_fig, use_container_width=True)
        
        with col2:
            st.subheader("📋 Summary Statistics")
            dept_stats = df.groupby('Department').agg({
                'Attrition': ['count', 'sum', 'mean'],
                'Salary': 'mean',
                'SatisfactionScore': 'mean'
            }).round(2)
            
            dept_stats.columns = ['Count', 'Left', 'Attrition_Rate', 'Avg_Salary', 'Avg_Satisfaction']
            dept_stats['Attrition_Rate'] = dept_stats['Attrition_Rate'] * 100
            
            st.dataframe(dept_stats, use_container_width=True)
            
            # Download data
            csv_data, filename = save_to_csv(df)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv"
            )
        
        # Correlation heatmap
        st.subheader("🔥 Attrition Correlation Analysis")
        heatmap_fig = create_attrition_heatmap(df)
        st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # Tab 2: ML Prediction
    with tab2:
        st.header("🤖 Machine Learning Prediction")
        
        if st.button("🚀 Train ML Models", type="primary"):
            with st.spinner("Training machine learning models..."):
                # Prepare features
                X_train, X_test, y_train, y_test, feature_cols, scaler, le_dept, le_job = prepare_features_for_ml(df)
                
                # Train models
                lr_model, lr_pred, lr_proba, lr_acc, lr_roc, lr_importance = train_logistic_regression(X_train, X_test, y_train, y_test)
                rf_model, rf_pred, rf_proba, rf_acc, rf_roc, rf_importance = train_random_forest(X_train, X_test, y_train, y_test)
                
                # Store results
                st.session_state.ml_models = {
                    'lr_model': lr_model, 'rf_model': rf_model,
                    'scaler': scaler, 'le_dept': le_dept, 'le_job': le_job,
                    'X_test': X_test, 'y_test': y_test,
                    'lr_results': {'accuracy': lr_acc, 'roc_auc': lr_roc, 'predictions': lr_pred},
                    'rf_results': {'accuracy': rf_acc, 'roc_auc': rf_roc, 'predictions': rf_pred},
                    'lr_importance': lr_importance, 'rf_importance': rf_importance
                }
                st.success("✅ Models trained successfully!")
                st.rerun()
        
        if 'ml_models' in st.session_state and st.session_state.ml_models is not None:
            models = st.session_state.ml_models
            
            # Model comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Model Performance Comparison")
                comparison_fig = create_model_comparison(models['lr_results'], models['rf_results'])
                st.plotly_chart(comparison_fig, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Model Metrics")
                st.markdown("**Logistic Regression:**")
                st.metric("Accuracy", f"{models['lr_results']['accuracy']:.3f}")
                st.metric("ROC AUC", f"{models['rf_results']['roc_auc']:.3f}")
                
                st.markdown("**Random Forest:**")
                st.metric("Accuracy", f"{models['rf_results']['accuracy']:.3f}")
                st.metric("ROC AUC", f"{models['rf_results']['roc_auc']:.3f}")
            
            # Feature importance
            st.subheader("🔍 Feature Importance Analysis")
            importance_fig = create_feature_importance_comparison(models['lr_importance'], models['rf_importance'])
            st.plotly_chart(importance_fig, use_container_width=True)
        else:
            st.info("🚀 Click the 'Train ML Models' button above to start training machine learning models for attrition prediction.")
    
    # Tab 3: AI Agents
    with tab3:
        st.header("🧠 Multi-Agent AI System")
        
        if st.button("🤖 Run AI Agent Analysis", type="primary"):
            with st.spinner("Running multi-agent AI analysis..."):
                multi_agent = MultiAgentSystem()
                insights = multi_agent.run_comprehensive_analysis(df)
                st.session_state.ai_insights = insights
                st.session_state.multi_agent = multi_agent
                st.success("✅ AI agent analysis completed!")
                st.rerun()
        
        if 'ai_insights' in st.session_state and 'multi_agent' in st.session_state and st.session_state.ai_insights is not None:
            insights = st.session_state.ai_insights
            multi_agent = st.session_state.multi_agent
            
            # Agent insights dashboard
            st.subheader("📊 Multi-Agent Insights Dashboard")
            agent_dashboard = multi_agent.create_agent_insights_dashboard(insights)
            st.plotly_chart(agent_dashboard, use_container_width=True)
            
            # Metrics insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔍 HR Metrics Agent Insights")
                for insight in insights['metrics_insights']:
                    st.markdown(f"• {insight}")
            
            with col2:
                st.subheader("💬 Feedback Agent Patterns")
                for pattern in insights['feedback_patterns']:
                    st.markdown(f"• {pattern}")
        else:
            st.info("🤖 Click the 'Run AI Agent Analysis' button above to start the multi-agent AI analysis.")
    
    # Tab 4: GenAI Insights
    with tab4:
        st.header("🌟 Generative AI Insights")
        
        if st.button("🌟 Generate AI Insights", type="primary"):
            with st.spinner("Generating AI-powered insights..."):
                genai_results = simulate_openai_analysis(df, {})
                st.session_state.genai_results = genai_results
                st.success("✅ AI insights generated!")
                st.rerun()
        
        if 'genai_results' in st.session_state and st.session_state.genai_results is not None:
            results = st.session_state.genai_results
            
            # Feedback summary
            st.subheader("📊 AI-Generated Feedback Summary")
            st.markdown(results['feedback_summary'])
            
            # Retention strategies
            st.subheader("🚀 AI-Generated Retention Strategies")
            for strategy in results['retention_strategies']:
                st.markdown(strategy)
            
            # HR Chatbot
            st.markdown("---")
            create_hr_chatbot_interface()
        else:
            st.info("🌟 Click the 'Generate AI Insights' button above to start the GenAI analysis and get AI-powered insights.")

if __name__ == "__main__":
    main()
