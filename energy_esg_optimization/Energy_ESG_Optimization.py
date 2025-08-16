"""
Energy Optimization & ESG Compliance Streamlit Application
Manufacturing Energy Management with AI-Powered Optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Import custom modules
from energy_esg_optimization.data.energy_synthetic_data import generate_sample_data
from energy_esg_optimization.models.ml_forecasting import EnergyConsumptionForecaster
from energy_esg_optimization.ai.optimization_engine import EnergyOptimizationEngine
from energy_esg_optimization.genai.esg_report_generator import ESGReportGenerator

# Page configuration
st.set_page_config(
    page_title="Energy Optimization & ESG Compliance",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main function for the Energy Optimization & ESG Compliance app."""
    
    # Header
    st.title("⚡ Energy Optimization & ESG Compliance")
    st.markdown("### AI-Powered Manufacturing Energy Management & Sustainability Reporting")
    
    # Info section
    if st.session_state.get('show_info', False):
        with st.expander("ℹ️ Module Information & Use Cases", expanded=True):
            st.markdown("""
            ## 🎯 **Use Case: Energy Optimization & ESG Compliance**
            
            This module addresses the critical challenge of **energy management and sustainability reporting** in manufacturing operations, 
            providing AI-powered insights to optimize energy consumption, reduce costs, and ensure ESG compliance.
            
            ### 🏭 **Target Industries**
            - **Steel Manufacturing**: High-energy operations with significant optimization potential
            - **Automotive Assembly**: Production line energy efficiency and sustainability
            - **Chemical Processing**: Energy-intensive processes and emissions management
            - **Electronics Manufacturing**: Clean energy integration and efficiency
            - **Food Processing**: Sustainable energy practices and compliance
            - **Textile Manufacturing**: Energy optimization and environmental impact
            - **Pharmaceutical Production**: Compliance-focused energy management
            - **Paper Manufacturing**: Renewable energy integration and efficiency
            - **Cement Production**: High-emission operations requiring optimization
            
            ### 🚀 **Key Functionalities**
            
            #### **📈 Traditional Analysis**
            - **Energy Consumption Monitoring**: Track daily energy usage across plants
            - **CO2 Emissions Tracking**: Monitor environmental impact and compliance
            - **Plant Performance Comparison**: Benchmark efficiency across facilities
            - **Compliance Status**: ISO certifications and ESG ratings tracking
            - **Trend Analysis**: Historical energy patterns and seasonal variations
            
            #### **🤖 Machine Learning Forecasting**
            - **Energy Consumption Prediction**: Forecast usage up to 90 days ahead
            - **Multiple ML Models**: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting
            - **Feature Importance Analysis**: Understand what drives energy consumption
            - **Production Correlation**: Energy vs production relationship analysis
            - **Model Performance Comparison**: Evaluate and select best forecasting models
            
            #### **🧠 AI-Powered Optimization**
            - **Pattern Recognition**: Identify peak demand, efficiency trends, seasonal patterns
            - **Optimization Opportunities**: Rule-based recommendations for improvement
            - **Reinforcement Learning**: Simulated optimization environment
            - **Cost-Benefit Analysis**: ROI estimates and implementation timelines
            - **Plant Performance Benchmarking**: Compare and rank facility performance
            
            #### **🌟 GenAI ESG Reporting**
            - **AI-Generated Reports**: Comprehensive sustainability reporting
            - **ESG Performance Dashboard**: Visual metrics and compliance tracking
            - **Intelligent Chatbot**: ESG-related question answering
            - **Strategic Recommendations**: AI-powered improvement suggestions
            - **Compliance Frameworks**: GRI, SASB, TCFD, CDP standards support
            
            ### 💰 **Business Value**
            - **Cost Reduction**: 15-25% energy cost savings potential
            - **Efficiency Improvement**: 10-20% energy consumption reduction
            - **Compliance Assurance**: Automated ESG reporting and monitoring
            - **Risk Management**: Proactive identification of optimization opportunities
            - **Sustainability Leadership**: Enhanced ESG ratings and reputation
            
            ### 🔧 **Technical Features**
            - **Synthetic Data Generation**: Realistic energy and ESG datasets
            - **Interactive Visualizations**: Dynamic charts and dashboards
            - **Real-time Analysis**: On-demand insights and recommendations
            - **Scalable Architecture**: Support for multiple plants and time periods
            - **API Integration Ready**: Prepared for real energy management systems
            
            ### 📊 **Data Sources**
            - **Energy Consumption**: kWh usage, peak demand, efficiency metrics
            - **Environmental Impact**: CO2 emissions, renewable energy percentage
            - **Production Data**: Units produced, downtime, quality scores
            - **Compliance Metrics**: ISO scores, ESG ratings, audit status
            - **Operational Data**: Maintenance hours, production efficiency
            
            ### 🎯 **User Personas**
            - **Energy Managers**: Monitor consumption, optimize efficiency, reduce costs
            - **Sustainability Teams**: ESG reporting, compliance monitoring, stakeholder communication
            - **Operations Managers**: Production optimization, maintenance planning, performance benchmarking
            - **Business Leaders**: Strategic planning, ROI analysis, risk management
            - **Compliance Officers**: Regulatory adherence, audit preparation, reporting automation
            """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data parameters
        st.subheader("📊 Data Parameters")
        num_days = st.slider("Number of Days", min_value=90, max_value=365, value=365, step=30)
        num_plants = st.slider("Number of Plants", min_value=3, max_value=10, value=5, step=1)
        random_seed = st.number_input("Random Seed", min_value=1, max_value=1000, value=42, step=1)
        
        # Analysis approach selection
        st.subheader("🔍 Analysis Approach")
        approach = st.selectbox(
            "Select Analysis Approach",
            ["Traditional", "ML", "AI", "GenAI"],
            help="Choose the analysis method to explore"
        )
        
        # Generate data button
        if st.button("🔄 Generate New Dataset", type="primary"):
            st.session_state.energy_data = None
            st.session_state.production_data = None
            st.session_state.compliance_data = None
            st.rerun()
        
        st.markdown("---")
        
        # Info button
        if st.button("ℹ️ Module Info", type="secondary"):
            st.session_state.show_info = not st.session_state.get('show_info', False)
        
        # App information
        st.subheader("ℹ️ About")
        st.markdown("""
        This application demonstrates four approaches to energy optimization:
        
        **📈 Traditional**: Static reports and basic analytics
        **🤖 ML**: Machine learning for energy consumption forecasting
        **🧠 AI**: AI-powered optimization and reinforcement learning
        **🌟 GenAI**: AI-generated ESG reports and chatbot assistance
        
        **Data**: Synthetic manufacturing energy data with realistic patterns
        """)
    
    # Initialize session state
    if 'energy_data' not in st.session_state:
        st.session_state.energy_data = None
    if 'production_data' not in st.session_state:
        st.session_state.production_data = None
    if 'compliance_data' not in st.session_state:
        st.session_state.compliance_data = None
    if 'data_summary' not in st.session_state:
        st.session_state.data_summary = None
    if 'ml_models' not in st.session_state:
        st.session_state.ml_models = None
    if 'ai_optimization' not in st.session_state:
        st.session_state.ai_optimization = None
    if 'esg_report' not in st.session_state:
        st.session_state.esg_report = None
    
    # Generate or load data
    if st.session_state.energy_data is None:
        with st.spinner("🔄 Generating synthetic energy and ESG dataset..."):
            energy_df, production_df, compliance_df, summary = generate_sample_data(num_days, num_plants)
            st.session_state.energy_data = energy_df
            st.session_state.production_data = production_df
            st.session_state.compliance_data = compliance_df
            st.session_state.data_summary = summary
        st.success(f"✅ Generated dataset with {len(energy_df)} energy records for {num_plants} plants")
    
    energy_df = st.session_state.energy_data
    production_df = st.session_state.production_data
    compliance_df = st.session_state.compliance_data
    summary = st.session_state.data_summary
    

    
    # Key metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            st.metric("Total Energy Consumption", f"{summary.get('total_energy_consumption', 0):,.0f} kWh")
        except (KeyError, TypeError):
            st.metric("Total Energy Consumption", "N/A")
    
    with col2:
        try:
            st.metric("Total CO2 Emissions", f"{summary.get('total_emissions', 0):,.0f} kg")
        except (KeyError, TypeError):
            st.metric("Total CO2 Emissions", "N/A")
    
    with col3:
        try:
            st.metric("Average Energy Efficiency", f"{summary.get('avg_energy_efficiency', 0):.3f}")
        except (KeyError, TypeError):
            st.metric("Average Energy Efficiency", "N/A")
    
    with col4:
        try:
            st.metric("Renewable Energy", f"{summary.get('avg_renewable_percentage', 0):.1f}%")
        except (KeyError, TypeError):
            st.metric("Renewable Energy", "N/A")
    
    # Main content based on selected approach
    if approach == "Traditional":
        display_traditional_analysis(energy_df, production_df, compliance_df, summary)
    elif approach == "ML":
        display_ml_analysis(energy_df, production_df, compliance_df)
    elif approach == "AI":
        display_ai_analysis(energy_df, production_df, compliance_df)
    elif approach == "GenAI":
        display_genai_analysis(energy_df, production_df, compliance_df)

def display_traditional_analysis(energy_df, production_df, compliance_df, summary):
    """Display traditional analysis with static reports and basic charts."""
    
    st.header("📈 Traditional Analysis")
    st.markdown("Static reports and basic analytics for energy and ESG performance.")
    
    # Create tabs for different analysis types
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary Statistics", "📈 Energy Trends", "🏭 Plant Comparison", "📋 Compliance Status"])
    
    with tab1:
        st.subheader("Summary Statistics")
        
        # Create summary table
        summary_data = {
            'Metric': [
                'Total Energy Consumption (kWh)',
                'Total CO2 Emissions (kg)',
                'Average Energy Efficiency',
                'Average Renewable Energy (%)',
                'Total Production (units)',
                'Average Production Efficiency (%)',
                'Average Compliance Score',
                'Total Energy Cost (USD)'
            ],
            'Value': [
                f"{summary.get('total_energy_consumption', 0):,.0f}",
                f"{summary.get('total_emissions', 0):,.0f}",
                f"{summary.get('avg_energy_efficiency', 0):.3f}",
                f"{summary.get('avg_renewable_percentage', 0):.1f}%",
                f"{summary.get('total_production', 0):,.0f}",
                f"{summary.get('avg_production_efficiency', 0):.1f}%",
                f"{summary.get('avg_compliance_score', 0):.1f}",
                f"${summary.get('total_energy_cost', 0):,.2f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)
        
        # Additional insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Energy Cost Savings Potential", f"${summary.get('energy_cost_savings_potential', 0):,.2f}")
            st.metric("Emissions Reduction Potential", f"{summary.get('emissions_reduction_potential', 0):,.0f} kg")
        
        with col2:
            st.metric("Certified Plants", f"{summary.get('certified_plants', 0):,}")
            st.metric("Total Plants", f"{summary.get('total_plants', 0):,}")
    
    with tab2:
        st.subheader("Energy Consumption Trends")
        
        # Energy consumption over time
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Energy consumption by date
        energy_df['Date'] = pd.to_datetime(energy_df['Date'])
        daily_energy = energy_df.groupby('Date')['EnergyConsumption_kWh'].sum()
        
        ax1.plot(daily_energy.index, daily_energy.values, linewidth=2, color='blue')
        ax1.set_title('Daily Energy Consumption Trend')
        ax1.set_ylabel('Energy Consumption (kWh)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: CO2 emissions by date
        daily_emissions = energy_df.groupby('Date')['CO2Emissions_kg'].sum()
        
        ax2.plot(daily_emissions.index, daily_emissions.values, linewidth=2, color='red')
        ax2.set_title('Daily CO2 Emissions Trend')
        ax2.set_ylabel('CO2 Emissions (kg)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Energy vs Production correlation
        st.subheader("Energy vs Production Correlation")
        
        # Merge energy and production data
        merged_df = energy_df.merge(production_df, on=['Date', 'Plant', 'PlantType'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(merged_df['EnergyConsumption_kWh'], merged_df['Production_Units'], alpha=0.6)
        ax.set_xlabel('Energy Consumption (kWh)')
        ax.set_ylabel('Production (Units)')
        ax.set_title('Energy Consumption vs Production Output')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(merged_df['EnergyConsumption_kWh'], merged_df['Production_Units'], 1)
        p = np.poly1d(z)
        ax.plot(merged_df['EnergyConsumption_kWh'], p(merged_df['EnergyConsumption_kWh']), "r--", alpha=0.8)
        
        st.pyplot(fig)
        
        # Calculate correlation
        correlation = merged_df['EnergyConsumption_kWh'].corr(merged_df['Production_Units'])
        st.metric("Correlation Coefficient", f"{correlation:.3f}")
    
    with tab3:
        st.subheader("Plant Performance Comparison")
        
        # Plant comparison metrics
        plant_metrics = energy_df.groupby('Plant').agg({
            'EnergyConsumption_kWh': ['mean', 'std'],
            'CO2Emissions_kg': 'mean',
            'EnergyEfficiency': 'mean',
            'RenewableEnergy_Percentage': 'mean'
        }).round(3)
        
        st.write("Plant Performance Metrics:")
        st.dataframe(plant_metrics)
        
        # Plant comparison chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        plants = energy_df['Plant'].unique()
        
        # Energy consumption by plant
        avg_energy = energy_df.groupby('Plant')['EnergyConsumption_kWh'].mean()
        ax1.bar(plants, avg_energy.values, color='skyblue')
        ax1.set_title('Average Energy Consumption by Plant')
        ax1.set_ylabel('Energy (kWh)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Energy efficiency by plant
        avg_efficiency = energy_df.groupby('Plant')['EnergyEfficiency'].mean()
        ax2.bar(plants, avg_efficiency.values, color='lightgreen')
        ax2.set_title('Average Energy Efficiency by Plant')
        ax2.set_ylabel('Efficiency')
        ax2.tick_params(axis='x', rotation=45)
        
        # Renewable energy by plant
        avg_renewable = energy_df.groupby('Plant')['RenewableEnergy_Percentage'].mean()
        ax3.bar(plants, avg_renewable.values, color='orange')
        ax3.set_title('Average Renewable Energy by Plant')
        ax3.set_ylabel('Renewable Energy (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        # CO2 emissions by plant
        avg_emissions = energy_df.groupby('Plant')['CO2Emissions_kg'].mean()
        ax4.bar(plants, avg_emissions.values, color='lightcoral')
        ax4.set_title('Average CO2 Emissions by Plant')
        ax4.set_ylabel('CO2 Emissions (kg)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab4:
        st.subheader("Compliance Status")
        
        if len(compliance_df) > 0:
            # Compliance overview
            st.write("Compliance Overview:")
            st.dataframe(compliance_df)
            
            # Compliance score distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            compliance_df['ComplianceScore'].hist(bins=20, ax=ax, color='lightblue', edgecolor='black')
            ax.set_title('Distribution of Compliance Scores')
            ax.set_xlabel('Compliance Score')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # ISO certification status
            iso_status = compliance_df['ISO50001_Status'].value_counts()
            st.write("ISO 50001 Certification Status:")
            st.write(iso_status)
            
            # ESG rating distribution
            esg_ratings = compliance_df['ESG_Rating'].value_counts()
            st.write("ESG Rating Distribution:")
            st.write(esg_ratings)
        else:
            st.info("No compliance data available for the selected time period.")

def display_ml_analysis(energy_df, production_df, compliance_df):
    """Display ML analysis for energy consumption forecasting."""
    
    st.header("🤖 Machine Learning Analysis")
    st.markdown("Machine learning models for energy consumption forecasting and prediction.")
    
    # Initialize ML forecaster
    if st.session_state.ml_models is None:
        with st.spinner("🔄 Training machine learning models..."):
            forecaster = EnergyConsumptionForecaster()
            X_train, X_test, y_train, y_test, feature_names, df = forecaster.prepare_features(energy_df, production_df)
            model_performance = forecaster.train_models(X_train, X_test, y_train, y_test)
            st.session_state.ml_models = forecaster
        st.success("✅ ML models trained successfully!")
    
    forecaster = st.session_state.ml_models
    
    # Create tabs for ML analysis
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Performance", "🔍 Feature Importance", "📈 Forecasting", "💡 Insights"])
    
    with tab1:
        st.subheader("Model Performance Comparison")
        
        # Display model performance summary
        performance_df = forecaster.get_model_summary()
        st.dataframe(performance_df)
        
        # Model performance visualization
        performance_chart = forecaster.create_model_comparison_chart()
        st.plotly_chart(performance_chart, use_container_width=True)
        
        # Best performing model
        best_model = performance_df.loc[performance_df['Test R²'].astype(float).idxmax(), 'Model']
        st.success(f"🏆 Best performing model: **{best_model}**")
    
    with tab2:
        st.subheader("Feature Importance Analysis")
        
        # Model selection for feature importance
        model_name = st.selectbox(
            "Select Model for Feature Importance:",
            list(forecaster.models.keys())
        )
        
        if model_name in forecaster.feature_importance:
            feature_chart = forecaster.create_feature_importance_chart(forecaster.prepare_features(energy_df, production_df)[4], model_name)
            if feature_chart:
                st.plotly_chart(feature_chart, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
    
    with tab3:
        st.subheader("Energy Consumption Forecasting")
        
        # Model selection for forecasting
        forecast_model = st.selectbox(
            "Select Model for Forecasting:",
            list(forecaster.models.keys()),
            key="forecast_model"
        )
        
        # Days to forecast
        forecast_days = st.slider("Days to Forecast:", min_value=7, max_value=90, value=30, step=7)
        
        if st.button("🚀 Generate Forecast"):
            with st.spinner("Generating forecast..."):
                # Get the prepared data
                X_train, X_test, y_train, y_test, feature_names, df = forecaster.prepare_features(energy_df, production_df)
                
                # Create forecast visualization
                forecast_chart = forecaster.create_forecast_visualization(y_train, y_test, forecast_model)
                if forecast_chart:
                    st.plotly_chart(forecast_chart, use_container_width=True)
                
                # Generate future forecast
                future_forecast = forecaster.forecast_future_consumption(energy_df, production_df, forecast_days, forecast_model)
                
                st.subheader("Future Energy Consumption Forecast")
                st.dataframe(future_forecast)
                
                # Forecast summary
                avg_forecast = future_forecast['EnergyConsumption_kWh'].mean()
                current_avg = energy_df['EnergyConsumption_kWh'].mean()
                change_percent = ((avg_forecast - current_avg) / current_avg) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Average", f"{current_avg:,.0f} kWh")
                with col2:
                    st.metric("Forecasted Average", f"{avg_forecast:,.0f} kWh")
                with col3:
                    st.metric("Change", f"{change_percent:+.1f}%")
    
    with tab4:
        st.subheader("ML Insights & Recommendations")
        
        # Key insights
        st.info("""
        **Key ML Insights:**
        
        • **Feature Importance**: Time-based features and production metrics are most predictive
        • **Seasonality**: Energy consumption shows clear seasonal patterns
        • **Production Correlation**: Strong correlation between production and energy usage
        • **Model Performance**: Ensemble methods (Random Forest, Gradient Boosting) perform best
        
        **Recommendations:**
        
        • Use ensemble models for production forecasting
        • Incorporate seasonal adjustments in energy planning
        • Monitor production-energy correlation for optimization
        • Implement real-time monitoring for predictive maintenance
        """)
        
        # Data quality assessment
        st.subheader("Data Quality Assessment")
        
        # Check for missing values
        missing_energy = energy_df.isnull().sum().sum()
        missing_production = production_df.isnull().sum().sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Missing Values (Energy)", missing_energy)
        with col2:
            st.metric("Missing Values (Production)", missing_production)
        
        if missing_energy == 0 and missing_production == 0:
            st.success("✅ Data quality is excellent - no missing values detected!")
        else:
            st.warning("⚠️ Missing values detected. Consider data cleaning before ML analysis.")

def display_ai_analysis(energy_df, production_df, compliance_df):
    """Display AI analysis with optimization and reinforcement learning."""
    
    st.header("🧠 AI Analysis")
    st.markdown("AI-powered energy optimization and reinforcement learning simulation.")
    
    # Initialize AI optimization engine
    if st.session_state.ai_optimization is None:
        with st.spinner("🔄 Initializing AI optimization engine..."):
            optimization_engine = EnergyOptimizationEngine()
            st.session_state.ai_optimization = optimization_engine
        st.success("✅ AI optimization engine initialized!")
    
    optimization_engine = st.session_state.ai_optimization
    
    # Create tabs for AI analysis
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Pattern Analysis", "⚡ Optimization", "🤖 RL Simulation", "📊 Dashboard"])
    
    with tab1:
        st.subheader("Energy Pattern Analysis")
        
        if st.button("🔍 Run Pattern Analysis"):
            with st.spinner("Analyzing energy patterns..."):
                analysis_results = optimization_engine.analyze_energy_patterns(energy_df, production_df)
                st.session_state.analysis_results = analysis_results
            
            st.success("✅ Pattern analysis completed!")
        
        if 'analysis_results' in st.session_state:
            results = st.session_state.analysis_results
            
            # Display analysis results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Peak Demand Analysis")
                peak_analysis = results['peak_demand_analysis']
                st.write(f"**High Peak Demand Plants:** {', '.join(peak_analysis['high_peak_plants'])}")
                st.write(f"**Reduction Potential:** {peak_analysis['peak_demand_reduction_potential']}")
                
                st.subheader("Efficiency Analysis")
                efficiency_analysis = results['efficiency_analysis']
                st.write(f"**Current Efficiency:** {efficiency_analysis['current_efficiency']:.3f}")
                st.write(f"**Target Efficiency:** {efficiency_analysis['target_efficiency']:.3f}")
                st.write(f"**Improvement Potential:** {efficiency_analysis['improvement_potential']:.1f}%")
            
            with col2:
                st.subheader("Seasonal Patterns")
                seasonal_analysis = results['seasonal_patterns']
                st.write(f"**Peak Month:** {seasonal_analysis['peak_consumption_month']}")
                st.write(f"**Low Month:** {seasonal_analysis['low_consumption_month']}")
                st.write(f"**Seasonal Variation:** {seasonal_analysis['seasonal_variation']:.1f}%")
                
                st.subheader("Plant Performance")
                plant_analysis = results['plant_comparison']
                st.write(f"**Top Performer:** {plant_analysis['top_performer']}")
                st.write(f"**Bottom Performer:** {plant_analysis['bottom_performer']}")
                st.write(f"**Performance Gap:** {plant_analysis['performance_gap']:.3f}")
    
    with tab2:
        st.subheader("Optimization Opportunities")
        
        if 'analysis_results' in st.session_state:
            results = st.session_state.analysis_results
            opportunities = results['optimization_opportunities']
            
            if opportunities:
                st.write("**Identified Optimization Opportunities:**")
                
                for i, opp in enumerate(opportunities):
                    with st.expander(f"Opportunity {i+1}: {opp['action']}"):
                        st.write(f"**Condition:** {opp['condition']}")
                        st.write(f"**Action:** {opp['action']}")
                        st.write(f"**Affected Plants:** {', '.join(opp['affected_plants'])}")
                        st.write(f"**Expected Savings:** {opp['expected_savings']}")
                        st.write(f"**Priority:** {opp['priority']}")
                
                # Cost analysis
                cost_analysis = results['cost_analysis']
                st.subheader("Cost Analysis")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Energy Cost", f"${cost_analysis['total_energy_cost']:,.2f}")
                with col2:
                    st.metric("Potential Savings", f"${cost_analysis['total_potential_savings']:,.2f}")
                with col3:
                    st.metric("ROI Estimate", f"{cost_analysis['roi_estimate']:.1f}%")
            else:
                st.info("No optimization opportunities identified with current thresholds.")
    
    with tab3:
        st.subheader("Reinforcement Learning Simulation")
        
        # Simulation parameters
        col1, col2 = st.columns(2)
        with col1:
            simulation_days = st.slider("Simulation Days:", min_value=7, max_value=90, value=30, step=7)
        with col2:
            if st.button("🚀 Run RL Simulation"):
                with st.spinner("Running reinforcement learning simulation..."):
                    simulation_results = optimization_engine.run_reinforcement_learning_simulation(
                        energy_df, production_df, simulation_days
                    )
                    st.session_state.simulation_results = simulation_results
                st.success("✅ RL simulation completed!")
        
        if 'simulation_results' in st.session_state:
            results = st.session_state.simulation_results
            
            # Simulation summary
            analysis = results['simulation_analysis']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Reward", f"{analysis['total_reward']:.0f}")
            with col2:
                st.metric("Avg Daily Reward", f"{analysis['avg_daily_reward']:.1f}")
            with col3:
                st.metric("Energy Trend", analysis['energy_consumption_trend'])
            with col4:
                st.metric("Most Effective Action", analysis['most_effective_action'])
            
            # Optimization actions
            st.subheader("Recommended Optimization Actions")
            actions = results['optimization_actions']
            
            for action in actions:
                with st.expander(f"{action['action']} - {action['expected_impact']} Impact"):
                    st.write(f"**Description:** {action['description']}")
                    st.write(f"**Implementation Time:** {action['implementation_time']}")
                    st.write(f"**Expected Impact:** {action['expected_impact']}")
    
    with tab4:
        st.subheader("AI Optimization Dashboard")
        
        if 'analysis_results' in st.session_state:
            results = st.session_state.analysis_results
            
            # Create optimization dashboard
            dashboard = optimization_engine.create_optimization_dashboard(results)
            st.plotly_chart(dashboard, use_container_width=True)
        else:
            st.info("Run pattern analysis first to generate the optimization dashboard.")

def display_genai_analysis(energy_df, production_df, compliance_df):
    """Display GenAI analysis with ESG reporting and chatbot."""
    
    st.header("🌟 GenAI Analysis")
    st.markdown("AI-generated ESG reports and intelligent chatbot assistance.")
    
    # Initialize ESG report generator
    if st.session_state.esg_report is None:
        with st.spinner("🔄 Initializing ESG report generator..."):
            esg_generator = ESGReportGenerator()
            st.session_state.esg_report = esg_generator
        st.success("✅ ESG report generator initialized!")
    
    esg_generator = st.session_state.esg_report
    
    # Create tabs for GenAI analysis
    tab1, tab2, tab3, tab4 = st.tabs(["📊 ESG Report", "📈 ESG Dashboard", "🤖 Chatbot", "💡 Recommendations"])
    
    with tab1:
        st.subheader("AI-Generated ESG Report")
        
        if st.button("📋 Generate ESG Report"):
            with st.spinner("Generating comprehensive ESG report..."):
                esg_report = esg_generator.generate_esg_report(energy_df, production_df, compliance_df)
                st.session_state.esg_report_data = esg_report
            st.success("✅ ESG report generated successfully!")
        
        if 'esg_report_data' in st.session_state:
            report = st.session_state.esg_report_data
            
            # Report metadata
            st.subheader("Report Information")
            metadata = report['report_metadata']
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Report Title:** {metadata['report_title']}")
                st.write(f"**Report Type:** {metadata['report_type']}")
            with col2:
                st.write(f"**Generation Date:** {metadata['generation_date']}")
                st.write(f"**Data Period:** {metadata['data_period']}")
            
            # Executive summary
            st.subheader("Executive Summary")
            exec_summary = report['executive_summary']
            st.write(exec_summary['overview'])
            
            st.write("**Key Highlights:**")
            for highlight in exec_summary['key_highlights']:
                st.write(f"• {highlight}")
            
            st.write("**Strategic Focus:**")
            for focus in exec_summary['strategic_focus']:
                st.write(f"• {focus}")
            
            # Environmental performance
            st.subheader("Environmental Performance")
            env_perf = report['environmental_performance']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Energy Management:**")
                for key, value in env_perf['energy_management'].items():
                    st.write(f"• {key.replace('_', ' ').title()}: {value}")
            
            with col2:
                st.write("**Emissions Management:**")
                for key, value in env_perf['emissions_management'].items():
                    st.write(f"• {key.replace('_', ' ').title()}: {value}")
            
            # Social responsibility
            st.subheader("Social Responsibility")
            social_perf = report['social_responsibility']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Operational Excellence:**")
                for key, value in social_perf['operational_excellence'].items():
                    st.write(f"• {key.replace('_', ' ').title()}: {value}")
            
            with col2:
                st.write("**Workplace Safety:**")
                for key, value in social_perf['workplace_safety'].items():
                    st.write(f"• {key.replace('_', ' ').title()}: {value}")
            
            # Governance and compliance
            st.subheader("Governance & Compliance")
            gov_perf = report['governance_compliance']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Compliance Status:**")
                for key, value in gov_perf['compliance_status'].items():
                    st.write(f"• {key.replace('_', ' ').title()}: {value}")
            
            with col2:
                st.write("**ESG Ratings:**")
                for key, value in gov_perf['esg_ratings'].items():
                    st.write(f"• {key.replace('_', ' ').title()}: {value}")
    
    with tab2:
        st.subheader("ESG Performance Dashboard")
        
        if 'esg_report_data' in st.session_state:
            report = st.session_state.esg_report_data
            esg_insights = report['esg_insights']
            
            # Create ESG dashboard
            dashboard = esg_generator.create_esg_dashboard(esg_insights)
            st.plotly_chart(dashboard, use_container_width=True)
            
            # ESG score breakdown
            st.subheader("ESG Score Breakdown")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                env_score = esg_insights['environmental']['avg_energy_efficiency'] * 50
                st.metric("Environmental Score", f"{env_score:.1f}/50")
            
            with col2:
                social_score = esg_insights['social']['avg_production_efficiency']
                st.metric("Social Score", f"{social_score:.1f}/30")
            
            with col3:
                gov_score = esg_insights['governance']['avg_compliance_score']
                st.metric("Governance Score", f"{gov_score:.1f}/20")
            
            # Overall ESG rating
            overall_score = esg_insights['overall_esg_score']
            esg_rating = esg_insights['esg_rating']
            
            st.subheader("Overall ESG Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Overall ESG Score", f"{overall_score:.1f}/100")
            with col2:
                st.metric("ESG Rating", esg_rating)
        else:
            st.info("Generate ESG report first to view the dashboard.")
    
    with tab3:
        st.subheader("ESG Assistant Chatbot")
        
        # Create chatbot interface
        esg_generator.create_chatbot_interface()
    
    with tab4:
        st.subheader("Strategic Recommendations")
        
        if 'esg_report_data' in st.session_state:
            report = st.session_state.esg_report_data
            recommendations = report['strategic_recommendations']
            
            if recommendations:
                st.write("**AI-Generated Strategic Recommendations:**")
                
                for i, rec in enumerate(recommendations):
                    with st.expander(f"Recommendation {i+1}: {rec['recommendation']}"):
                        st.write(f"**Category:** {rec['category']}")
                        st.write(f"**Priority:** {rec['priority']}")
                        st.write(f"**Description:** {rec['recommendation']}")
                        st.write(f"**Expected Impact:** {rec['expected_impact']}")
                        st.write(f"**Timeline:** {rec['timeline']}")
                        st.write(f"**Investment Required:** {rec['investment_required']}")
            else:
                st.info("No specific recommendations generated. Your ESG performance is already strong!")
        else:
            st.info("Generate ESG report first to view strategic recommendations.")

if __name__ == "__main__":
    main()
