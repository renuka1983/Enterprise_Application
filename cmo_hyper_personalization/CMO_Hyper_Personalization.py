"""
CMO Hyper-Personalization & Market Intelligence Application
Manufacturing Products - Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import modules
try:
    # Layer 1: Relative imports (package context)
    from .data.synthetic_data import ManufacturingDataGenerator, generate_sample_data
    from .models.ml_models import CampaignResponsePredictor, CustomerSegmentation
except ImportError:
    try:
        # Layer 2: Absolute imports (direct execution)
        from data.synthetic_data import ManufacturingDataGenerator, generate_sample_data
        from models.ml_models import CampaignResponsePredictor, CustomerSegmentation
    except ImportError:
        # Layer 3: Dynamic path addition (final fallback)
        import os
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from data.synthetic_data import ManufacturingDataGenerator, generate_sample_data
        from models.ml_models import CampaignResponsePredictor, CustomerSegmentation

# Page configuration
st.set_page_config(
            page_title="CMO Hyper-Personalization - Manufacturing",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main function for the CMO Hyper-Personalization app."""
    
    # Header
    st.title("🎯 CMO Hyper-Personalization & Market Intelligence")
    st.markdown("### Manufacturing Products - AI-Powered Customer Personalization")
    
    # Info section
    if st.session_state.get('show_info', False):
        with st.expander("ℹ️ Module Information & Use Cases", expanded=True):
            st.markdown("""
            ## 🎯 **Use Case: CMO Hyper-Personalization & Market Intelligence**
            
            This module addresses the critical challenge of **customer personalization and market intelligence** in manufacturing, 
            providing AI-powered insights to create targeted marketing campaigns, understand customer behavior, and drive revenue growth.
            
            ### 🏭 **Target Industries**
            - **Steel Manufacturing**: B2B customer segmentation and market analysis
            - **Automotive Parts**: Customer behavior prediction and campaign optimization
            - **Chemical Manufacturing**: Market intelligence and competitive analysis
            - **Electronics Components**: Customer personalization and demand forecasting
            - **Food Processing**: Consumer behavior analysis and market trends
            - **Textile Manufacturing**: Customer segmentation and campaign effectiveness
            - **Pharmaceutical Production**: Market intelligence and regulatory insights
            - **Paper Manufacturing**: B2B customer relationship management
            - **Cement Production**: Construction industry market analysis
            
            ### 🚀 **Key Functionalities**
            
            #### **📈 Traditional Analysis**
            - **Customer Segmentation**: Demographic and behavioral segmentation
            - **Market Summary**: Regional and segment-based market overview
            - **Static Reports**: Basic customer and market intelligence
            - **Summary Statistics**: Customer behavior and market trends
            - **Regional Analysis**: Geographic distribution and performance
            
            #### **🤖 Machine Learning Prediction**
            - **Campaign Response Prediction**: Forecast customer response to marketing campaigns
            - **Multiple ML Models**: Logistic Regression and Random Forest classifiers
            - **Feature Importance Analysis**: Understand what drives customer behavior
            - **Response Probability**: Predict likelihood of campaign engagement
            - **Model Performance**: Evaluate and optimize prediction accuracy
            
            #### **🧠 AI-Powered Analysis**
            - **Customer Clustering**: K-means clustering for customer segmentation
            - **NLP Analysis**: Analyze competitor mentions and market sentiment
            - **Market Intelligence**: Competitive landscape and market dynamics
            - **Pattern Recognition**: Identify customer behavior patterns
            - **Predictive Analytics**: Forecast market trends and customer needs
            
            #### **🌟 GenAI Personalization**
            - **AI-Generated Pitches**: Personalized product recommendations
            - **Dynamic Content**: Context-aware marketing messages
            - **Customer Insights**: AI-powered customer understanding
            - **Campaign Optimization**: Personalized marketing strategies
            - **Revenue Impact**: Predict personalization benefits
            
            ### 💰 **Business Value**
            - **Revenue Growth**: 15-25% increase through personalization
            - **Customer Engagement**: Improved campaign response rates
            - **Market Intelligence**: Competitive advantage through insights
            - **Marketing Efficiency**: Optimized campaign targeting
            - **Customer Retention**: Enhanced customer relationship management
            
            ### 🔧 **Technical Features**
            - **Synthetic Data Generation**: Realistic customer and market datasets
            - **Interactive Visualizations**: Dynamic charts and dashboards
            - **Real-time Analysis**: On-demand insights and recommendations
            - **Scalable Architecture**: Support for large customer datasets
            - **API Integration Ready**: Prepared for real CRM and marketing systems
            
            ### 📊 **Data Sources**
            - **Customer Data**: Demographics, purchase history, behavior patterns
            - **Market Intelligence**: Competitor mentions, market events, trends
            - **Campaign Data**: Response rates, engagement metrics, performance
            - **Regional Data**: Geographic distribution and market dynamics
            - **Behavioral Data**: Website visits, purchase patterns, preferences
            
            ### 🎯 **User Personas**
            - **CMOs**: Marketing strategy, campaign optimization, revenue growth
            - **Marketing Managers**: Campaign execution, customer targeting, performance analysis
            - **Sales Teams**: Customer insights, lead qualification, relationship management
            - **Business Analysts**: Market research, competitive analysis, trend identification
            - **Product Managers**: Customer needs, market opportunities, product positioning
            """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data parameters
        st.subheader("📊 Data Parameters")
        num_customers = st.slider("Number of Customers", min_value=500, max_value=2000, value=1000, step=100)
        num_market_records = st.slider("Market Intelligence Records", min_value=200, max_value=1000, value=500, step=100)
        random_seed = st.number_input("Random Seed", min_value=1, max_value=1000, value=42, step=1)
        
        # Approach selection
        st.subheader("🔍 Analysis Approach")
        approach = st.selectbox(
            "Select Analysis Approach",
            ["Traditional", "ML", "AI", "GenAI"],
            help="Choose the analysis method to explore"
        )
        
        # Generate data button
        if st.button("🔄 Generate New Dataset", type="primary"):
            st.session_state.customer_data = None
            st.session_state.market_data = None
            st.rerun()
        
        st.markdown("---")
        
        # Info button
        if st.button("ℹ️ Module Info", type="secondary"):
            st.session_state.show_info = not st.session_state.get('show_info', False)
        
        # App information
        st.subheader("ℹ️ About")
        st.markdown("""
        This application demonstrates four approaches to customer personalization:
        
        **📈 Traditional**: Static segmentation and summary statistics
        **🤖 ML**: Machine learning for campaign response prediction
        **🧠 AI**: Clustering and NLP analysis of competitor mentions
        **🌟 GenAI**: AI-generated personalized product pitches
        
        **Data**: Synthetic manufacturing customer data with realistic patterns
        """)
    
    # Initialize session state
    if 'customer_data' not in st.session_state:
        st.session_state.customer_data = None
    if 'market_data' not in st.session_state:
        st.session_state.market_data = None
    if 'ml_models' not in st.session_state:
        st.session_state.ml_models = None
    if 'segmentation' not in st.session_state:
        st.session_state.segmentation = None
    
    # Generate or load data
    if st.session_state.customer_data is None:
        with st.spinner("🔄 Generating synthetic manufacturing customer dataset..."):
            customer_df, market_df, summary = generate_sample_data(num_customers, num_market_records)
            st.session_state.customer_data = customer_df
            st.session_state.market_data = market_df
            st.session_state.data_summary = summary
        st.success(f"✅ Generated dataset with {len(customer_df)} customers and {len(market_df)} market records")
    
    customer_df = st.session_state.customer_data
    market_df = st.session_state.market_data
    summary = st.session_state.data_summary
    
    # Key metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{summary['total_customers']:,}")
    
    with col2:
        st.metric("Campaign Response Rate", f"{summary['campaign_response_rate']:.1f}%")
    
    with col3:
        st.metric("Total Revenue Potential", f"₹{summary['total_revenue_potential']:.0f}L")
    
    with col4:
        st.metric("High-Value Customers", f"{summary['high_value_customers']:,}")
    
    # Main content based on selected approach
    if approach == "Traditional":
        show_traditional_analysis(customer_df, market_df, summary)
    elif approach == "ML":
        show_ml_analysis(customer_df)
    elif approach == "AI":
        show_ai_analysis(customer_df, market_df)
    elif approach == "GenAI":
        show_genai_analysis(customer_df)

def show_traditional_analysis(customer_df, market_df, summary):
    """Display traditional analysis with static segmentation."""
    
    st.header("📈 Traditional Analysis - Static Segmentation")
    st.markdown("Statistical analysis and manual insights using Excel-style calculations.")
    
    # Customer segmentation overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("👥 Customer Segmentation Overview")
        
        # Segment distribution
        segment_fig = px.pie(
            customer_df, 
            names='Segment', 
            title='Customer Distribution by Segment'
        )
        st.plotly_chart(segment_fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Segment Statistics")
        
        segment_stats = customer_df.groupby('Segment').agg({
            'RevenuePotential': ['mean', 'count'],
            'ResponseToCampaign': 'mean'
        }).round(2)
        
        segment_stats.columns = ['Avg_Revenue', 'Count', 'Response_Rate']
        segment_stats['Response_Rate'] = segment_stats['Response_Rate'] * 100
        
        st.dataframe(segment_stats, use_container_width=True)
    
    # Regional analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗺️ Regional Distribution")
        
        region_fig = px.bar(
            customer_df.groupby('Region').size().reset_index(name='Count'),
            x='Region',
            y='Count',
            title='Customer Distribution by Region'
        )
        region_fig.update_xaxes(tickangle=45)
        st.plotly_chart(region_fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Revenue by Region")
        
        revenue_by_region = customer_df.groupby('Region')['RevenuePotential'].mean().round(2)
        revenue_fig = px.bar(
            x=revenue_by_region.index,
            y=revenue_by_region.values,
            title='Average Revenue Potential by Region'
        )
        revenue_fig.update_xaxes(tickangle=45)
        st.plotly_chart(revenue_fig, use_container_width=True)
    
    # Behavioral analysis
    st.subheader("📊 Customer Behavior Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Past purchases distribution
        purchase_fig = px.histogram(
            customer_df,
            x='PastPurchases',
            title='Distribution of Past Purchases',
            nbins=20
        )
        st.plotly_chart(purchase_fig, use_container_width=True)
    
    with col2:
        # Website visits distribution
        visits_fig = px.histogram(
            customer_df,
            x='WebsiteVisits',
            title='Distribution of Website Visits',
            nbins=20
        )
        st.plotly_chart(visits_fig, use_container_width=True)
    
    # Campaign response analysis
    st.subheader("📢 Campaign Response Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Response by segment
        response_by_segment = customer_df.groupby('Segment')['ResponseToCampaign'].mean() * 100
        response_fig = px.bar(
            x=response_by_segment.index,
            y=response_by_segment.values,
            title='Campaign Response Rate by Segment (%)'
        )
        st.plotly_chart(response_fig, use_container_width=True)
    
    with col2:
        # Response by region
        response_by_region = customer_df.groupby('Region')['ResponseToCampaign'].mean() * 100
        response_region_fig = px.bar(
            x=response_by_region.index,
            y=response_by_region.values,
            title='Campaign Response Rate by Region (%)'
        )
        response_region_fig.update_xaxes(tickangle=45)
        st.plotly_chart(response_region_fig, use_container_width=True)
    
    # Data download
    st.subheader("💾 Download Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = customer_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Customer Data (CSV)",
            data=csv_data,
            file_name="tubes_india_customers.csv",
            mime="text/csv"
        )
    
    with col2:
        market_csv = market_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Market Intelligence (CSV)",
            data=market_csv,
            file_name="tubes_india_market_intelligence.csv",
            mime="text/csv"
        )

def show_ml_analysis(customer_df):
    """Display machine learning analysis."""
    
    st.header("🤖 Machine Learning Analysis - Campaign Response Prediction")
    st.markdown("Train and compare machine learning models for predicting campaign response.")
    
    if st.button("🚀 Train ML Models", type="primary"):
        with st.spinner("Training machine learning models..."):
            # Initialize predictor
            predictor = CampaignResponsePredictor()
            
            # Prepare features
            X_train, X_test, y_train, y_test, feature_names, scaler, encoders = predictor.prepare_features(customer_df)
            
            # Train models
            lr_results = predictor.train_logistic_regression(X_train, X_test, y_train, y_test, feature_names)
            rf_results = predictor.train_random_forest(X_train, X_test, y_train, y_test, feature_names)
            
            # Store results
            st.session_state.ml_models = predictor
            st.session_state.ml_data = {
                'X_test': X_test,
                'y_test': y_test,
                'feature_names': feature_names
            }
            st.success("✅ Models trained successfully!")
            st.rerun()
    
    if 'ml_models' in st.session_state:
        predictor = st.session_state.ml_models
        
        # Model comparison
        st.subheader("📊 Model Performance Comparison")
        comparison_fig = predictor.create_model_comparison_chart()
        if comparison_fig:
            st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Feature importance
        st.subheader("🔍 Feature Importance Analysis")
        importance_fig = predictor.create_feature_importance_chart()
        if importance_fig:
            st.plotly_chart(importance_fig, use_container_width=True)
        
        # Model summary
        st.subheader("📋 Model Summary")
        summary_df = predictor.get_model_summary()
        if not summary_df.empty:
            st.dataframe(summary_df, use_container_width=True)
        
        # Individual customer prediction
        st.subheader("👤 Individual Customer Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_segment = st.selectbox("Customer Segment", customer_df['Segment'].unique())
            segment_customers = customer_df[customer_df['Segment'] == selected_segment]
            selected_customer = st.selectbox("Customer", segment_customers['CustomerID'].tolist())
        
        with col2:
            if selected_customer:
                customer_data = customer_df[customer_df['CustomerID'] == selected_customer].iloc[0]
                st.markdown(f"**Customer Details:**")
                st.markdown(f"- Segment: {customer_data['Segment']}")
                st.markdown(f"- Region: {customer_data['Region']}")
                st.markdown(f"- Past Purchases: {customer_data['PastPurchases']}")
                st.markdown(f"- Revenue Potential: ₹{customer_data['RevenuePotential']:.0f}L")
        
        with col3:
            if selected_customer and st.button("🔮 Predict Response"):
                prediction, probability = predictor.predict_campaign_response(customer_data)
                
                if prediction == 1:
                    st.success(f"✅ **Likely to Respond**: {probability*100:.1f}% probability")
                else:
                    st.error(f"❌ **Unlikely to Respond**: {probability*100:.1f}% probability")
    else:
        st.info("🚀 Click the 'Train ML Models' button above to start training machine learning models for campaign response prediction.")

def show_ai_analysis(customer_df, market_df):
    """Display AI analysis with clustering and NLP."""
    
    st.header("🧠 AI Analysis - Clustering & NLP")
    st.markdown("Advanced AI techniques for customer segmentation and competitor analysis.")
    
    if st.button("🤖 Run AI Analysis", type="primary"):
        with st.spinner("Running AI analysis..."):
            # Customer segmentation
            segmentation = CustomerSegmentation()
            df_clustered = segmentation.perform_kmeans_segmentation(customer_df)
            
            # Store results
            st.session_state.segmentation = segmentation
            st.session_state.clustered_data = df_clustered
            st.success("✅ AI analysis completed!")
            st.rerun()
    
    if 'segmentation' in st.session_state:
        segmentation = st.session_state.segmentation
        df_clustered = st.session_state.clustered_data
        
        # Segmentation visualization
        st.subheader("🎯 Customer Segmentation - 3D View")
        seg_fig = segmentation.create_segmentation_visualization(df_clustered)
        if seg_fig:
            st.plotly_chart(seg_fig, use_container_width=True)
        
        # Cluster summary
        st.subheader("📊 Cluster Summary Statistics")
        cluster_summary = segmentation.get_cluster_summary(df_clustered)
        if not cluster_summary.empty:
            st.dataframe(cluster_summary, use_container_width=True)
        
        # Cluster analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Cluster distribution
            cluster_counts = df_clustered['ClusterName'].value_counts()
            cluster_fig = px.pie(
                values=cluster_counts.values,
                names=cluster_counts.index,
                title='Customer Distribution by Cluster'
            )
            st.plotly_chart(cluster_fig, use_container_width=True)
        
        with col2:
            # Revenue by cluster
            revenue_by_cluster = df_clustered.groupby('ClusterName')['RevenuePotential'].mean().round(2)
            revenue_cluster_fig = px.bar(
                x=revenue_by_cluster.index,
                y=revenue_by_cluster.values,
                title='Average Revenue by Cluster'
            )
            revenue_cluster_fig.update_xaxes(tickangle=45)
            st.plotly_chart(revenue_cluster_fig, use_container_width=True)
        
        # Competitor analysis
        st.subheader("🏢 Competitor Mention Analysis")
        
        # Extract competitor mentions
        competitor_mentions = customer_df[customer_df['CompetitorMentions'] != 'None']['CompetitorMentions']
        if not competitor_mentions.empty:
            # Count mentions
            all_mentions = []
            for mentions in competitor_mentions:
                all_mentions.extend([comp.strip() for comp in mentions.split(', ')])
            
            mention_counts = pd.Series(all_mentions).value_counts().head(10)
            
            mention_fig = px.bar(
                x=mention_counts.values,
                y=mention_counts.index,
                orientation='h',
                title='Top 10 Competitor Mentions'
            )
            st.plotly_chart(mention_fig, use_container_width=True)
        else:
            st.info("No competitor mentions found in the dataset.")
    else:
        st.info("🤖 Click the 'Run AI Analysis' button above to start the AI-powered customer segmentation and competitor analysis.")

def show_genai_analysis(customer_df):
    """Display GenAI analysis with personalized pitches."""
    
    st.header("🌟 GenAI Analysis - Personalized Product Pitches")
    st.markdown("AI-generated personalized recommendations and product pitches for individual customers.")
    
    st.info("🤖 **GenAI Integration Note**: This is a simulated AI analysis. In production, this would use OpenAI's GPT-4 API to generate dynamic, personalized content.")
    
    # Customer selection for personalization
    st.subheader("👤 Select Customer for Personalization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_segment = st.selectbox("Customer Segment", customer_df['Segment'].unique(), key="genai_segment")
        segment_customers = customer_df[customer_df['Segment'] == selected_segment]
        selected_customer = st.selectbox("Customer", segment_customers['CustomerID'].tolist(), key="genai_customer")
    
    with col2:
        if selected_customer:
            customer_data = customer_df[customer_df['CustomerID'] == selected_customer].iloc[0]
            st.markdown(f"**Customer Profile:**")
            st.markdown(f"- **ID**: {customer_data['CustomerID']}")
            st.markdown(f"- **Segment**: {customer_data['Segment']}")
            st.markdown(f"- **Region**: {customer_data['Region']}")
            st.markdown(f"- **Past Purchases**: {customer_data['PastPurchases']}")
            st.markdown(f"- **Revenue Potential**: ₹{customer_data['RevenuePotential']:.0f}L")
            st.markdown(f"- **Website Visits**: {customer_data['WebsiteVisits']:.1f}/month")
    
    # Product selection
    if selected_customer:
        st.subheader("🏭 Select Product for Personalization")
        
        products = ['Steel Tubes', 'Aluminum Tubes', 'Copper Tubes', 'Plastic Tubes', 'Composite Tubes', 'Specialty Tubes']
        selected_product = st.selectbox("Product", products)
        
        if st.button("🎯 Generate Personalized Pitch", type="primary"):
            with st.spinner("Generating personalized product pitch..."):
                # Simulate GenAI personalization
                personalization_results = generate_simulated_personalization(customer_data, selected_product)
                st.session_state.personalization_results = personalization_results
                st.success("✅ Personalized pitch generated!")
                st.rerun()
    
    # Display personalization results
    if 'personalization_results' in st.session_state:
        results = st.session_state.personalization_results
        
        st.subheader("🎯 Personalized Product Pitch")
        
        # Customer profile analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Customer Profile Analysis**")
            profile = results['customer_profile']
            
            profile_data = {
                'Metric': ['Segment', 'Region', 'Purchase History', 'Online Engagement', 'Revenue Potential'],
                'Value': [
                    profile['segment'],
                    profile['region'],
                    profile['purchase_history'],
                    f"{profile['online_engagement']:.1f}/month",
                    f"₹{profile['revenue_potential']:.0f}L"
                ]
            }
            
            profile_df = pd.DataFrame(profile_data)
            st.dataframe(profile_df, use_container_width=True)
        
        with col2:
            st.markdown("**🏭 Product Recommendation**")
            st.markdown(f"**Recommended Product**: {results['recommended_product']}")
            st.markdown(f"**Match Score**: {np.random.uniform(0.8, 0.95):.2f}")
            st.markdown(f"**Confidence**: {np.random.uniform(0.85, 0.98):.1%}")
        
        # Personalized pitch
        st.subheader("💬 AI-Generated Personalized Pitch")
        st.markdown(results['personalized_pitch'])
        
        # Recommendations
        st.subheader("💡 AI-Generated Recommendations")
        for rec in results['recommendations']:
            st.markdown(f"• {rec}")
        
        # Revenue impact
        st.subheader("💰 Revenue Impact Projection")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Revenue", f"₹{results['revenue_impact']['current_revenue']:.0f}L")
        
        with col2:
            st.metric("Potential Revenue", f"₹{results['revenue_impact']['potential_revenue']:.0f}L")
        
        with col3:
            st.metric("Growth Potential", f"+{results['revenue_impact']['growth_percentage']:.1f}%")
        
        # Revenue projection chart
        revenue_data = [
            results['revenue_impact']['current_revenue'],
            results['revenue_impact']['potential_revenue']
        ]
        revenue_labels = ['Current', 'Potential']
        
        revenue_fig = px.bar(
            x=revenue_labels,
            y=revenue_data,
            title='Revenue Projection',
            color=['Current', 'Potential'],
            color_discrete_map={'Current': 'blue', 'Potential': 'green'}
        )
        st.plotly_chart(revenue_fig, use_container_width=True)

def generate_simulated_personalization(customer_data, selected_product):
    """Generate simulated personalization results."""
    
    # Simulate customer profile analysis
    customer_profile = {
        'segment': customer_data['Segment'],
        'region': customer_data['Region'],
        'purchase_history': customer_data['PastPurchases'],
        'online_engagement': customer_data['WebsiteVisits'],
        'revenue_potential': customer_data['RevenuePotential'],
        'campaign_response': customer_data['ResponseToCampaign'],
        'competitor_awareness': customer_data['CompetitorMentions'] != 'None'
    }
    
    # Simulate personalized pitch
    personalized_pitch = f"""
**Personalized Product Pitch: {selected_product}**

Dear Valued Customer,

Based on your impressive track record of {customer_data['PastPurchases']} successful projects and your strategic presence in {customer_data['Region']}, we believe {selected_product} represents a perfect alignment with your growth objectives.

**Why {selected_product} is Ideal for Your Operations:**
- **Scalability**: Designed to support {customer_data['Segment'].lower()}-level operations
- **ROI Focus**: Pricing structure optimized for your revenue potential of ₹{customer_data['RevenuePotential']:.0f}L
- **Strategic Value**: Long-term partnership potential with dedicated support

**Key Benefits:**
• High-quality materials and construction
• Customizable specifications for your needs
• Comprehensive technical support
• Competitive pricing for bulk orders

**Perfect for Your Applications:**
• Industrial manufacturing
• Construction projects
• Infrastructure development
• Commercial applications

We're ready to discuss how {selected_product} can drive your next phase of growth.
    """
    
    # Simulate recommendations
    recommendations = [
        "**Engagement Strategy**: Implement targeted email campaigns to increase website visits",
        "**Upsell Opportunity**: Consider premium variants for enhanced performance",
        f"**Regional Advantage**: Leverage our strong presence in {customer_data['Region']} for faster delivery",
        "**Growth Partnership**: Establish long-term supply agreements to support expansion plans"
    ]
    
    # Simulate revenue impact
    base_revenue = customer_data['RevenuePotential']
    potential_revenue = base_revenue * np.random.uniform(1.2, 1.8)
    
    revenue_impact = {
        'current_revenue': base_revenue,
        'potential_revenue': potential_revenue,
        'revenue_increase': potential_revenue - base_revenue,
        'growth_percentage': ((potential_revenue - base_revenue) / base_revenue) * 100
    }
    
    return {
        'customer_profile': customer_profile,
        'recommended_product': selected_product,
        'personalized_pitch': personalized_pitch.strip(),
        'recommendations': recommendations,
        'revenue_impact': revenue_impact
    }

if __name__ == "__main__":
    main()
