"""
GenAI Personalization Engine for CMO Hyper-Personalization
Manufacturing Products - AI-Generated Personalized Pitches
"""

import pandas as pd
import numpy as np
import streamlit as st
import os
import json

# Try to import dotenv, but make it optional
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
    # Load environment variables
    load_dotenv()
except ImportError:
    DOTENV_AVAILABLE = False
    st.info("ℹ️ python-dotenv not available. Environment variables will be loaded from system or Streamlit secrets.")

class TubesIndiaPersonalizationEngine:
    """
    AI-powered personalization engine for generating customized
            product pitches and recommendations for manufacturing customers.
    """
    
    def __init__(self):
        """Initialize the personalization engine."""
        self.openai_api_key = self._get_openai_api_key()
        self.product_catalog = self._initialize_product_catalog()
        self.personalization_templates = self._initialize_templates()
    
    def _get_openai_api_key(self):
        """Get OpenAI API key from environment or Streamlit secrets."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            try:
                api_key = st.secrets["OPENAI_API_KEY"]
            except:
                api_key = None
        return api_key
    
    def _initialize_product_catalog(self):
        """Initialize manufacturing product catalog."""
        return {
            'Steel Tubes': {
                'description': 'High-quality steel tubes for construction and industrial applications',
                'key_features': ['Corrosion resistant', 'High tensile strength', 'Durable'],
                'applications': ['Construction', 'Automotive', 'Manufacturing'],
                'price_range': '₹500-5000 per meter'
            },
            'Aluminum Tubes': {
                'description': 'Lightweight aluminum tubes for aerospace and automotive industries',
                'key_features': ['Lightweight', 'Corrosion resistant', 'High conductivity'],
                'applications': ['Aerospace', 'Automotive', 'Electronics'],
                'price_range': '₹800-8000 per meter'
            },
            'Copper Tubes': {
                'description': 'Premium copper tubes for electrical and plumbing applications',
                'key_features': ['Excellent conductivity', 'Antimicrobial', 'Long lifespan'],
                'applications': ['Electrical', 'Plumbing', 'HVAC'],
                'price_range': '₹1200-12000 per meter'
            },
            'Plastic Tubes': {
                'description': 'Versatile plastic tubes for various industrial applications',
                'key_features': ['Chemical resistant', 'Lightweight', 'Cost-effective'],
                'applications': ['Chemical', 'Food', 'Pharmaceutical'],
                'price_range': '₹200-2000 per meter'
            },
            'Composite Tubes': {
                'description': 'Advanced composite tubes for high-performance applications',
                'key_features': ['High strength-to-weight ratio', 'Customizable', 'Advanced'],
                'applications': ['Defense', 'Sports', 'Aerospace'],
                'price_range': '₹5000-50000 per meter'
            },
            'Specialty Tubes': {
                'description': 'Custom-engineered tubes for specialized applications',
                'key_features': ['Custom design', 'Specialized materials', 'High precision'],
                'applications': ['Medical', 'Research', 'Defense'],
                'price_range': '₹10000-100000 per meter'
            }
        }
    
    def _initialize_templates(self):
        """Initialize personalization templates."""
        return {
            'enterprise': {
                'tone': 'Professional and strategic',
                'focus': 'ROI, scalability, and long-term partnership',
                'language': 'Business-focused with technical specifications'
            },
            'smb': {
                'tone': 'Friendly and solution-oriented',
                'focus': 'Cost-effectiveness, reliability, and support',
                'language': 'Clear benefits with practical examples'
            },
            'startup': {
                'tone': 'Innovative and encouraging',
                'focus': 'Growth potential, flexibility, and innovation',
                'language': 'Forward-thinking with growth opportunities'
            },
            'government': {
                'tone': 'Formal and compliant',
                'focus': 'Regulatory compliance, quality standards, and documentation',
                'language': 'Official with emphasis on standards'
            },
            'educational': {
                'tone': 'Educational and informative',
                'focus': 'Learning value, safety, and educational benefits',
                'language': 'Clear explanations with educational context'
            }
        }
    
    def generate_personalized_pitch(self, customer_data, selected_product=None):
        """
        Generate a personalized product pitch for a specific customer.
        
        Args:
            customer_data (dict): Customer information
            selected_product (str): Specific product to focus on
            
        Returns:
            dict: Personalized pitch and recommendations
        """
        
        # Analyze customer characteristics
        customer_profile = self._analyze_customer_profile(customer_data)
        
        # Select optimal product if none specified
        if not selected_product:
            selected_product = self._recommend_optimal_product(customer_profile)
        
        # Generate personalized pitch
        pitch = self._create_personalized_pitch(customer_profile, selected_product)
        
        # Generate supporting recommendations
        recommendations = self._generate_recommendations(customer_profile, selected_product)
        
        # Calculate potential revenue impact
        revenue_impact = self._calculate_revenue_impact(customer_profile, selected_product)
        
        return {
            'customer_profile': customer_profile,
            'recommended_product': selected_product,
            'personalized_pitch': pitch,
            'recommendations': recommendations,
            'revenue_impact': revenue_impact
        }
    
    def _analyze_customer_profile(self, customer_data):
        """Analyze customer profile for personalization."""
        
        profile = {
            'segment': customer_data['Segment'],
            'region': customer_data['Region'],
            'purchase_history': customer_data['PastPurchases'],
            'online_engagement': customer_data['WebsiteVisits'],
            'revenue_potential': customer_data['RevenuePotential'],
            'campaign_response': customer_data['ResponseToCampaign'],
            'competitor_awareness': customer_data['CompetitorMentions'] != 'None'
        }
        
        # Add behavioral insights
        profile['customer_type'] = self._classify_customer_type(profile)
        profile['engagement_level'] = self._classify_engagement_level(profile)
        profile['growth_potential'] = self._assess_growth_potential(profile)
        
        return profile
    
    def _classify_customer_type(self, profile):
        """Classify customer type based on characteristics."""
        
        if profile['segment'] == 'Enterprise' and profile['revenue_potential'] > 150:
            return 'High-Value Enterprise'
        elif profile['segment'] == 'SMB' and profile['revenue_potential'] > 75:
            return 'Growing SMB'
        elif profile['segment'] == 'Startup' and profile['online_engagement'] > 20:
            return 'Tech-Savvy Startup'
        elif profile['segment'] == 'Government':
            return 'Government Institution'
        elif profile['segment'] == 'Educational':
            return 'Educational Institution'
        else:
            return 'Standard Customer'
    
    def _classify_engagement_level(self, profile):
        """Classify customer engagement level."""
        
        if profile['online_engagement'] > 20 and profile['purchase_history'] > 5:
            return 'Highly Engaged'
        elif profile['online_engagement'] > 10 or profile['purchase_history'] > 2:
            return 'Moderately Engaged'
        else:
            return 'Low Engagement'
    
    def _assess_growth_potential(self, profile):
        """Assess customer growth potential."""
        
        if profile['segment'] == 'Startup' and profile['revenue_potential'] < 50:
            return 'High Growth Potential'
        elif profile['segment'] == 'SMB' and profile['revenue_potential'] < 100:
            return 'Medium Growth Potential'
        elif profile['segment'] == 'Enterprise' and profile['revenue_potential'] > 200:
            return 'Expansion Potential'
        else:
            return 'Stable Customer'
    
    def _recommend_optimal_product(self, customer_profile):
        """Recommend optimal product based on customer profile."""
        
        segment = customer_profile['segment']
        revenue_potential = customer_profile['revenue_potential']
        engagement_level = customer_profile['engagement_level']
        
        # Product recommendations by segment
        segment_recommendations = {
            'Enterprise': ['Steel Tubes', 'Composite Tubes', 'Specialty Tubes'],
            'SMB': ['Steel Tubes', 'Aluminum Tubes', 'Plastic Tubes'],
            'Startup': ['Plastic Tubes', 'Aluminum Tubes', 'Steel Tubes'],
            'Government': ['Steel Tubes', 'Copper Tubes', 'Specialty Tubes'],
            'Educational': ['Plastic Tubes', 'Aluminum Tubes', 'Steel Tubes']
        }
        
        # Consider revenue potential
        if revenue_potential > 200:
            return 'Specialty Tubes'  # High-value customers
        elif revenue_potential > 100:
            return 'Composite Tubes'  # Medium-high value
        elif revenue_potential > 50:
            return 'Copper Tubes'     # Medium value
        else:
            return 'Steel Tubes'      # Standard value
    
    def _create_personalized_pitch(self, customer_profile, product):
        """Create personalized product pitch."""
        
        segment = customer_profile['segment']
        template = self.personalization_templates.get(segment, self.personalization_templates['smb'])
        product_info = self.product_catalog[product]
        
        # Generate pitch based on customer characteristics
        if segment == 'Enterprise':
            pitch = f"""
**Strategic Partnership Opportunity: {product}**

Dear Valued Enterprise Partner,

Based on your impressive track record of {customer_profile['purchase_history']} successful projects and your strategic presence in {customer_profile['region']}, we believe {product} represents a perfect alignment with your growth objectives.

**Why {product} is Ideal for Your Operations:**
- **Scalability**: Designed to support enterprise-level operations with {product_info['key_features'][0].lower()} capabilities
- **ROI Focus**: {product_info['price_range']} pricing structure optimized for large-scale procurement
- **Strategic Value**: Long-term partnership potential with dedicated enterprise support

**Key Benefits:**
{chr(10).join([f"• {feature}" for feature in product_info['key_features']])}

**Applications in Your Industry:**
{chr(10).join([f"• {app}" for app in product_info['applications']])}

We're ready to discuss how {product} can drive your next phase of growth.
            """
        
        elif segment == 'SMB':
            pitch = f"""
**Growth Solution: {product} for Your Business**

Hello Business Leader,

Your {customer_profile['purchase_history']} successful projects and {customer_profile['online_engagement']} monthly website visits show you're ready for the next level. {product} is the solution you've been looking for.

**Why {product} Fits Your Business:**
- **Cost-Effective**: {product_info['price_range']} - designed for growing businesses
- **Reliable Quality**: {product_info['key_features'][0]} ensuring consistent performance
- **Expert Support**: Our team is here to help you succeed

**Perfect for Your Applications:**
{chr(10).join([f"• {app}" for app in product_info['applications'][:2]])}

**Immediate Benefits:**
{chr(10).join([f"• {feature}" for feature in product_info['key_features'][:2]])}

Let's discuss how {product} can accelerate your business growth.
            """
        
        else:
            pitch = f"""
**Innovation Opportunity: {product}**

Greetings,

Your forward-thinking approach and {customer_profile['engagement_level'].lower()} engagement style make you an ideal candidate for {product}.

**Why {product} is Perfect for You:**
- **Innovation**: {product_info['key_features'][0]} technology
- **Flexibility**: {product_info['price_range']} - adaptable to your needs
- **Future-Ready**: Designed for tomorrow's challenges

**Applications:**
{chr(10).join([f"• {app}" for app in product_info['applications'][:2]])}

**Features:**
{chr(10).join([f"• {feature}" for feature in product_info['key_features'][:2]])}

Ready to explore how {product} can transform your operations?
            """
        
        return pitch.strip()
    
    def _generate_recommendations(self, customer_profile, product):
        """Generate personalized recommendations."""
        
        recommendations = []
        
        # Engagement recommendations
        if customer_profile['engagement_level'] == 'Low Engagement':
            recommendations.append("**Engagement Strategy**: Implement targeted email campaigns and personalized content to increase website visits")
        
        # Product recommendations
        if customer_profile['revenue_potential'] > 100:
            recommendations.append(f"**Upsell Opportunity**: Consider {product} premium variants for enhanced performance")
        
        # Regional recommendations
        if customer_profile['region'] in ['North India', 'West India']:
            recommendations.append("**Regional Advantage**: Leverage our strong presence in {customer_profile['region']} for faster delivery and local support")
        
        # Competitor recommendations
        if customer_profile['competitor_awareness']:
            recommendations.append("**Competitive Edge**: Our {product} offers superior {self.product_catalog[product]['key_features'][0].lower()} compared to alternatives")
        
        # Growth recommendations
        if customer_profile['growth_potential'] == 'High Growth Potential':
            recommendations.append("**Growth Partnership**: Establish long-term supply agreements to support your expansion plans")
        
        return recommendations
    
    def _calculate_revenue_impact(self, customer_profile, product):
        """Calculate potential revenue impact."""
        
        base_revenue = customer_profile['revenue_potential']
        product_multiplier = {
            'Steel Tubes': 1.2,
            'Aluminum Tubes': 1.3,
            'Copper Tubes': 1.4,
            'Plastic Tubes': 1.1,
            'Composite Tubes': 1.6,
            'Specialty Tubes': 1.8
        }
        
        multiplier = product_multiplier.get(product, 1.2)
        potential_revenue = base_revenue * multiplier
        
        return {
            'current_revenue': base_revenue,
            'potential_revenue': potential_revenue,
            'revenue_increase': potential_revenue - base_revenue,
            'growth_percentage': ((potential_revenue - base_revenue) / base_revenue) * 100
        }
    
    def simulate_openai_generation(self, customer_data, product=None):
        """
        Simulate OpenAI API call for demo purposes.
        
        Args:
            customer_data (dict): Customer information
            product (str): Specific product to focus on
            
        Returns:
            dict: Generated personalization results
        """
        
        st.info("🤖 **GenAI Integration Note**: This is a simulated AI analysis. In production, this would use OpenAI's GPT-4 API to generate dynamic, personalized content.")
        
        # Generate personalized pitch
        personalization_results = self.generate_personalized_pitch(customer_data, product)
        
        return personalization_results
    
    def create_personalization_dashboard(self, personalization_results):
        """
        Create dashboard for personalization results.
        
        Args:
            personalization_results (dict): Results from generate_personalized_pitch
            
        Returns:
            plotly.graph_objects.Figure: Personalization dashboard
        """
        
        if not personalization_results:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Customer Profile Analysis',
                'Product Recommendation',
                'Revenue Impact Projection',
                'Personalization Insights'
            ]
        )
        
        profile = personalization_results['customer_profile']
        revenue_impact = personalization_results['revenue_impact']
        
        # 1. Customer Profile Analysis
        profile_metrics = ['Segment', 'Region', 'Purchase History', 'Online Engagement']
        profile_values = [
            profile['segment'],
            profile['region'],
            profile['purchase_history'],
            profile['online_engagement']
        ]
        
        fig.add_trace(
            go.Bar(
                x=profile_metrics,
                y=profile_values,
                name='Profile Metrics',
                marker_color='blue'
            ),
            row=1, col=1
        )
        
        # 2. Product Recommendation
        products = list(self.product_catalog.keys())
        product_scores = [np.random.uniform(0.6, 1.0) for _ in products]  # Simulated scores
        
        fig.add_trace(
            go.Bar(
                x=products,
                y=product_scores,
                name='Product Match Score',
                marker_color='green'
            ),
            row=1, col=2
        )
        
        # 3. Revenue Impact Projection
        revenue_data = [
            revenue_impact['current_revenue'],
            revenue_impact['potential_revenue']
        ]
        revenue_labels = ['Current', 'Potential']
        
        fig.add_trace(
            go.Bar(
                x=revenue_labels,
                y=revenue_data,
                name='Revenue (Lakhs)',
                marker_color='red'
            ),
            row=2, col=1
        )
        
        # 4. Personalization Insights
        insight_categories = ['Engagement', 'Growth', 'Competition', 'Regional']
        insight_scores = [0.8, 0.7, 0.6, 0.9]  # Simulated scores
        
        fig.add_trace(
            go.Bar(
                x=insight_categories,
                y=insight_scores,
                name='Insight Scores',
                marker_color='orange'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="AI Personalization Dashboard",
            showlegend=False
        )
        
        return fig
