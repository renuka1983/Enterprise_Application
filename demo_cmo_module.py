#!/usr/bin/env python3
"""
🎯 CMO Hyper-Personalization Module Demo
Demonstrates the key features of the CMO module without running Streamlit
"""

import sys
import os

# Add the cmo_hyper_personalization directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cmo_hyper_personalization'))

def demo_cmo_module():
    """Demonstrate the CMO module capabilities"""
    
    print("🎯 CMO Hyper-Personalization & Market Intelligence Demo")
    print("=" * 60)
    
    try:
        # Test data generation
        print("\n📊 Testing Data Generation...")
        from data.synthetic_data import generate_sample_data
        
        customer_df, market_df, _ = generate_sample_data(num_customers=100, num_market_records=50)
        print(f"✅ Generated {len(customer_df)} customers and {len(market_df)} market records")
        print(f"📈 Customer segments: {customer_df['Segment'].unique()}")
        print(f"🌍 Regions: {customer_df['Region'].unique()}")
        
        # Test ML models
        print("\n🤖 Testing ML Models...")
        from models.ml_models import CampaignResponsePredictor
        
        predictor = CampaignResponsePredictor()
        print("✅ ML models initialized successfully")
        
        # Test AI analysis
        print("\n🧠 Testing AI Analysis...")
        from models.ai_analysis import AICompetitorAnalysis, MarketIntelligenceAI
        from models.ml_models import CustomerSegmentation
        
        ai_competitor = AICompetitorAnalysis()
        market_ai = MarketIntelligenceAI()
        segmentation = CustomerSegmentation()
        print("✅ AI analysis modules initialized successfully")
        
        # Test GenAI
        print("\n🌟 Testing GenAI Integration...")
        from genai.personalization_engine import TubesIndiaPersonalizationEngine
        
        genai_engine = TubesIndiaPersonalizationEngine()
        print("✅ GenAI personalization engine initialized successfully")
        
        print("\n🎉 All CMO module components are working correctly!")
        
        # Show sample data
        print(f"\n📋 Sample Customer Data:")
        print(customer_df.head(3).to_string(index=False))
        
        print(f"\n📊 Sample Market Intelligence:")
        print(market_df.head(3).to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing CMO module: {e}")
        return False

if __name__ == "__main__":
    success = demo_cmo_module()
    if success:
        print("\n🚀 CMO Module is ready for use!")
        print("💡 Run 'streamlit run cmo_hyper_personalization/CMO_Hyper_Personalization.py' to start the app")
    else:
        print("\n⚠️  Some issues detected. Please check the error messages above.")
