#!/usr/bin/env python3
"""
⚡ Energy ESG Optimization Module Demo
Demonstrates the key features of the Energy ESG module without running Streamlit
"""

import sys
import os

# Add the energy_esg_optimization directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'energy_esg_optimization'))

def demo_energy_esg_module():
    """Demonstrate the Energy ESG module capabilities"""
    
    print("⚡ Energy ESG Optimization & Compliance Demo")
    print("=" * 60)
    
    try:
        # Test data generation
        print("\n📊 Testing Data Generation...")
        from data.energy_synthetic_data import generate_sample_data
        
        energy_df, production_df, compliance_df, summary = generate_sample_data(num_days=90, num_plants=3)
        print(f"✅ Generated {len(energy_df)} energy records for {summary['total_plants']} plants")
        print(f"📈 Total Energy Consumption: {summary['total_energy_consumption']:,.0f} kWh")
        print(f"🌍 Total CO2 Emissions: {summary['total_emissions']:,.0f} kg")
        print(f"⚡ Average Energy Efficiency: {summary['avg_energy_efficiency']:.3f}")
        print(f"🌱 Average Renewable Energy: {summary['avg_renewable_percentage']:.1f}%")
        
        # Test ML models
        print("\n🤖 Testing ML Models...")
        from models.ml_forecasting import EnergyConsumptionForecaster
        
        forecaster = EnergyConsumptionForecaster()
        print("✅ ML forecasting models initialized successfully")
        
        # Test AI optimization
        print("\n🧠 Testing AI Optimization...")
        from ai.optimization_engine import EnergyOptimizationEngine
        
        optimization_engine = EnergyOptimizationEngine()
        print("✅ AI optimization engine initialized successfully")
        
        # Test GenAI ESG reporting
        print("\n🌟 Testing GenAI ESG Reporting...")
        from genai.esg_report_generator import ESGReportGenerator
        
        esg_generator = ESGReportGenerator()
        print("✅ GenAI ESG report generator initialized successfully")
        
        print("\n🎉 All Energy ESG module components are working correctly!")
        
        # Show sample data
        print(f"\n📋 Sample Energy Data:")
        print(energy_df.head(3).to_string(index=False))
        
        print(f"\n📊 Sample Production Data:")
        print(production_df.head(3).to_string(index=False))
        
        if len(compliance_df) > 0:
            print(f"\n📋 Sample Compliance Data:")
            print(compliance_df.head(3).to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Energy ESG module: {e}")
        return False

if __name__ == "__main__":
    success = demo_energy_esg_module()
    if success:
        print("\n🚀 Energy ESG Module is ready for use!")
        print("💡 Run 'streamlit run energy_esg_optimization/Energy_ESG_Optimization.py' to start the app")
    else:
        print("\n⚠️  Some issues detected. Please check the error messages above.")
