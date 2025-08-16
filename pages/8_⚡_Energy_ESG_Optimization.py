import streamlit as st
import sys
import os

# Add the energy_esg_optimization directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
energy_esg_dir = os.path.join(current_dir, '..', 'energy_esg_optimization')

# Ensure the path exists and add it to sys.path
if os.path.exists(energy_esg_dir):
    sys.path.insert(0, energy_esg_dir)
    st.info(f"✅ Added Energy ESG module path: {energy_esg_dir}")
else:
    st.error(f"❌ Energy ESG module directory not found at: {energy_esg_dir}")
    st.stop()

try:
    # Import and run the Energy ESG optimization app
    from Energy_ESG_Optimization import main
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    st.error(f"❌ Import Error: {str(e)}")
    st.error("Please check that all required modules are available.")
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected Error: {str(e)}")
    st.stop()
