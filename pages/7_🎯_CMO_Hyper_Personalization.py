import streamlit as st
import sys
import os

# Add the cmo_hyper_personalization directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
cmo_dir = os.path.join(current_dir, '..', 'cmo_hyper_personalization')

# Ensure the path exists and add it to sys.path
if os.path.exists(cmo_dir):
    sys.path.insert(0, cmo_dir)
    st.info(f"✅ Added CMO module path: {cmo_dir}")
else:
    st.error(f"❌ CMO module directory not found at: {cmo_dir}")
    st.stop()

try:
    # Import and run the CMO hyper-personalization app
    from CMO_Hyper_Personalization import main
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    st.error(f"❌ Import Error: {str(e)}")
    st.error("Please check that all required modules are available.")
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected Error: {str(e)}")
    st.stop()
