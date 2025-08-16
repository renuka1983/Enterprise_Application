import streamlit as st
import sys
import os

# Add the cfo_case_study directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
cfo_dir = os.path.join(current_dir, '..', 'cfo_case_study')

# Ensure the path exists and add it to sys.path
if os.path.exists(cfo_dir):
    sys.path.insert(0, cfo_dir)
    st.info(f"✅ Added CFO module path: {cfo_dir}")
else:
    st.error(f"❌ CFO module directory not found at: {cfo_dir}")
    st.stop()

try:
    # Import and run the CFO case study app
    from CFO_Case_Study import main
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    st.error(f"❌ Import Error: {str(e)}")
    st.error("Please check that all required modules are available.")
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected Error: {str(e)}")
    st.stop()
