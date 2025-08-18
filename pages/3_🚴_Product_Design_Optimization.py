import streamlit as st
import sys
import os

# Add the product_design directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
product_design_dir = os.path.join(current_dir, '..', 'product_design')

if os.path.exists(product_design_dir):
    sys.path.insert(0, product_design_dir)
    st.info(f"✅ Added Product Design module path: {product_design_dir}")
else:
    st.error(f"❌ Product Design module directory not found at: {product_design_dir}")
    st.stop()

try:
    from design_optimization import main
    if __name__ == "__main__":
        main()
except ImportError as e:
    st.error(f"❌ Import Error: {str(e)}")
    st.error("Please check that all required modules are available.")
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected Error: {str(e)}")
    st.stop()
