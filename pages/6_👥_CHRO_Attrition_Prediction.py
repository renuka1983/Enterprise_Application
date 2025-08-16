import streamlit as st
import sys
import os

# Add the chro_attrition_prediction directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'chro_attrition_prediction'))

# Import and run the CHRO attrition prediction app
from CHRO_Attrition_Prediction import main

if __name__ == "__main__":
    main()
