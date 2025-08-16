import streamlit as st
import sys
import os

# Add the energy_esg_optimization directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
energy_esg_dir = os.path.join(current_dir, '..', 'energy_esg_optimization')
sys.path.append(energy_esg_dir)

# Import and run the Energy ESG optimization app
from Energy_ESG_Optimization import main

if __name__ == "__main__":
    main()
