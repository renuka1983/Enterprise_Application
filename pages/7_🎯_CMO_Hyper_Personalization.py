import streamlit as st
import sys
import os

# Add the cmo_hyper_personalization directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cmo_hyper_personalization'))

# Import and run the CMO hyper-personalization app
from CMO_Hyper_Personalization import main

if __name__ == "__main__":
    main()
