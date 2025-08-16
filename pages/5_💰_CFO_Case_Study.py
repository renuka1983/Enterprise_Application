import streamlit as st
import sys
import os

# Add the cfo_case_study directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cfo_case_study'))

# Import and run the CFO case study app
from CFO_Case_Study import main

if __name__ == "__main__":
    main()
