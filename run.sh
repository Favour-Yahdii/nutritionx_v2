#!/bin/bash

# Start the FastAPI server in the background
uvicorn main_test:app --reload &

# Start the Streamlit application
streamlit run app_test.py
