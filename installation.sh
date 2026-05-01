#!/bin/bash
echo "Installing required packages..."
pip install -r requirements.txt
echo ""
echo "Running the project..."
python3 churn_prediction_project.py
