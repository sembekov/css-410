@echo off
echo Installing required packages...
pip install -r requirements.txt
echo.
echo Running the project...
python churn_prediction_project.py
pause
