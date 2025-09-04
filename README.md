🏙️ Delhi Air Quality Time Series Analysis

This project explores and analyzes Delhi’s air quality data using Python. The focus is on time series analysis of PM2.5 levels (fine particulate matter) along with visualizations by month and other temporal patterns.

📂 Dataset

Source: Delhi Air Quality Dataset (CSV format)

Columns used:

date → Date of recording

pm2_5 → PM2.5 concentration (µg/m³)

Other pollutants like co, no, no2, o3, so2, pm10, nh3 were dropped for this project to keep the analysis focused on PM2.5.

⚙️ Project Workflow

Wrangling:

Load CSV file

Drop unnecessary pollutant columns

Convert date to datetime

Extract month and month_name

Aggregation:

Group data by month

Calculate average monthly PM2.5

Visualization:

Line Plot: Trends of PM2.5 across months

Bar Plot: Average PM2.5 concentration by month

(Can extend with seasonal decomposition, forecasting, etc.)

📊 Sample Visualization

Monthly average PM2.5 levels in Delhi:

🚀 Tech Stack

Python

Pandas (data wrangling)

Matplotlib & Seaborn (visualization)

Streamlit (interactive app)

🌍 Motivation

Delhi consistently ranks among the most polluted cities in the world. By analyzing time series air quality data, we aim to:

Identify seasonal trends

Provide insights for policymakers

Raise public awareness about air pollution 
