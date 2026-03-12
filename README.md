Project Overview

A Machine Learning-based system designed to predict the outcomes of English Premier League matches. This project analyzes historical data from the last 15 seasons (over 5,600 matches) to identify patterns and team momentum, aiming to provide statistically-backed predictions for match winners.

Tech Stack
Language: Python
Data Analysis: Pandas (Data Cleaning, Feature Engineering, Rolling Averages)
Machine Learning: Scikit-Learn (Random Forest Classifier)
Environment: Jupyter Notebook / PyCharm

Key Features

Massive Data Aggregation: Successfully merged and normalized data from 15 different seasons, handling inconsistent date formats and missing columns across over a decade of records.
Feature Engineering (Momentum Analysis): Developed a custom "Rolling Averages" system to calculate team performance over their last 3 matches (e.g., average goals scored/conceded) to capture current form.
Robust ML Modeling: Implemented a Random Forest Classifier to handle categorical data and mitigate overfitting, ensuring the model generalizes well to unseen future matches.
Advanced Preprocessing: Created automated pipelines to handle missing values (NaNs), convert categorical team names into numerical codes, and extract temporal features like match hour and day of the week.

Performance Metrics

The model currently achieves a Precision Score of approximately 49% for Home Win predictions.
Note: In the Premier League, the home team wins roughly 45% of the time. This model demonstrates a statistical edge by outperforming the baseline "home-win" bias through data-driven insights.

Future Roadmap

Web UI: Developing a Flask or Streamlit web application to allow users to select teams and view live predictions.
Enhanced Features: Integrating player injury data and ELO ratings to further improve accuracy.
OOP Refactoring: Transitioning the codebase to a fully Object-Oriented structure for better scalability.
