# ⚽ Premier League Match Predictor (AI-Powered)

An end-to-end Machine Learning web application that predicts the outcome of English Premier League matches using historical data and team momentum.

## 🚀 Live Demo
Check out the live app here: [YOUR-RENDER-URL-HERE]

## 🧠 How it Works
The system uses a **Random Forest Classifier** trained on over 5,000 Premier League matches. It calculates **rolling averages** for goals scored and conceded over the last 3 matches to capture current team form (momentum).

### Key Features:
- **ML Engine:** Built with Scikit-learn and Pandas.
- **Asynchronous UI:** Responsive interface using JavaScript Fetch API (No page reloads).
- **Dynamic Visuals:** Real-time team logo updates and probability momentum bars.
- **Three-Way Prediction:** Calculates Win/Draw/Loss probabilities.

## 🛠️ Technical Stack
- **Backend:** Python (Flask), Gunicorn
- **Machine Learning:** Scikit-learn, Pandas, NumPy
- **Frontend:** HTML5, CSS3 (Flexbox), JavaScript (ES6+)
- **Deployment:** Render, GitHub Actions

## 📂 Project Structure
- `app.py`: Flask server and API endpoints.
- `predictor.py`: The core ML class for data processing and prediction.
- `matches_data/`: Historical match data (CSV).
- `static/`: Frontend assets (CSS, JS, Team Logos).

## 🎓 About the Developer
Developed as a personal project by **Omer Pojejinsky**, a Computer Science student at Reichman University. This project demonstrates skills in data engineering, model deployment, and full-stack web development.