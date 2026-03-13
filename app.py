from flask import Flask, render_template, request, jsonify
from predictor import PremierLeaguePredictor

app = Flask(__name__)

# --- AI Engine Initialization ---
# The model is loaded and trained once when the server starts
print("⏳ Initializing Machine Learning Engine...")
ml_system = PremierLeaguePredictor()
ml_system.load_and_prepare_data()
ml_system.make_prediction()
print("✅ Model is loaded and ready for web requests!")


# --- Routes ---

@app.route('/', methods=['GET'])
def home():
    """Renders the initial HTML page with the list of available teams."""
    teams = list(ml_system.team_mapping.keys())
    return render_template('index.html', teams=teams)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Asynchronous API endpoint.
    Receives JSON data from JavaScript and returns the prediction result.
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided!'}), 400

    home_team = data.get('home_team')
    away_team = data.get('away_team')

    if not home_team or not away_team:
        return jsonify({'error': 'Please select two teams!'}), 400

    # Logic check: ensuring teams are unique
    if home_team == away_team:
        return jsonify({'error': '⚠️ You must select two different teams!'})

    # Execute the ML model prediction
    try:
        prediction = ml_system.predict_single_match(home_team, away_team)

        if prediction == 1:
            result_text = f"🏆 Prediction: {home_team} will WIN!"
        else:
            result_text = f"⚖️ Prediction: Draw or {away_team} will WIN."

        return jsonify({'result': result_text})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Internal model error'}), 500



if __name__ == '__main__':
    # Running on local development server
    app.run(debug=True, port=5000)