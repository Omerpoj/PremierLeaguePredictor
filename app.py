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
        # gets the result and the probabilities from the module
        prediction, prob_h, prob_d, prob_a = ml_system.predict_single_match(home_team, away_team)

        # turns the probabilities into precent
        pct_h = round(prob_h * 100, 1)
        pct_d = round(prob_d * 100, 1)
        pct_a = round(prob_a * 100, 1)

        # styling the text
        if prediction == 'H':
            winner_text = f"🏆 {home_team} will WIN!"
        elif prediction == 'A':
            winner_text = f"🏆 {away_team} will WIN!"
        else:
            winner_text = "⚖️ It's going to be a DRAW!"

        # building the text for the HTML
        result_html = f"""
        <div style="font-size: 22px; margin-bottom: 15px;">{winner_text}</div>
        <hr style="border: 0; height: 1px; background: #e0e0e0; margin: 15px 0;">
        <div style="display: flex; justify-content: space-around; font-size: 16px; color: #555;">
            <div><strong>{home_team}</strong><br><span style="color: #28a745; font-size: 20px;">{pct_h}%</span></div>
            <div><strong>Draw</strong><br><span style="color: #6c757d; font-size: 20px;">{pct_d}%</span></div>
            <div><strong>{away_team}</strong><br><span style="color: #dc3545; font-size: 20px;">{pct_a}%</span></div>
        </div>
        """

        return jsonify({'result': result_html})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Internal model error'}), 500


if __name__ == '__main__':
    # Running on local development server
    app.run(debug=True, port=5000)