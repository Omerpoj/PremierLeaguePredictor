from flask import Flask, render_template, request, jsonify
from predictor import PremierLeaguePredictor

app = Flask(__name__)

# --- AI Engine Initialization ---
# The model is loaded and trained once when the server starts
print("⏳ Initializing Machine Learning Engine...")
ml_system = PremierLeaguePredictor()
#updates to the latest data
ml_system.update_latest_data()
ml_system.load_and_prepare_data()
ml_system.make_prediction()
print("✅ Model is loaded and ready for web requests!")


# --- Routes ---

@app.route('/', methods=['GET'])
def home():
    """Renders the initial HTML page with the list of available teams."""
    teams = list(ml_system.team_mapping.keys())
    return render_template('index.html', teams=teams)


@app.route('/analytics', methods=['GET'])
def analytics():
    """Renders the analytics page showing the inner workings of the ML model."""
    feature_data = ml_system.get_feature_importance()

    # styling the percents
    model_precision = round(ml_system.precision * 100, 1)

    # sends the data to the HTML
    return render_template('analytics.html', feature_data=feature_data, model_precision=model_precision)


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

    # Get form data for both teams
    home_form = ml_system.get_team_form(home_team)
    away_form = ml_system.get_team_form(away_team)

    # Helper function to generate HTML circles for form
    def format_form_html(form_list):
        colors = {"W": "#28a745", "D": "#6c757d", "L": "#dc3545"}
        html = '<div style="display: flex; justify-content: center; gap: 5px; margin-top: 5px;">'
        for res in form_list:
            html += f'<span style="background:{colors[res]}; color:white; border-radius:50%; width:20px; height:20px; font-size:12px; display:flex; align-items:center; justify-content:center; font-weight:bold;">{res}</span>'
        html += '</div>'
        return html

    # Execute the ML model prediction
    try:
        # gets the result and the probabilities from the module
        prediction, prob_h, prob_d, prob_a = ml_system.predict_single_match(home_team, away_team)

        # turns the probabilities into percent
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

        home_form_html = format_form_html(home_form)
        away_form_html = format_form_html(away_form)

        # building the text for the HTML
        result_html = f"""
                <div style="font-size: 22px; margin-bottom: 15px; font-weight: bold;">{winner_text}</div>

                <div style="display: flex; justify-content: space-around; gap: 20px;">
                    <div style="flex: 1;">
                        <strong>{home_team}</strong>
                        {home_form_html}
                        <div style="margin-top: 10px;">
                            <span style="font-size: 18px;">{pct_h}%</span>
                            <div class="prob-bar-container"><div class="prob-bar-fill home-fill" style="width: {pct_h}%"></div></div>
                        </div>
                    </div>

                    <div style="flex: 1; display: flex; flex-direction: column; justify-content: flex-end;">
                        <strong>Draw</strong><br>
                        <div style="margin-top: 10px;">
                            <span style="font-size: 18px;">{pct_d}%</span>
                            <div class="prob-bar-container"><div class="prob-bar-fill draw-fill" style="width: {pct_d}%"></div></div>
                        </div>
                    </div>

                    <div style="flex: 1;">
                        <strong>{away_team}</strong>
                        {away_form_html}
                        <div style="margin-top: 10px;">
                            <span style="font-size: 18px;">{pct_a}%</span>
                            <div class="prob-bar-container"><div class="prob-bar-fill away-fill" style="width: {pct_a}%"></div></div>
                        </div>
                    </div>
                </div>
                """

        return jsonify({'result': result_html})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Internal model error'}), 500


if __name__ == '__main__':
    # Running on local development server
    app.run(debug=True, port=5001)