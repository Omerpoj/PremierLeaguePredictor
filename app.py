from flask import Flask, render_template, request
from predictor import PremierLeaguePredictor

app = Flask(__name__)

# ==========================================
# 1. הפעלת מנוע ה-AI
# ==========================================
print("⏳ Initializing Machine Learning Engine...")
ml_system = PremierLeaguePredictor()
ml_system.load_and_prepare_data()
ml_system.make_prediction()
print("✅ Model is loaded and ready for web requests!")


# ==========================================
# 2. ניתוב (Route) לעמוד הבית וקבלת נתונים
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def home():
    # מושכים את רשימת הקבוצות מהמילון של המודל כדי להציג בבחירה
    teams = list(ml_system.team_mapping.keys())
    result_text = None

    # אם המשתמש לחץ על כפתור "Predict Match"
    if request.method == 'POST':
        home_team = request.form.get('home_team')
        away_team = request.form.get('away_team')

        # מוודאים שלא נבחרו אותן קבוצות
        if home_team == away_team:
            result_text = "⚠️ You must select two different teams!"
        else:
            # מריצים את החיזוי דרך המודל שלנו
            prediction = ml_system.predict_single_match(home_team, away_team)

            if prediction == 1:
                result_text = f"🏆 Prediction: {home_team} will WIN!"
            else:
                result_text = f"⚖️ Prediction: Draw or {away_team} will WIN."

    # שולחים את הנתונים לקובץ ה-HTML כדי שיצייר את המסך
    return render_template('index.html', teams=teams, result_text=result_text)


if __name__ == '__main__':
    app.run(debug=True, port=5000)