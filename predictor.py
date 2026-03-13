import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import glob


class PremierLeaguePredictor:
    def __init__(self, data_path="matches_data/*.csv"):
        """sys innit definition of files path, attributes and the module"""
        self.data_path = data_path
        self.predictors = ["home_code", "away_code", "day_code", "hour",
                           "home_rolling_goals", "home_rolling_conceded",
                           "away_rolling_goals", "away_rolling_conceded"]

        # module definition
        self.model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

        # variables to store the data and the team encodings
        self.data = None
        self.team_mapping = {}


    def update_latest_data(self):
        """Downloads the latest match data directly from the web to keep the model updated."""
        print("📥 Fetching latest Premier League results from the web...")

        #The direct link to the current 25/26 season file
        live_csv_url = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"

        #to check that the name matches with the local file
        local_save_path = "matches_data/Prem25_26.csv"

        try:
            #reading straight from the url
            df_latest = pd.read_csv(live_csv_url)
            #saving the new data
            df_latest.to_csv(local_save_path, index=False)
            print("✅ Successfully updated the current season data!")

        except Exception as e:
            #if there is no connection use the local data
            print(f"⚠️ Could not fetch live data (using existing files). Error: {e}")


    def load_and_prepare_data(self):
        """loading and concatenating the matches_data and applying all preprocessing"""
        file_paths = glob.glob(self.data_path)
        all_seasons = [pd.read_csv(file) for file in file_paths]
        df_combined = pd.concat(all_seasons, ignore_index=True)

        # checking if the loading was successful
        print(f"Loaded {len(file_paths)} files successfully!")
        print(f"Total matches in dataset: {len(df_combined)}")

        # processing the matches_data
        df_preprocessed = self._preprocess(df_combined)

        # adding the rolling stats and saving to self.data
        self.data = self._add_rolling_averages(df_preprocessed)

    def _preprocess(self, df):
        """"cleans and preprocess matches_data for the module"""
        # makes a copy to work on in order to not overload on the memory
        df = df.copy()

        # strips the dates
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")

        # [CRITICAL FIX]: making a dictionary code for every team
        all_teams_series = (pd.concat([df["HomeTeam"], df["AwayTeam"]]))
        all_teams = all_teams_series.dropna().unique()
        all_teams = sorted(str(team) for team in all_teams)
        self.team_mapping = {team: index for index, team in enumerate(all_teams)}

        df["home_code"] = df["HomeTeam"].map(self.team_mapping)
        df["away_code"] = df["AwayTeam"].map(self.team_mapping)

        # takes the date and hour cleanly
        df["day_code"] = df["Date"].dt.dayofweek
        if "Time" in df.columns:
            # strips the hour and turns it into an integer
            df["hour"] = df["Time"].str.replace(":.+", "", regex=True).fillna(15).astype(int)
        else:
            # default hour typical for English football
            df["hour"] = 15

        df["target"] = df["FTR"]
        return df

    def _add_rolling_averages(self, df):
        """calculating the rolling average per team in the last 3 games"""
        # making a copy to work with
        df = df.copy()
        # sorting the games by date in order to calculate the momentum
        df = df.sort_values("Date")

        # creating a mini table for every team and calculating the goals/conceded goals from the last three games
        df["home_rolling_goals"] = df.groupby("HomeTeam")["FTHG"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        df["home_rolling_conceded"] = df.groupby("HomeTeam")["FTAG"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        df["away_rolling_goals"] = df.groupby("AwayTeam")["FTAG"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        df["away_rolling_conceded"] = df.groupby("AwayTeam")["FTHG"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean())

        # filling the none values in the rolling columns
        rolling_cols = ["home_rolling_goals", "home_rolling_conceded", "away_rolling_goals", "away_rolling_conceded"]
        df[rolling_cols] = df[rolling_cols].fillna(0)
        return df

    def make_prediction(self, split_date_str="2025-01-01"):
        """trains the model on historical data and tests it"""
        split_date = pd.to_datetime(split_date_str)

        # train and test partition
        train = self.data[self.data["Date"] < split_date]
        test = self.data[self.data["Date"] >= split_date]

        # testing and prediction
        self.model.fit(train[self.predictors], train["target"])
        preds = self.model.predict(test[self.predictors])

        # initialization of a table that shows the result and the truth
        combined = pd.DataFrame(dict(actual=test["target"], prediction=preds), index=test.index)

        return combined, precision_score(test["target"], preds, average='macro', zero_division=0)

    def get_team_form(self, team_name):
        """Returns the last 5 match results for a team as a list of 'W', 'D', 'L'"""
        # Filter matches where the team played as home or away
        team_matches = self.data[(self.data["HomeTeam"] == team_name) |
                                 (self.data["AwayTeam"] == team_name)].sort_values("Date")

        last_5 = team_matches.tail(5)
        form = []

        for _, row in last_5.iterrows():
            if row["FTR"] == "D":
                form.append("D")
            elif (row["HomeTeam"] == team_name and row["FTR"] == "H") or \
                    (row["AwayTeam"] == team_name and row["FTR"] == "A"):
                form.append("W")
            else:
                form.append("L")
        return form

    def predict_single_match(self, home_team, away_team, day_code=5, hour=15):
        """
        Predicts the outcome of a specific match between two teams based on their current momentum.
        """
        if home_team not in self.team_mapping or away_team not in self.team_mapping:
            return "Error: Team name not found in the database. Please check spelling."

        h_code = self.team_mapping[home_team]
        a_code = self.team_mapping[away_team]

        home_games = self.data[self.data["HomeTeam"] == home_team]
        latest_home = home_games.iloc[-1]
        h_goals = latest_home["home_rolling_goals"]
        h_conceded = latest_home["home_rolling_conceded"]

        away_games = self.data[self.data["AwayTeam"] == away_team]
        latest_away = away_games.iloc[-1]
        a_goals = latest_away["away_rolling_goals"]
        a_conceded = latest_away["away_rolling_conceded"]

        match_features = pd.DataFrame([{
            "home_code": h_code,
            "away_code": a_code,
            "day_code": day_code,
            "hour": hour,
            "home_rolling_goals": h_goals,
            "home_rolling_conceded": h_conceded,
            "away_rolling_goals": a_goals,
            "away_rolling_conceded": a_conceded
        }])

        prediction = self.model.predict(match_features)[0]
        probabilities = self.model.predict_proba(match_features)[0]

        classes = list(self.model.classes_)
        away_win_chance = probabilities[classes.index('A')]
        draw_chance = probabilities[classes.index('D')]
        home_win_chance = probabilities[classes.index('H')]

        # return the result and the probabilities
        return prediction, home_win_chance, draw_chance, away_win_chance


# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    predictor = PremierLeaguePredictor()

    # data loading and module training
    predictor.load_and_prepare_data()
    results, precision = predictor.make_prediction()
    print(f"Overall Model Precision: {precision:.2%}")

    # examples for single games predictions
    predictor.predict_single_match("Arsenal", "Chelsea")
    predictor.predict_single_match("Liverpool", "Everton")