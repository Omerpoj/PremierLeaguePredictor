import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score
import glob
print("ready")
PREDICTORS = ["home_code", "away_code", "day_code", "hour",
              "home_rolling_goals", "away_rolling_goals",
              "home_rolling_conceded", "away_rolling_conceded"]
split_date = pd.to_datetime("2025-01-01")
def preprocess(df):
    """"cleans and preprocess matches_data for the module"""
    #makes a copy to work on in order to not overload on the memory
    df = df.copy()
    #strips the dates
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True,format="mixed")
    # making a code for every team
    df["home_code"] = df["HomeTeam"].astype("category").cat.codes
    df["away_code"] = df["AwayTeam"].astype("category").cat.codes
    # takes the date and hour cleanly
    df["day_code"] = df["Date"].dt.dayofweek
    if "Time" in df.columns:
        # strips the hour and turns it into an integer
        df["hour"] = df["Time"].str.replace(":.+", "", regex=True).fillna(15).astype(int)
    else:
        # default hour typical for English football
        df["hour"] = 15
    df["target"] = (df["FTR"] == "H").astype(int)
    return df

def add_rolling_averages(df):
    """calculating the rolling average per team in the last 3 games"""
    #making a copy to work with
    df = df.copy()
    #sorting the games by date in order to calculate the momentum
    df = df.sort_values("Date")
    #creating a mini table for every team and calculating the goals/conceded goals from the last three games
    df["home_rolling_goals"] = df.groupby("HomeTeam")["FTHG"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["home_rolling_conceded"] = df.groupby("HomeTeam")["FTAG"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["away_rolling_goals"] = df.groupby("AwayTeam")["FTAG"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["away_rolling_conceded"] = df.groupby("AwayTeam")["FTHG"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    #filling the none values in the rolling columns
    rolling_cols = ["home_rolling_goals", "home_rolling_conceded", "away_rolling_goals", "away_rolling_conceded"]
    df[rolling_cols] = df[rolling_cols].fillna(0)
    return df



def make_prediction(data, predictors):
    #train and test partition
    train = data[data["Date"] < split_date]
    test = data[data["Date"] >= split_date]

    # module definition
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

    # testing and prediction
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])

    # initialization of a table that shows the result and the truth
    combined = pd.DataFrame(dict(actual=test["target"], prediction=preds), index=test.index)

    return combined, precision_score(test["target"], preds)


#loading and concatenating the matches_data
file_paths = glob.glob("matches_data/*.csv")
all_seasons = [pd.read_csv(file) for file in file_paths]
df_combined = pd.concat(all_seasons, ignore_index=True)
#checking if the loading was successful
print(f"Loaded {len(file_paths)} files successfully!")
print(f"Total matches in dataset: {len(df_combined)}")

#processing the matches_data
df_preprocesed = preprocess(df_combined)

#adding the rolling stats
df_with_rolling_averages = add_rolling_averages(df_preprocesed)

#predict and results
results, precision = make_prediction(df_with_rolling_averages, PREDICTORS)
print(f"The module precision : {precision:.2%}")
print("\nThe results (Actual vs Prediction):")
print(results.head())


