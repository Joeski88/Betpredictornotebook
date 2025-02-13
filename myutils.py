import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

save_path = "./model/"
# Set which features to focus on
features = [
    'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST',
    'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 
    'AvgH', 'AvgD', 'AvgA',
    'GoalDifference', 'MarketConsensus',
    'FTR'
]

def feature_engineering(data):
    data['GoalDifference'] = data['FTHG'] - data['FTAG']
    data['MarketConsensus'] = (data['AvgH'] + data['AvgD'] + data['AvgA']) / 3
    # Set which features to focus on

    # Map team names to numeric IDs
    team_mapping = {team: idx for idx, team in enumerate(data['HomeTeam'].unique())}
    data['HomeTeam'] = data['HomeTeam'].map(team_mapping)
    data['AwayTeam'] = data['AwayTeam'].map(team_mapping)

    # Ensure 'FTR' is encoded as numeric
    ftr_mapping = {'H': 0, 'D': 1, 'A': 2}
    data['FTR'] = data['FTR'].map(ftr_mapping)

    # Verify data is numeric
    for column in ['HomeTeam', 'AwayTeam', 'FTR']:
        if not np.issubdtype(data[column].dtype, np.number):
            raise ValueError(f"Column '{column}' is not numeric after encoding.")

    return data, team_mapping, ftr_mapping

def plot_metrics(df, team1, team2=None, metrics =  ['FTHG', 'FTAG', 'HS', 'HST', 'HC', 'AS', 'AST', 'AC'], mapping={}, ax=None):
    # Example dataset (replace with your actual dataset)
    cols = ['HomeTeam']
    cols.extend(metrics)
    
    # Aggregate stats by team
    stats = df[cols].groupby('HomeTeam').mean()
    
    # Select teams and stats to compare
    
    if team2 == None:
        teams = [team1]
    else:
        teams = [team1, team2]

    stats_to_plot = cols[1:] # ignore HomeTeam
    
    # Normalize data to a range of [0, 1]
    normalized_stats = stats[stats_to_plot].apply(lambda x: x / x.max(), axis=0)
    
    # Radar plot setup
    num_vars = len(stats_to_plot)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    if ax == None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    else:
        fig = ax.get_figure()
        
    for team in teams:
        if isinstance(team, pd.DataFrame):
            team = team.iloc[0, 0]

        print("Normalised:", normalized_stats)
        values = normalized_stats.loc[team].values.flatten().tolist()

        values += values[:1]  # Complete the circle
        ax.plot(angles, values, label=team, linewidth=2)
        ax.fill(angles, values, alpha=0.25)
    
    # Add labels and gridlines
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], color='gray', fontsize=10)
    ax.set_xticks(angles[:-1])
    
    labels = [mapping[s] for s in stats_to_plot]
        
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_title("Team Performance Comparison", size=14, pad=20)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    return fig, ax

def calculateOverallPerformance(data):
    temp = data.copy()
    # Calculate wins for home and away
    home_wins = temp[temp['FTR'] == 'H'].groupby('HomeTeam').size()
    away_wins = temp[temp['FTR'] == 'A'].groupby('AwayTeam').size()

    # Calculate total matches for home and away
    home_matches = temp.groupby('HomeTeam').size()
    away_matches = temp.groupby('AwayTeam').size()

    # Aggregate wins and matches
    total_wins = home_wins.add(away_wins, fill_value=0)           # sum of home wins + away wins
    total_matches = home_matches.add(away_matches, fill_value=0)  # sum of home matches + away matches

    # Calculate win percentage
    win_percentage = (total_wins / total_matches) * 100

    # Sort teams by descending win percentage
    win_percentage = win_percentage.sort_values(ascending=False)

    win_percentage.plot(kind='bar', color='skyblue', alpha=0.8, edgecolor='black')

    temp['HomeTeamWinPct'] = temp['HomeTeam'].map(win_percentage)
    temp['AwayTeamWinPct'] = temp['AwayTeam'].map(win_percentage)

    return temp, win_percentage

def getParallelPlot(data, features, teams_to_plot):
    # Group by HomeTeam and calculate the mean for numeric features
    aggregated_data = data.groupby('HomeTeam')[features].mean().reset_index()

    # Normalize numeric features for parallel coordinates
    normalized_data = aggregated_data.copy()
    for feature in features:
        if feature in aggregated_data.columns:  # Ensure the feature exists in the DataFrame
            normalized_data[feature] = (
                (aggregated_data[feature] - aggregated_data[feature].min()) /
                (aggregated_data[feature].max() - aggregated_data[feature].min())
            )

    # Filter for specific teams
    filtered_data = normalized_data[normalized_data['HomeTeam'].isin(teams_to_plot)]

    return filtered_data

def preprocess_data(data):
    # This is where we will preprocess the data and create some new features 
    # by calculating them given the stats we have in this dataset
    # Ensure HomeTeam & AwayTeam are string
    data['HomeTeam'] = data['HomeTeam'].astype(str)
    data['AwayTeam'] = data['AwayTeam'].astype(str)

    # Create Historical Features
    # (Requires known post-match data for older matches, used to calculate historical averages)
    data['HomeGoalsAvg'] = data.groupby('HomeTeam')['FTHG'].transform('mean')
    data['AwayGoalsAvg'] = data.groupby('AwayTeam')['FTAG'].transform('mean')
    data['HomeWinPct']   = data.groupby('HomeTeam')['FTR'].transform(lambda x: (x == 'H').mean())
    data['AwayWinPct']   = data.groupby('AwayTeam')['FTR'].transform(lambda x: (x == 'A').mean())
    data['GoalDifference'] = data['HomeGoalsAvg'] - data['AwayGoalsAvg']

    # Create Betting Features from Pre-Match Odds
    # Make sure we don't divide by zero
    # Replace any 0 or missing odds with NaN to avoid `inf`` issues
    odds_columns = ['AvgH','AvgD','AvgA', 'Avg>2.5','Avg<2.5', 'AvgAHH','AvgAHA']
    for col in odds_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data[col] = data[col].replace(0, np.nan)

    # --- Core Implied Probabilities (1 / odds) ---
    if all(x in data.columns for x in ['AvgH','AvgD','AvgA']):
        data['Implied_Prob_H'] = 1 / data['AvgH']
        data['Implied_Prob_D'] = 1 / data['AvgD']
        data['Implied_Prob_A'] = 1 / data['AvgA']
        # Confidence = max implied probability
        data['Odds_Confidence'] = data[['Implied_Prob_H','Implied_Prob_D','Implied_Prob_A']].max(axis=1)
    else:
        data['Implied_Prob_H'] = np.nan
        data['Implied_Prob_D'] = np.nan
        data['Implied_Prob_A'] = np.nan
        data['Odds_Confidence'] = np.nan

    # --- Over/Under 2.5 Probabilities ---
    if all(x in data.columns for x in ['Avg>2.5','Avg<2.5']):
        data['Over2.5_Prob']  = 1 / data['Avg>2.5']
        data['Under2.5_Prob'] = 1 / data['Avg<2.5']
        data['Expected_Goals'] = data['Over2.5_Prob'] - data['Under2.5_Prob']
    else:
        data['Over2.5_Prob']  = np.nan
        data['Under2.5_Prob'] = np.nan
        data['Expected_Goals'] = np.nan

    # --- Asian Handicap Probabilities ---
    if all(x in data.columns for x in ['AvgAHH','AvgAHA']):
        data['AH_Prob_H'] = 1 / data['AvgAHH']
        data['AH_Prob_A'] = 1 / data['AvgAHA']
    else:
        data['AH_Prob_H'] = np.nan
        data['AH_Prob_A'] = np.nan

    # data['AHh'] should already exist as the handicap line.
    # If not, set default.
    if 'AHh' not in data.columns:
        data['AHh'] = 0.0

    # Market Discrepancy - just a possible measure, but not sure
    # Market_Expectation = Implied_Prob_H * 3 - 1

    # Multiplying by 3 is just a simplified stand-in for “typical” home odds (e.g., if 
    # average home odds are around 3.00, that’s a break-even point).
    # Subtracting 1 adjusts it so that a positive value implies the market is “overrating” the 
    # home team (believing the home team is more likely to win than 
    # a neutral baseline would suggest), whereas a negative value implies the market 
    # is “underrating” the home team.
    if 'Implied_Prob_H' in data.columns:
        data['Market_Expectation'] = data['Implied_Prob_H'] * 3 - 1
    else:
        data['Market_Expectation'] = np.nan

    # Encode Teams & Result
    unique_teams = pd.concat([data['HomeTeam'], data['AwayTeam']]).unique()
    team_encoder = LabelEncoder()
    team_encoder.fit(unique_teams)

    data['HomeTeam'] = team_encoder.transform(data['HomeTeam'])
    data['AwayTeam'] = team_encoder.transform(data['AwayTeam'])

    joblib.dump(team_encoder, save_path + 'team_encoder.pkl')

    # Encode final result FTR: 0=H, 1=D, 2=A
    ftr_encoder = LabelEncoder()
    data['FTR'] = ftr_encoder.fit_transform(data['FTR'])
    joblib.dump(ftr_encoder, save_path + 'ftr_encoder.pkl')

    # Fill any missing values
    data.fillna(0, inplace=True)

    return data, team_encoder

def predict_match(
    home_team_str, away_team_str,
    model, scaler, team_encoder,
    normalised_data
):
    
    # Build a feature vector for a future match.
    # Here, we just look up the historical features from any existing row.
    
    # Check team existence
    if home_team_str not in team_encoder.classes_:
        raise ValueError(f"Unknown home team: {home_team_str}")
    if away_team_str not in team_encoder.classes_:
        raise ValueError(f"Unknown away team: {away_team_str}")

    home_encoded = team_encoder.transform([home_team_str])[0]
    away_encoded = team_encoder.transform([away_team_str])[0]

    # We'll try to find any row in 'normalised_data' where HomeTeam==home_encoded,
    # and similarly for away. Then we can glean the historical stats + betting odds.

    home_rows = normalised_data[normalised_data['HomeTeam'] == home_encoded]
    away_rows = normalised_data[normalised_data['AwayTeam'] == away_encoded]

    if home_rows.empty or away_rows.empty:
        raise ValueError(
            f"Could not find historical data for {home_team_str} or {away_team_str}"
        )

    # Just pick the first row for each
    home_goals_avg = home_rows['HomeGoalsAvg'].iloc[0]
    away_goals_avg = away_rows['AwayGoalsAvg'].iloc[0]
    home_win_pct   = home_rows['HomeWinPct'].iloc[0]
    away_win_pct   = away_rows['AwayWinPct'].iloc[0]
    goal_diff      = home_goals_avg - away_goals_avg

    # For betting features, we pick them from the same row.
    # Use the home_rows row for the home side and away_rows row for the away side.

    implied_prob_h = home_rows['Implied_Prob_H'].iloc[0]
    implied_prob_d = home_rows['Implied_Prob_D'].iloc[0]
    implied_prob_a = away_rows['Implied_Prob_A'].iloc[0]
    odds_conf      = max(implied_prob_h, implied_prob_d, implied_prob_a)

    over2_5_prob   = home_rows['Over2.5_Prob'].iloc[0]
    under2_5_prob  = home_rows['Under2.5_Prob'].iloc[0]
    expected_goals = home_rows['Expected_Goals'].iloc[0]

    ah_prob_h      = home_rows['AH_Prob_H'].iloc[0]
    ah_prob_a      = away_rows['AH_Prob_A'].iloc[0]
    ahh_line       = home_rows['AHh'].iloc[0]

    market_exp     = home_rows['Market_Expectation'].iloc[0]

    # Construct the feature array in same order as 'features'
    input_vector = np.array([
        home_encoded,
        away_encoded,
       # home_goals_avg,
        #away_goals_avg,
        home_win_pct,
        away_win_pct,
        goal_diff,
        #implied_prob_h,
        #implied_prob_d,
        implied_prob_a,
        odds_conf,
       # over2_5_prob,
       # under2_5_prob,
        expected_goals,
       # ah_prob_h,
       # ah_prob_a,
       # ahh_line,
        market_exp
    ]).reshape(1, -1)

    # Scale it
    input_scaled = scaler.transform(input_vector)

    # Predict
    # Keras model -> returns probability distribution over 3 classes
    preds = model.predict(input_scaled, verbose=0)
    label_num = np.argmax(preds, axis=1)[0]

    mapping = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
    return mapping[label_num]

if __name__ == "__main__":
    print("untilsimported")