import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

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

def plot_metrics(df, team1, team2, metrics =  ['FTHG', 'FTAG', 'HS', 'HST', 'HC', 'AS', 'AST', 'AC'], mapping={}):
    # Example dataset (replace with your actual dataset)
    cols = ['HomeTeam']
    cols.extend(metrics)
    
    # Aggregate stats by team
    stats = df[cols].groupby('HomeTeam').mean()
    
    # Select teams and stats to compare
    teams = [team1, team2] #['Man United', 'Chelsea']#, 'Arsenal', 'Liverpool']
    stats_to_plot = cols[1:] # ignore HomeTeam
    
    # Normalize data to a range of [0, 1]
    normalized_stats = stats[stats_to_plot].apply(lambda x: x / x.max(), axis=0)
    
    # Radar plot setup
    num_vars = len(stats_to_plot)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Plot data for each team
    
    for team in teams:
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
    return fig

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

if __name__ == "__main__":
    print("untilsimported")