import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

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
    plt.show()

if __name__ == "__main__":
    print("untilsimported")