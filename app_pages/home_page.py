import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


    # Load the dataset
file_path = "./jupyter_notebooks/data/full_dataset.csv"  # Replace with your dataset path
data = pd.read_csv(file_path)

# Set which features to focus on
features = [
    'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST',
    'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 
    'AvgH', 'AvgD', 'AvgA',
    'GoalDifference', 'MarketConsensus',
    'FTR'
]
# Overall win percentage function
def calculateOverallPerformance(data):
    # Calculate total matches and wins for each team (Home = 1, Away=-1
    home_wins = data[data['FTR'] == 'H'].groupby('HomeTeam').size()  # Home wins
    away_wins = data[data['FTR'] == 'A'].groupby('AwayTeam').size()  # Away wins
    
    home_matches = data.groupby('HomeTeam').size()  # Total home matches
    away_matches = data.groupby('AwayTeam').size()  # Total away matches
    
    # Total matches and wins
    total_wins = home_wins.add(away_wins, fill_value=0)
    total_matches = home_matches.add(away_matches, fill_value=0)
    
    # Calculate win percentage
    win_percentage = (total_wins / total_matches) * 100
    
    # Sort win percentage in descending order
    win_percentage = win_percentage.sort_values(ascending=False)
    
    # Bar plot code
    fig = plt.figure(figsize=(12, 6))
    win_percentage.plot(kind='bar', color='skyblue', alpha=0.8, edgecolor='black')
    plt.title('Team Win Percentage')
    plt.xlabel('Team Name')
    plt.ylabel('Win Percentage (%)')
    plt.xticks(rotation=90, fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
def feature_engineering(data):
    data['GoalDifference'] = data['FTHG'] - data['FTAG']
    data['MarketConsensus'] = (data['AvgH'] + data['AvgD'] + data['AvgA']) / 3

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

def app():
    st.title("Football betting notebook")

    st.image("./images/premlogo.png")

    st.header("Project Summary")

    st.subheader("**Objective**")
    st.write(
        """
        The Betting Tips Predictor Jupyter Notebook project is a predictive 
        analytics tool designed to analyze historical sports data and provide 
        informed betting recommendations. By leveraging machine learning and 
        statistical models, the project aims to empower users with data-driven 
        insights to enhance decision-making in the sports betting industry.
        This initiative offers a cost-effective and scalable solution for both 
        casual bettors and organizations looking to optimize their betting 
        strategies.
        """
    )

    st.subheader("Business Requirements")

    st.write(
        """
        - Develop a Jupyter Notebook with a modular structure for easy 
          scalability.
        - Implement predictive models for win/loss probabilities and score 
          forecasting.
        - Incorporate data visualization tools for insightful and actionable 
          outputs.
        - Ensure reproducibility and transparency in the analytical process.
        - Deliver a user-friendly interface to cater to both technical and 
          non-technical users.

        """
    )

    st.header("Data Set Summary")
    st.write("The link for the dataset key is located beneath.")
    st.page_link("https://www.football-data.co.uk/notes.txt", 
                label="Dataset Key", icon="⚽")
    st.write(
        """
        The data set is compiled from football statistics collected from over 
        the last few years. they consist of various metrics, including:
        - Goals scored
        - Goals conceded
        - Corners for, and corners against
        - Fouls for and against
        - Booking points

        The list is extensive and there are many more. See more information and 
        see the abbreviations used broken down and explained in the above link. 
        """
    )
    # Data Preprocessing
    # Handle missing values
    data.ffill(inplace=True)
    
    st.write(data.head())

    st.header("Overall Data Summary Visualisations")

    st.subheader("Overall Win Percentage Bar Plot")

    st.subheader("Team Comparison Radar Plot")

    st.subheader("Individual Team Comparison Radar Plot, Year By Year")

    st.subheader("Outliers")