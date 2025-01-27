import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
#from myutils import plot_metrics

# Load the dataset
file_path = "./jupyter_notebooks/data/full_dataset.csv"  # Replace with your dataset path
data = pd.read_csv(file_path)

metrics = ['FTHG', 'FTAG', 'HS', 'HST', 'HC', 'AS', 'AST', 'AC']
mapping = {
    'FTHG': 'Full time Home Goals', 
    'FTAG': 'Full time Away Goals', 
    'HS': 'Home Shots', 
    'HST': 'Home Shots on Target', 
    'HC': 'Home Corners', 
    'AS': 'Away Shots', 
    'AST': 'Away Shots on Target',
    'AC': 'Away Corners'
}


def app():
    st.title("Prediction Hub")

    st.subheader("Individual Match Predictior")

    st.write("""
    Select Fixtures below, you can get predictions for a whole match week. 
    
    """)
    
    Teams = ("Arsenal", "Aston Villa", "Bournemouth","Brentford","Brighton",
    "Crystal Palace","Chelsea","Everton","Fulham","Ipswich",
    "Nottingham Forest","Man City","Man United","Liverpool",
    "Tottenham","Newcastle","Southampton","Wolves","West Ham","Leicester",)

    # Loop to create 10 dropdowns for team selections in matches
    for i in range(1, 11):  # For 10 matches
        st.subheader(f"Match {i}")
        
        # Select the first team for the match
        team1 = st.selectbox(
            f"Select Team 1 for Match {i}",
            Teams,
            key=f"team1_{i}"  # Unique key for each selectbox
        )
        
        # Select the second team for the match
        team2 = st.selectbox(
            f"Select Team 2 for Match {i}",
            Teams,
            key=f"team2_{i}"  # Unique key for each selectbox
        )

        #myutils.plot_metrics(df, team1, team2, metrics, mapping)
        # Optionally, show the selected teams for the match
        st.write(f"Match {i} teams: {team1} vs {team2}")
        st.write('---')  # Separator for readability