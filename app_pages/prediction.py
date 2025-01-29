import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import myutils

# Load the dataset
file_path = "./jupyter_notebooks/data/full_dataset.csv"
data = pd.read_csv(file_path)

def app():
    st.title("Prediction Hub")

    st.image("./images/pitch.jpg")

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

        # Optionally, show the selected teams for the match
        st.write(f"Match {i} teams: {team1} vs {team2}")
        st.write('---')  # Separator for readability

        # from multiselect
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

        fig = myutils.plot_metrics(data, team1, team2, metrics, mapping)

        st.pyplot(fig)
        # Optionally, show the selected teams for the match
        st.write(f"Match {i} teams: {team1} vs {team2}")
        st.write('---')  # Separator for readability

    st.header("Request A Bet Predictor/Probability")

    options = st.multiselect(
    "What metrics would you like to inlcude?",
    ["Home Goals", "Away goals", "Total match yellow cards", "Total match red cards",
    "Total match corners", "home corners", "away corners", "home SOT", "away SOT"],
    )

    st.write("You selected:", options)