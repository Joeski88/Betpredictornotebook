import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def app():
    st.title("Prediction Hub")

    st.subheader("Individual Match Predictior")

    st.write("""
    Select Fixtures below, you can get predictions for a whole match week. 
    
    """)
    
    Teams = ("Arsenal", "Aston Villa", "Bournemouth","Brentford","Brighton",
    "Crystal Palace","Chelsea","Everton","Fulham","Ipswich",
    "Nottingham Forest","Manchester City","Manchester United","Liverpool",
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
