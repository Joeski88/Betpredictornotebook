import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

import myutils

save_path = "./model/"

# Load the dataset
file_path = "./jupyter_notebooks/data/full_dataset.csv"
data = pd.read_csv(file_path)
df =  pd.read_csv("./jupyter_notebooks/data/full_dataset.csv")

normalised_data, label_encoder = myutils.preprocess_data(data)
normalised_data.head()

def app():
    nn_model_loaded = tf.keras.models.load_model(save_path + 'nn_model.h5')

    # Load the scaler & team_encoder
    scaler_loaded = joblib.load(save_path + 'scaler.pkl')
    team_encoder_loaded = joblib.load(save_path + 'team_encoder.pkl')

    st.title("Prediction Hub")

    # st.image("./images/pitch.jpg")

    st.subheader("Individual Match Predictior")

    st.write("""
    Select Fixtures below, you can get predictions for a whole match week. 
    """)

    st.write("""
    On the right is the prediction o f the chosen result, home win, away win and 
    draw.
    """)

    Teams = ("Arsenal", "Aston Villa", "Bournemouth","Brentford","Brighton",
    "Crystal Palace","Chelsea","Everton","Fulham","Ipswich",
    "Nott'm Forest","Man City","Man United","Liverpool",
    "Tottenham","Newcastle","Southampton","Wolves","West Ham","Leicester",)

    col1, col2, col3 = st.columns([0.2,0.2,0.33]) # To make match selection side by side, instead of stacked
    match_results = []
    # Loop to create 10 dropdowns for team selections in matches
    for i in range(1, 11):  # For 10 matches
        #st.subheader(f"Match {i}")
        
        with col1:
        # Select the first team for the match
            team1 = st.selectbox(
                f"Select Team 1 for Match {i}",
                Teams,
                key=f"team1_{i}"  # Unique key for each selectbox
        )
        with col2:
        # Select the second team for the match
            team2 = st.selectbox(
                f"Select Team 2 for Match {i}",
                Teams,
                key=f"team2_{i}"
            )

        nn_result = myutils.predict_match(
                team1, team2,
                nn_model_loaded,
                scaler_loaded,
                team_encoder_loaded,
                normalised_data
        )

        with col3:
            st.selectbox(
                "",
                [nn_result],
                key=f"NN_{i}",
            )

        match_results.append([team1, team2, nn_result])

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

    st.header("Request A Bet Predictor/Probability")

    st.write("""
    Select which metrics your interested in betting on for a specific team. The radar plot
    will generate a plot of average data from the last 5 years. Allowing you to 
    get an idea of the kind of numbers to be looking at.
    """)

    options = st.multiselect(
        "What metrics would you like to include?",
        metrics
    )

    # st.inputbox for team 1 and team2
    team1 = st.selectbox(
        "Teams",
         Teams,
         key=f"team1"
    )
 
    fig, ax = myutils.plot_metrics(df, team1, None, options, mapping, ax=None)
    st.pyplot(fig)

    st.write("""
    I struggled to change the key on the radar plot from percentage to numerical.
    It's essentially the same as 10% = 1 and 100% = 10.
    """)

    st.write('---')