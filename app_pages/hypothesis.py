import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import myutils

# Load dataset
file_path = "./jupyter_notebooks/data/full_dataset.csv"
data = pd.read_csv(file_path)
df =  pd.read_csv("./jupyter_notebooks/data/full_dataset.csv")

def app():
    st.title("Hypothesis")

    # Hypothesis Section
    st.header("Hypothesis")
    st.markdown("""
    We hypothesize that by leveraging historical football match data, including:
    - Team statistics (e.g., goals scored, goals conceded, FT results).
    - Recent match performance (e.g., win/loss streaks).
    - Player statistics (e.g., top scorer, key injuries).
    - Match conditions (e.g., home/away advantage).

    We can use machine learning to:
    1. Predict match outcomes (win/draw/lose) with high accuracy.
    2. Provide data-driven insights for better decision-making in 
    football betting.

    ### Assumptions
    - The predictions depend on the availability and quality of historical data.
    - The accuracy of the predictor may vary based on external factors 
    (e.g., unexpected injuries, referee decisions).

    By integrating these insights into a Streamlit app, we aim to create an 
    interactive, user-friendly interface for analyzing predictions and 
    visualizing match data.
    """)

    st.header("Heat Map Win Percentage")

    st.header("Multiteam Parallel Coordinate Plot")

    st.header("Individual Team Analysis")

    st.subheader("Interactive Bar Plot, adjustable metrics to show different data")

    st.subheader("SOT to Win Percentage stacked bar plot")

    # Plot histograms for numeric columns
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns].hist(figsize=(15, 10), bins=20, color='skyblue', edgecolor='black')
    plt.suptitle('Histograms of Numeric Features', fontsize=16)
    plt.tight_layout()
    st.pyplot(plt.gcf())
