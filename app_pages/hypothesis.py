import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px
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

    st.header("Multiteam Parallel Coordinate Plot")
    # Convert numeric-like columns stored as object
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_numeric(df[col])  # Convert if possible
        except ValueError:
            pass  # Keep as object if conversion fails

    # Select relevant columns (ensure only numeric data is used)
    categorical_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HC', 'AC', 'HR', 'AR', 'HS', 'AS', 'HST', 'AST']
    df[categorical_cols] = df[categorical_cols].apply(lambda x: pd.factorize(x)[0])

    # Drop unnecessary columns
    df.drop(['Date', 'Time', 'Referee'], axis=1, inplace=True, errors='ignore')

    # Select columns for visualization
    selected_columns = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HC', 'AC', 'HR', 'AR', 'HS', 'AS', 'HST', 'AST']
    existing_columns = [col for col in selected_columns if col in df.columns]

    if existing_columns:
        # Normalize data for parallel coordinates plot
        df_normalized = (df[existing_columns] - df[existing_columns].min()) / (df[existing_columns].max() - df[existing_columns].min())

        # Create Parallel Coordinates Plot
        fig = px.parallel_coordinates(
            df_normalized,
            dimensions=existing_columns,
            color=df['FTR'],  # Ensure this column exists
            labels={col: col.replace('_', ' ') for col in existing_columns},
            title="Parallel Coordinates Plot for Match Stats"
        )

        # Show plot
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Selected columns are missing from the dataset.")

    st.header("Individual Team Analysis")

    ### Arsenal individual Analysis

    # Convert numeric-like columns stored as object
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_numeric(df[col])  # Convert if possible
        except ValueError:
            pass  # Keep as object if conversion fails

    # Encode categorical columns
    categorical_cols = ['HomeTeam', 'AwayTeam', 'FTR', 'HTR']
    df[categorical_cols] = df[categorical_cols].apply(lambda x: pd.factorize(x)[0])

    # Drop unnecessary columns
    df.drop(['Date', 'Time', 'Referee'], axis=1, inplace=True, errors='ignore')

    # Filter for Arsenal Matches
    team_name = "Arsenal"
    arsenal_df = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)]

    # Selected columns for analysis
    selected_columns = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC']
    existing_columns = [col for col in selected_columns if col in arsenal_df.columns]

    # Display title
    st.title(f"Individual Team Analysis: {team_name}")

    # Display summary statistics
    st.subheader("Summary Statistics")
    st.write(arsenal_df[existing_columns].describe())

    # Convert data into long format for Plotly
    melted_goals = arsenal_df[['FTHG', 'FTAG']].reset_index().melt(id_vars='index', var_name='Goal Type', value_name='Goals')

    # bar chart with x and y values
    fig1 = px.bar(
        melted_goals,
        x='index',
        y='Goals',
        color='Goal Type',
        labels={'Goals': 'Number of Goals', 'index': 'Match Index'},
        title=f"{team_name} - Goals Scored & Conceded",
        barmode='group'
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Line Plot: Shots on Target Trend
    st.subheader("Shots on Target Over Matches")
    
    # Convert data into long format for Plotly line chart
    melted_line_data = arsenal_df[['FTHG', 'FTAG']].reset_index().melt(id_vars='index', var_name='Goal Type', value_name='Goals')

    # Create the line chart
    fig2 = px.line(
        melted_line_data,
        x='index',  
        y='Goals',  
        color='Goal Type',
        labels={'Goals': 'Number of Goals', 'index': 'Match Index'},
        title=f"{team_name} - Goals Scored & Conceded Over Time",
        markers=True
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Heatmap: Correlation Matrix
    st.subheader("Correlation Heatmap for Arsenal Matches")
    fig, ax = plt.subplots(figsize=(10, 6))
    sb.heatmap(arsenal_df[existing_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

    ### SOT to win percentage stacked bar plot
    st.subheader("SOT to Win Percentage stacked bar plot")
        # Ensure correct column types
    df['HST'] = pd.to_numeric(df['HST'], errors='coerce')
    df['FTR'] = df['FTR'].astype(str)  # Ensure match result is a string

    # Filter necessary columns
    df_filtered = df[['HomeTeam', 'HST', 'FTR']].copy()

    # Convert match result into win/loss/draw category
    df_filtered['Win'] = df_filtered['FTR'].apply(lambda x: 1 if x == 'H' else 0)

    # Group by team and HST, calculate win percentage
    hst_win_df = df_filtered.groupby(['HomeTeam', 'HST']).agg(
        Matches=('FTR', 'count'),
        Wins=('Win', 'sum')
    ).reset_index()

    # Calculate win percentage
    hst_win_df['WinPercentage'] = (hst_win_df['Wins'] / hst_win_df['Matches']) * 100

    # Create stacked bar plot
    fig = px.bar(
        hst_win_df,
        x="HomeTeam",
        y="WinPercentage",
        color="HST",
        title="HST (Shots on Target) to Win Percentage by Team",
        labels={"WinPercentage": "Win %", "HST": "Shots on Target"},
        barmode="stack",
    )

    # Display chart in Streamlit
    st.plotly_chart(fig)

