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

def app():
        # Load dataset
    file_path = "./jupyter_notebooks/data/full_dataset.csv"

    df =  pd.read_csv("./jupyter_notebooks/data/full_dataset.csv")

    datasets = {
    "20-21": pd.read_csv("./jupyter_notebooks/data/20_21.csv"),
    "21-22": pd.read_csv("./jupyter_notebooks/data/21_22.csv"),
    "22-23": pd.read_csv("./jupyter_notebooks/data/22_23.csv"),
    "23-24": pd.read_csv("./jupyter_notebooks/data/23_24.csv"),
    "24-25": pd.read_csv("./jupyter_notebooks/data/24_25.csv")}

    # st.write(df.head())
    print(df.columns)
    print("Main dataset columns:", df.columns)
    print("Loaded dataset keys:", datasets.keys())
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

    st.write('---')

    ### Arsenal individual Analysis
    
    st.header("Individual Team Analysis - Arsenal")

    # Dropdown to select the dataset
    dataset_name = st.selectbox("Select a Dataset", list(datasets.keys()))

    df = datasets[dataset_name]

    # List of teams from the dataset
    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique().tolist())

    team_name = st.selectbox("Select a Team for Analysis", teams)

    categorical_cols = ['HomeTeam', 'AwayTeam', 'FTR', 'HTR']

    df.drop(['Date', 'Time', 'Referee'], axis=1, inplace=True, errors='ignore')

    # Filter dataset for the selected team
    team_df = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)]

    # Selected columns for analysis
    selected_columns = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC']
    existing_columns = [col for col in selected_columns if col in team_df.columns]

    st.write("""
    In this section I will analyse individual team data, 
    and I will focus on Arsenal 
    (Mainly because of alphabetical order, but also as I am a huge Arsenal fan).
    I have also added a feature that allows you to edit the plots to display 
    any teams or season/year data on the given metrics. So you can change to see 
    a teams development over the past 5 years.
    """)

    st.write('---')

    st.subheader("Arsenal - Goals For and Against")

    # Convert data into long format for Plotly
    melted_goals = team_df[['FTHG', 'FTAG']].reset_index().melt(id_vars='index', var_name='Goal Type', value_name='Goals')

    # Bar chart with x and y values
    fig1 = px.bar(
        melted_goals,
        x='index',
        y='Goals',
        color='Goal Type',
        labels={'Goals': 'Number of Goals', 'index': 'Match Index'},
        title=f"{team_name} - Goals Scored & Conceded ({dataset_name})",
        barmode='group',
        color_discrete_sequence=['red', 'blue']  # Custom color to plot
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.write('---')

    # Line Plot: Shots on Target Trend
    st.subheader("Home Shot vs Away Shots")
    
    # Convert data into long format for Plotly line chart
    melted_line_data = team_df[['HS', 'AS']].reset_index().melt(id_vars='index', var_name='Shot Type', value_name='Shots')

    # Create the line chart
    fig2 = px.line(
        melted_line_data,
        x='index',  
        y='Shots',  
        color='Shot Type',
        labels={'Shots': 'Number of Shots', 'index': 'Match Index'},
        title=f"{team_name} - Shots For and Against",
        markers=True,
        color_discrete_sequence=['orange', 'green']
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.write('---')

    # Heatmap: Correlation Matrix

    st.subheader("Correlation Heatmap for Arsenal Matches")
    fig, ax = plt.subplots(figsize=(10, 6))
    sb.heatmap(team_df[existing_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.write("""
    Herre you can see some clear correlations, the obvious ones sucha as, HST
    (home shots on target) is predictably highly correlated with
    FTHG (full time home goals), and shots on target being highly 
    correlated with total shots. This further enhances the hypothesis that the 
    more attacking options you create, the higher chance there is of the team 
    scoring goals, and essentially winning games.
    """)

    st.write('---')

    ### SOT to win percentage stacked bar plot
    st.subheader("SOT to Win Percentage stacked bar plot")

    df_filtered = df[['HomeTeam', 'FTHG', 'FTR']].copy()

    # Convert match result into binary win indicator (1=Win, 0=Not Win)
    df_filtered['Win'] = (df_filtered['FTR'] == 'H').astype(int)
    print("Filtered", df_filtered)
    # Get total Goals and Wins for each HomeTeam
    team_stats_df = df_filtered.groupby('HomeTeam').agg(
        TotalGoals=('FTHG', 'sum'),
        TotalWins=('Win', 'sum')
    ).reset_index()

    fig, ax = plt.subplots()

    x = np.arange(len(team_stats_df['HomeTeam']))
    bar_width = 0.6

    # Plot Goals and Wins
    bars1 = ax.bar(x - bar_width/2, team_stats_df['TotalGoals'], bar_width, label='Total Goals', color='blue')
    bars2 = ax.bar(x + bar_width/2, team_stats_df['TotalWins'], bar_width, label='Total Wins', color='orange')

    ax.set_xlabel("Home Team")
    ax.set_ylabel("Total Count")
    ax.set_title("Goals scored vs Wins")
    ax.set_xticks(x)

    ax.set_xticklabels(team_stats_df['HomeTeam'], rotation=90)
    ax.legend()

    st.plotly_chart(fig)

    st.write('---')

    # TO ADD A MULTI-DATASET BAR PLOT
    st.subheader("📊 Multi-Dataset Bar Plot")

    # Select dataset
    selected_dataset = st.selectbox("Choose a dataset:", list(datasets.keys()))

    # Get the selected DataFrame
    df = datasets[selected_dataset]
    avg_shots_df = df.groupby("HomeTeam")["HST"].mean().reset_index()

    st.write("###  Selected Dataset", df)

     # Plot bar chart
    fig, ax = plt.subplots()

    avg_shots_df.set_index("HomeTeam").plot(kind="bar", legend=False, ax=ax, color='skyblue')

    # Set axis labels and title
    ax.set_xlabel("Home Team")
    ax.set_ylabel("Average Shots on Target (HST)")
    ax.set_xticklabels(avg_shots_df["HomeTeam"], rotation=90)
    ax.set_title(f"{selected_dataset} - Average Shots on Target Per Team")

    st.pyplot(fig)

    st.write('---')