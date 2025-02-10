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

    # st.header("Team Comparison Parallel Coordinate Plot")

    # st.write("""
    # Select teams to compare.
    # """)
    # # Ensure numeric-like columns stored as object are converted
    # for col in df.select_dtypes(include=['object']).columns:
    #     try:
    #         df[col] = pd.to_numeric(df[col])  # Convert if possible
    #     except ValueError:
    #         pass  # Keep as object if conversion fails

    # # Select relevant columns (ensure only numeric data is used)
    # categorical_cols = ['FTHG', 'FTAG', 'FTR', 'HC', 'AC', 'HR', 'AR', 'HS', 'AS', 'HST', 'AST']
    # df[categorical_cols] = df[categorical_cols].apply(lambda x: pd.factorize(x)[0])

    # # Drop unnecessary columns
    # df.drop(['Date', 'Time', 'Referee'], axis=1, inplace=True, errors='ignore')

    # # Get unique team names
    # teams = sorted(set(df['HomeTeam'].unique()).union(df['AwayTeam'].unique()))

    # # Team selection
    # team_1 = st.selectbox("Select First Team:", teams, index=0)
    # team_2 = st.selectbox("Select Second Team:", teams, index=1)

    # # Filter dataset to only include selected teams
    # filtered_df = df[(df['HomeTeam'].isin([team_1, team_2])) | (df['AwayTeam'].isin([team_1, team_2]))]

    # # Select columns for visualization
    # selected_columns = ['FTHG', 'FTAG', 'FTR', 'HC', 'AC', 'HR', 'AR', 'HS', 'AS', 'HST', 'AST']
    # existing_columns = [col for col in selected_columns if col in filtered_df.columns]

    # if existing_columns and not filtered_df.empty:
    #     # Normalize data for parallel coordinates plot
    #     df_normalized = (filtered_df[existing_columns] - filtered_df[existing_columns].min()) / (filtered_df[existing_columns].max() - filtered_df[existing_columns].min())

    #     # Create Parallel Coordinates Plot
    #     fig = px.parallel_coordinates(
    #         df_normalized,
    #         dimensions=existing_columns,
    #         color=filtered_df['FTR'],  # Color based on Full Time Result
    #         labels={col: col.replace('_', ' ') for col in existing_columns},
    #         title=f"Comparison: {team_1} vs {team_2}"
    #     )

    #     # Show plot
    #     st.plotly_chart(fig, use_container_width=True)
    # else:
    #     st.warning("No data available for the selected teams.")

    # st.write(""" 
    # This parallel cooridnated plot shows a comparison of 2 teams of your choice, 
    # I will compare Arsenal and Aston Villa.

    # As you can see.............
    # """)

    ### Arsenal individual Analysis
    # st.write(df.dtypes)  # Display the data types of all columns
    
    st.header("Individual Team Analysis - Arsenal")

    # Dropdown to select the dataset
    dataset_name = st.selectbox("Select a Dataset", list(datasets.keys()))

    df = datasets[dataset_name]

    # List of teams from the dataset
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique().tolist()

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

    # Display summary statistics
    # st.subheader("Summary Statistics")
    # st.write(team_df[existing_columns].describe())
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

    # Line Plot: Shots on Target Trend
    st.subheader("Home Shot vs Away Shots")
    
    # Convert data into long format for Plotly line chart
    melted_line_data = team_df[['HS', 'AS']].reset_index().melt(id_vars='index', var_name='Shots', value_name='Shots')

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

    # Heatmap: Correlation Matrix
    st.subheader("Correlation Heatmap for Arsenal Matches")
    fig, ax = plt.subplots(figsize=(10, 6))
    sb.heatmap(team_df[existing_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

    ### SOT to win percentage stacked bar plot
    st.subheader("SOT to Win Percentage stacked bar plot")

    # Filter necessary columns
    df_filtered = df[['HomeTeam', 'HST', 'FTR']].copy()

    # Convert match result into win/loss/draw category
    df_filtered['Win'] = df_filtered['FTR']
    print("df_filtered", df_filtered.groupby(['HomeTeam', 'HST']).head())
    # Group by team and HST, calculate win percentage
    hst_win_df = df_filtered.groupby(['HomeTeam', 'HST']).agg(
        Matches=('FTR', 'count'),
        Wins=('Win', 'count')
    ).reset_index()

    # Calculate win percentage
    hst_win_df['WinPercentage'] = (hst_win_df['Wins'] / hst_win_df['Matches']) * 100
    print(hst_win_df.head())
    # Create stacked bar plot
    fig = px.bar(
        hst_win_df,
        x="HomeTeam",
        y="WinPercentage",
        color="HST",
        title="HST (Shots on Target) to Win Percentage by Team",
        labels={"WinPercentage": "Win %", "HST": "Shots on Target"},
        barmode="stack",
        color_discrete_sequence= ["Red", "Yellow"]
    )

    # Display chart in Streamlit
    st.plotly_chart(fig)

    # TO ADD A MULTI-DATASET BAR PLOT
    st.subheader("📊 Multi-Dataset Bar Plot")

    # Select dataset
    selected_dataset = st.selectbox("Choose a dataset:", list(datasets.keys()))

    # Get the selected DataFrame
    df = datasets[selected_dataset]

    # Display dataset
    st.write("### 📋 Selected Dataset", df)
    # st.write("Column data types:", df.dtypes)

    # Plot bar chart
    fig, ax = plt.subplots()
    df.plot(kind="bar", x="HST", y=df.columns[1], legend=False, ax=ax)
    ax.set_ylabel(df.columns[1])
    ax.set_title(f"{selected_dataset} Bar Chart")
    st.pyplot(fig)