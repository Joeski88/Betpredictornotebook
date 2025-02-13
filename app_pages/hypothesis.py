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

    st.subheader("Goals")
    
    st.write("""
    Goals win games. We've established that a huge focus on football betting is 
    how many goals a team scores. If you back a team that is not scoring goals 
    for fun, there's always a risk.  
    """)

    st.write("Let's look into how goals affect games in a little more detail.")

    # Filter matches where the selected team was either Home or Away
    team_matches = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)].copy()

    # Create a column for opponent team
    team_matches['Opponent'] = team_matches.apply(
        lambda row: row['AwayTeam'] if row['HomeTeam'] == team_name else row['HomeTeam'], axis=1
    )

    # Assign goals scored and conceded based on home/away status
    team_matches['Goals_For'] = team_matches.apply(
        lambda row: row['FTHG'] if row['HomeTeam'] == team_name else row['FTAG'], axis=1
    )
    team_matches['Goals_Against'] = team_matches.apply(
        lambda row: row['FTAG'] if row['HomeTeam'] == team_name else row['FTHG'], axis=1
    )

    # Group by opponent and sum the goals
    team_goals = team_matches.groupby('Opponent').agg(
        Total_Goals_For=('Goals_For', 'sum'),
        Total_Goals_Against=('Goals_Against', 'sum')
    ).reset_index()

    # Convert data into long format for Plotly
    melted_goals = team_goals.melt(
        id_vars='Opponent', var_name='Goal Type', value_name='Goals'
    )

    # Bar chart with x and y values
    fig1 = px.bar(
        melted_goals,
        x='Opponent',  # X-axis: Opposing teams
        y='Goals',  # Y-axis: Goals scored/conceded
        color='Goal Type',  
        labels={'Goals': 'Total Goals', 'Opponent': 'Opponent Team'},
        title=f"{team_name} - Total Goals Scored & Conceded Against All Teams",
        barmode='group',  
        color_discrete_sequence=['red', 'blue']  # Red for goals scored, blue for conceded
    )

    st.plotly_chart(fig1, use_container_width=True)
    
    st.write("""
    If we look at the 23/24 season with Arsenal, they out scored their opponents
    on aggregate most times throughout the season. In fact only 2 teams
    outscored them, Fulham and Aston Villa. That trajcetory has continued into 
    this season, if you look at 24/25 dataset, you will see that despite not all
    games having been played, are on course for a similar finish to the season.
    """)

    st.write('---')

    st.write("""
    Let's investigate if shot creation and prevention will have an effect.
    """)

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
    fig2.for_each_trace(lambda t: t.update(name={'HS': team_name, 'AS': 'Away team'}.get(t.name, t.name)))
    st.plotly_chart(fig2, use_container_width=True)

    st.write("""
    This graph shows shows a change. The number of shots very much fluctuates.
    Making it very difficult to make any predictions from, in the first graph, 
    it shows Arsenal more often than not score more goals than their opponent, 
    with that in mind you would assume that they would also take a lot of shots
    to make that happen. Whereas this graph is plotting without any real
    pattern.
    """)

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

    ### SOT to win percentage grouped bar plot
    st.subheader("Average shots on target to total average win bar plot")

    # Filter necessary columns
    df_filtered = df[['HomeTeam', 'FTHG', 'FTR']].copy()

    # Convert match result into binary win indicator (1=Win, 0=Not Win)
    df_filtered['Win'] = (df_filtered['FTR'] == 'H').astype(int)
   # print("Filtered", df_filtered)
    # Get total Goals and Wins for each HomeTeam
    team_stats_df = df_filtered.groupby('HomeTeam').agg(
        TotalGoals=('FTHG', 'sum'),
        TotalWins=('Win', 'sum')
    ).reset_index()

    teams = team_stats_df['HomeTeam']
    goals = team_stats_df['TotalGoals']
    wins = team_stats_df['TotalWins']

    # Set up the x locations for the groups
    x = np.arange(len(teams))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12,6))

    # Plot bars for Total Goals
    rects1 = ax.bar(x - width/2, goals, width, label='Total Goals', color='skyblue')

    # Plot bars for Total Wins
    rects2 = ax.bar(x + width/2, wins, width, label='Total Wins', color='orange')

    # Add some text for labels, title and custom x-axis tick labels
    ax.set_ylabel('Count')
    ax.set_title('Total Goals vs. Total Wins by Home Team')
    ax.set_xticks(x)
    ax.set_xticklabels(teams, rotation=45, ha='right')
    ax.legend()

    st.pyplot(fig)


    st.write("""
    Here we can clearly see where the cliche 'goals win games' comes from. 
    There's a clear trend here, where the teams with a higher SOT count, 
    generally score more goals than teams that dont. Always worth thinking about
    these metrics when thinking of an under/over goals bet. 
    """)

    st.write('---')

    # TO ADD A MULTI-DATASET BAR PLOT
    st.subheader("Multi-Dataset Bar Plot")

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

    st.write("""
    As the focus on this page has been on Arsenal, we will continue with that. 
    After loading each of the 5 years data sets separately, Arsenal's stats were
    as follows:""")
    st.write("20/21: 4 SOT ave")
    st.write("21/22: 6 SOT ave")    
    st.write("22/23: 6.5 SOT ave")
    st.write("23/24: 6.5 SOT ave")
    st.write("24/25: 6.5 SOT ave (so far)")
    
    st.write("""
    This shows a clear trend and improvement over the last 4 years. Arsenal have
    only become a major force again in the premier league over the last 5 years.
    This highlights the attacking improvement, but, not only improvement, 
    but consistency too, which is a major part of sports betting.
    """)

    st.write('---')