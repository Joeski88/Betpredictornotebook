import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
from scipy.interpolate import make_interp_spline

import myutils

# Feature selection
features = [
        'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST',
        'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 
        'AvgH', 'AvgD', 'AvgA'
]

def app():

    # Load dataset
    file_path = "./jupyter_notebooks/data/full_dataset.csv"
    data = pd.read_csv(file_path)
    df =  pd.read_csv("./jupyter_notebooks/data/full_dataset.csv")

    datasets = {
    "20-21": pd.read_csv("./jupyter_notebooks/data/20_21.csv"),
    "21-22": pd.read_csv("./jupyter_notebooks/data/21_22.csv"),
    "22-23": pd.read_csv("./jupyter_notebooks/data/22_23.csv"),
    "23-24": pd.read_csv("./jupyter_notebooks/data/23_24.csv"),
    "24-25": pd.read_csv("./jupyter_notebooks/data/24_25.csv")}
    st.title("Data Analysis")

    st.header("Data Set Summary")
    st.write("The link for the dataset key is located beneath.")
    st.page_link("https://www.football-data.co.uk/notes.txt", 
                label="Dataset Key", icon="⚽")

    st.write(
        """
        On this page we will undertake a more detailed study of the data set 
        I'm using. The main metrics to be used are:
        
        - HomeTeam = Home Team
        - AwayTeam = Away Team
        - FTHG and HG = Full Time Home Team Goals
        - FTAG and AG = Full Time Away Team Goals
        - FTR and Res = Full Time Result (H=Home Win, D=Draw, A=Away Win)
        - HTHG = Half Time Home Team Goals
        - HTAG = Half Time Away Team Goals
        - HTR = Half Time Result (H=Home Win, D=Draw, A=Away Win)
        - Attendance = Crowd Attendance
        - Referee = Match Referee
        - HS = Home Team Shots
        - AS = Away Team Shots
        - HST = Home Team Shots on Target
        - AST = Away Team Shots on Target
        - HHW = Home Team Hit Woodwork
        - AHW = Away Team Hit Woodwork
        - HC = Home Team Corners
        - AC = Away Team Corners
        - HF = Home Team Fouls Committed
        - AF = Away Team Fouls Committed
        - HFKC = Home Team Free Kicks Conceded
        - AFKC = Away Team Free Kicks Conceded
        - HO = Home Team Offsides
        - AO = Away Team Offsides
        - HY = Home Team Yellow Cards
        - AY = Away Team Yellow Cards
        - HR = Home Team Red Cards
        - AR = Away Team Red Cards
        - HBP = Home Team Bookings Points (10 = yellow, 25 = red)
        - ABP = Away Team Bookings Points (10 = yellow, 25 = red)

        A huge stat in football data analysis is a metric called 'XG',
        short for Expected Goals, however I have decided against using this
        metric as this is a fairly recent statistic, and wont go back far enough 
        which therefore makes it  hard as I want to use as much historical data
        as possible.
        """)

    st.header("Correlation Matrix")

    st.write("""
    A correlation matrix is a table that shows the pairwise correlation 
    coefficients between variables in a dataset. Below there are 2 matrix, 1 
    containing the whole set of metrics provided within the dataset, and the 
    second using a selected amount of essential data metrics.
    """)

    st.subheader("Correlation Matrix 1")

    st.write("""
    This is the correlation matrix of the untouched dataset. As you can see it 
    provides information that can be used, such as, the most correlated data is 
    between the various betting data companies. This information isnt really 
    useful for the app's intentions. So I have dropped the unimportant data and 
    created a new matrix, featuring only data being used in the project.
    """)
    correlation_matrix = data.corr(numeric_only=True)
    corr_plot = sb.heatmap(correlation_matrix, cmap="YlGnBu", annot=False)
    fig = corr_plot.get_figure()
    st.pyplot(fig)
    plt.clf() # To clear the figure
    data_feat, team_mapping, ftr_mapping = myutils.feature_engineering(data)

    st.subheader("Correlation Matrix 2")

    st.write("""
    The second correlation matrix, as you can see is alot more readable. 
    There are several points of interest that are highly correlaated aswell as 
    some with low correlation. For example, home & away shots on target and 
    home & away goals are highly correlated in a positive way, indicating and obvious trend in 
    having more shots on target means a higher potential to score more goals.
    In comparison, yellow cards and red cards, are not really correlated 
    as most of the points sit at close to 0 which means they dont tend to 
    affect each other in terms of outcome.
    """)

    numeric_features = [feat for feat in features if feat in data_feat.select_dtypes(include=[np.number]).columns]
    corr_matrix_feats = data_feat[numeric_features].corr()
    corr_plot2 = sb.heatmap(corr_matrix_feats, cmap="YlGnBu", annot=False)
    fig = corr_plot2.get_figure()
    st.pyplot(fig)

    st.write('---')
        
    st.header("Overall Data Summary Visualisations")

    st.subheader("Outliers")

    st.write("""Analyzing outliers in football data involves identifying data 
             points that deviate significantly from the norm, which can provide 
             valuable insights into player performance, team dynamics, and game
             outcomes. The issue with this, is that, as the cliche goes, 
             'Anything can happen in a game of football' and at any moment. 
             This will account for many of the outliers in football data.""")

    data_feat, team_mapping, ftr_mapping = myutils.feature_engineering(data)

    # Define metric pairs for scatter plots
    metric_pairs = [
        ("FTHG", "HST"),  # Full-time home goals vs Home shots on target
        ("HS", "AS"),    # Home shots vs away shots
        ("HF", "AF"),     #  Home fouls vs Away fouls
        ("HC", "AC")    # home corners vs away corners
    ]

    # Set up figure and axes for 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

        # Function to detect outliers using Z-score
    def detect_outliers_zscore(df, column, threshold=3):
            z_scores = zscore(df[column].dropna())
            return df.loc[abs(z_scores) > threshold]

    # necessary to repeat dataset path as parallel plot doesnt work without - unknown why this bug happens
    datasets = {
            "20-21": pd.read_csv("./jupyter_notebooks/data/20_21.csv"),
            "21-22": pd.read_csv("./jupyter_notebooks/data/21_22.csv"),
            "22-23": pd.read_csv("./jupyter_notebooks/data/22_23.csv"),
            "23-24": pd.read_csv("./jupyter_notebooks/data/23_24.csv"),
            "24-25": pd.read_csv("./jupyter_notebooks/data/24_25.csv")}

    # Load dataset
    file_path = "./jupyter_notebooks/data/full_dataset.csv"
    data = pd.read_csv(file_path)
    df =  pd.read_csv("./jupyter_notebooks/data/full_dataset.csv")

    # Iterate over metric pairs and datasets
    for i, (x_col, y_col) in enumerate(metric_pairs):
        season = list(datasets.keys())[i]
        df_season = datasets[season]

        # Ensure 'Team' column exists
        if 'HomeTeam' not in df_season.columns:
            print(f"Skipping {season}, 'HomeTeam' column missing")
            continue

        # Drop missing values
        df_season = df_season[[x_col, y_col, 'HomeTeam']].dropna()

        # Detect outliers
        outliers_x = detect_outliers_zscore(df_season, x_col)
        outliers_y = detect_outliers_zscore(df_season, y_col)

        # Combine outliers
        outliers = pd.concat([outliers_x, outliers_y]).drop_duplicates()

        # Scatter plot with regression line
        sb.regplot(data=df_season, x=x_col, y=y_col, ax=axes[i], scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})

        # Highlight outliers in red
        sb.scatterplot(data=outliers, x=x_col, y=y_col, ax=axes[i], color="red", marker="x", s=100, label="Outliers")

        # Add team labels for outliers
        for _, row in outliers.iterrows():
            axes[i].text(row[x_col], row[y_col], row["HomeTeam"], fontsize=9, ha="right", color="black")

        # Customize plot appearance
        axes[i].set_title(f"{season}: {x_col} vs {y_col} (Outliers Marked)")
        axes[i].set_xlabel(x_col)
        axes[i].set_ylabel(y_col)
        axes[i].grid(True)

        # Remove unwanted columns from the feature list
    myutils.features = [col for col in myutils.features if col not in ['GoalDifference', 'MarketConsensus']]

    # Ensure data_feat is properly defined by merging datasets
    data_feat = pd.concat(datasets.values(), ignore_index=True)

    # Drop unwanted columns
    data_feat = data_feat.drop(columns=['GoalDifference', 'MarketConsensus'], errors='ignore')

    # Select only numeric columns for correlation
    numeric_columns = data_feat.select_dtypes(include=['number']).columns

    # Compute correlation matrix using only numerical columns
    if not numeric_columns.empty:
        corr_matrix_feats = data_feat[numeric_columns].corr()
    else:
        print("⚠️ No numeric columns found for correlation!")

    # Debugging Check
    print("Numeric columns used for correlation:", numeric_columns)
    print("Correlation matrix shape:", corr_matrix_feats.shape if 'corr_matrix_feats' in locals() else "N/A")

    st.pyplot(fig)

    st.write('---')

    st.subheader("Parallel Coordinate Plot")
    
    st.write("""Here we analyse 3 top clubs in Arsenal, Chelsea and Man United.
              As expected, all three have averaged out over the last 5 years and
              statistically are pretty even in alot of aspects.
              From 'AC - away corners' onwards, Man United, they concede much
              more corners in comparison to Arsenal. You can then see that
              Man United's disciplinary record is better than both
              Chelsea and Arsenal. This aids in betting prediction as it shows 
              that Man United are less likely to recieve a yellow or red card 
              than Arsenal and Chelsea.""")

    # List of features to analyse
    features_to_analyse = [
            'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST',
            'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 
            'AvgH', 'AvgD', 'AvgA'
        ]

    teams_to_plot = ['Man United', 'Arsenal', 'Chelsea'] 

    filtered_data = myutils.getParallelPlot(data, features_to_analyse, teams_to_plot)
    print(data)
    # For data plotting
    x = np.arange(len(features_to_analyse))  # X-axis positions for features

    # Plot parallel oordinates with Bezier curves
    fig, ax = plt.subplots(figsize=(12, 6)) 
    print(filtered_data)
    for _, row in filtered_data.iterrows():
        team = row['HomeTeam']
        
        y = row[features_to_analyse].values  # Y-axis values for the features
        print(team, y)

        # Create Bezier curve (smooth line)
        x_smooth = np.linspace(x.min(), x.max(), 300)  # Smooth x-axis
        spline = make_interp_spline(x, y, k=3)  # Bezier spline (k=3 for cubic)
        y_smooth = spline(x_smooth)

        # Plot the curve
        plt.plot(x_smooth, y_smooth, alpha=0.7, label=team)

        # Add vertical lines for each metric
        for i in range(len(features_to_analyse)):
            plt.axvline(x=i, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)

        # Customise the plot
    plt.xticks(ticks=x, labels=features_to_analyse, rotation=90)  # Feature names as x-axis labels
    plt.xlabel('Metrics')
    plt.ylabel('Normalised Values')
    plt.title('Parallel Coordinates Plot (Averaged by HomeTeam)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.legend(loc='upper right')
    st.pyplot(fig)

    st.write('---')

    st.subheader("Team Comparison Radar Plot")

    st.write(""" 
    Radar plots are widely used in football analytics. Mainly due to their 
    ability to read multiple metrics in a straightforward and easy to read 
    layout. In this radar plot you can use the dropdown menu to choose which 
    teams you would like to compare. 
    """)

    Teams = ("Arsenal", "Aston Villa", "Bournemouth","Brentford","Brighton",
    "Crystal Palace","Chelsea","Everton","Fulham","Ipswich",
    "Nottingham Forest","Man City","Man United","Liverpool",
    "Tottenham","Newcastle","Southampton","Wolves","West Ham","Leicester",)

    # Team1 Team 2 select boxes
    team1 = st.selectbox(
        f"Select Team 1",
        Teams,
        key="0"
    )
    team2 = st.selectbox(
        f"Select Team 2",
        Teams,
        key="1"
    )

    st.write(f"Comparison: {team1} vs {team2}")

    st.write('---')

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

    fig, ax = myutils.plot_metrics(df, team1, team2, metrics, mapping)

    st.pyplot(fig)

    st.write(f"Match: {team1} vs {team2}")
    st.write('---')
    st.write("""
    Let's take Arsenal vs West Ham for example, below is an image of their 
    comparison. The 'home' data is for Arsenal and West Ham respectivley and 
    the away stats are the data of the teams Arsenal and West Ham have faced.
    """)

    st.image("./images/arswhuradarplot.png")

    st.write("""
    From this plot you can see Arsenal are much stronger at home and away than 
    West Ham. Arsenal are a high percentage on all home statistics, at a glance 
    it would appear that West Ham are stronger than Arsenal in away games. 
    However the plot is slightly misleading, as the data is actually showing 
    that West Ham actually *conceed* more SOT's, corners and goals away than 
    Arsenal do. You would look at this and come to a pretty quick conclusion 
    that if these teams were to play each other, your money should be on 
    Arsenal.
    """)

    st.write('---')
    
    st.subheader("Individual Team Comparison Radar Plot, Year By Year")

    # Select seasons from available datasets
    season1 = st.selectbox("Select First Season", list(datasets.keys()), key="season1")
    season2 = st.selectbox("Select Second Season", list(datasets.keys()), key="season2")

    # Ensure these variables are assigned before using them
    df1 = datasets[season1]  
    df2 = datasets[season2]

    # Dropdowns for team selection
    team = st.selectbox("Select a Team to Compare Across Seasons", Teams, key="team")

    # Ensure the dataset contains only relevant metrics
    selected_metrics = ["FTHG", "FTAG", "HS", "HST", "HC", "AS", "AST", "AC"]

    # Normalize the stats for team comparison
    normalized_stats = df1[selected_metrics].mean().to_frame().T  # Example normalization
    normalized_stats.index = [team]  # Ensure the team is set as index

    # Load the datasets for the selected seasons
    df1 = datasets[season1]
    df2 = datasets[season2]

    # Ensure the dataset contains the selected team
    df1_team = df1[df1["HomeTeam"] == team] if "HomeTeam" in df1.columns else pd.DataFrame()
    df2_team = df2[df2["HomeTeam"] == team] if "HomeTeam" in df2.columns else pd.DataFrame()

    # Debugging: Check if the DataFrames are empty
    if df1_team.empty:
        st.write(f"⚠️ Warning: No data found for {team} in {season1}.")
    if df2_team.empty:
        st.write(f"⚠️ Warning: No data found for {team} in {season2}.")

    # Ensure `normalized_stats` is valid before using `.loc[team]`
    if team not in normalized_stats.index:
        st.write(f"⚠️ Warning: {team} not found in normalized_stats index.")
        values = []
    else:
        values = normalized_stats.loc[team].values.flatten().tolist()

    # If values are empty, prevent plotting errors
    if not values:
        st.write("⚠️ No valid statistics found for the selected team. Please try another.")
    else:
        
        #fig = myutils.plot_metrics_season(df1_team, team, metrics, mapping)
        #fig = myutils.plot_metrics(df2_team, team, None, metrics, mapping)
        fig, ax = myutils.plot_metrics(df1, team1, None, metrics, mapping)
        fig, ax = myutils.plot_metrics(df2, team1, None, metrics, mapping, ax)
        st.pyplot(fig)
    
    st.write('---')
    
    st.subheader("Year by Year Comparison")

    st.write("Lets take a look at Arsenals 20/21 season, and compare it with last season, 23/24.")

    #st.image("./images/arsecomparison.png")

    st.write("""This shows a quite incredible change in form and results.
            In 20/21 Arsenals stats shown here are low. But they are only 
             exposed as such when you add last seasons stats on top.
             The home statistics are relatively close, however the major
             changes are obvious in the away stats, meaning Arsenals defence has
             drastically imporved over the 2 years. with goals conceded down by
             at least 20%, corners conceded and shots conceded all much lower, 
             and perhaps the biggest difference is the amount of goals arsenal
             scored in comparison to 20/21. This would indicate that Arsenal
             would be a strong contender for most bets your planning on placing.
             Especially when they are at home.""")

    

    