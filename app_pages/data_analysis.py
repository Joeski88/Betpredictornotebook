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

import myutils

    # Feature selection
features = [
        'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST',
        'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 
        'AvgH', 'AvgD', 'AvgA'
]

# Load dataset
file_path = "./jupyter_notebooks/data/full_dataset.csv"
data = pd.read_csv(file_path)
df =  pd.read_csv("./jupyter_notebooks/data/full_dataset.csv")
# notebook_dir = os.path.dirname(os.path.abspath("__file__"))  # Simulates the notebook's directory
# print(notebook_dir)
# data =  pd.read_csv(notebook_dir + "./jupyter_notebooks/data/full_datasetv2.csv")

def app():
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
        short for Expected Goals, however i have decided against using this
        metric as this is a fairly recent statistic, and wont go back far enough 
        which therefore makes it  hard as i want to use as much historical data
        as possible.
        """)

    st.write(df.describe())

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
    #calculateOverallPerformance(data)

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
    print(data_feat.head())  # Check if data_feat is a valid DataFrame

    corr_matrix_feats = data_feat[features].corr()
    corr_plot2 = sb.heatmap(corr_matrix_feats, cmap="YlGnBu", annot=False)
    fig = corr_plot2.get_figure()
    st.pyplot(fig)

    # # Split data into features and target
    # X = data.drop('HomeTeam', axis=1)
    # y = data['AwayTeam']

    # # Split into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Label encode non-numeric columns
    # label_encoder = LabelEncoder()
    # X_train['HomeTeam'] = label_encoder.fit_transform(X_train['HomeTeam'])
    # X_test['AwayTeam'] = label_encoder.transform(X_test['AwayTeam'])

    # # Initialize and train the model
    # model = RandomForestClassifier(random_state=42)
    # model.fit(X_train, y_train)

    # # Predict and evaluate
    # predictions = model.predict(X_test)
    # print(predictions)

    # Feature engineering
    # data['GoalDifference'] = data['FTHG'] - data['FTAG']
    # data['MarketConsensus'] = (data['AvgH'] + data['AvgD'] + data['AvgA']) / 3
    # features.extend(['GoalDifference', 'MarketConsensus'])
    
    # # Get feature importances from the model
    # feature_importances = pd.DataFrame({
    #     'Feature': features,
    #     'Importance': model.feature_importances_
    # }).sort_values(by='Importance', ascending=False)

    # # Plot the feature importances
    # plt.figure(figsize=(10, 6))
    # plt.barh(feature_importances['Feature'], feature_importances['Importance'])
    # plt.gca().invert_yaxis()  # Invert y-axis to show most important feature at the top
    # plt.title('Feature Importances')
    # plt.xlabel('Importance')
    # plt.ylabel('Feature')
    # plt.show()

    # # Select only numeric columns
    # numerical_data = data.select_dtypes(include=['number'])

    # # Debug: Check if numerical_data is empty
    # if numerical_data.empty:
    #     print("No numeric columns found in the dataset.")
    # else:
    #     print(numerical_data.info())

    # # Compute correlation matrix
    # correlation_matrix = numerical_data.corr()
    # corr_matrix = cleaned_data.corr()
    # plt.figure(figsize=(15, 10))
    # sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    # st.pyplot(plt.gcf())

    # # Debug: Check if correlation_matrix is empty
    # print("Correlation matrix:")
    # print(correlation_matrix)

    # if correlation_matrix.empty:
    #     print("Correlation matrix is empty. Ensure dataset has multiple numeric columns.")
    # else:
    #     plt.title("Correlation Matrix")
    #     plt.show()

