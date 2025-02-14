import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sb
import myutils
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = "./jupyter_notebooks/data/full_dataset.csv"
data = pd.read_csv(file_path)

def app():
    # st.image("./images/premlogo.png")
    
    st.title("Football betting notebook")

    st.header("Project Summary")

    st.subheader("**Objective**")
    st.write(
        """
        The Betting Tips Predictor Jupyter Notebook project is a predictive 
        analytics tool designed to analyze historical sports data and provide 
        informed betting recommendations. By leveraging machine learning and 
        statistical models, the project aims to empower users with data-driven 
        insights to enhance decision-making in the sports betting industry.
        This initiative offers a cost-effective and scalable solution for both 
        casual bettors and organizations looking to optimize their betting 
        strategies.
        """
    )

    st.subheader("Business Requirements")

    st.write(
        """
        - Develop a Jupyter Notebook with a modular structure for easy 
          scalability.
        - Implement predictive models for win/loss probabilities and score 
          forecasting.
        - Incorporate data visualization tools for insightful and actionable 
          outputs.
        - Ensure reproducibility and transparency in the analytical process.
        - Deliver a user-friendly interface to cater to both technical and 
          non-technical users.

        """
    )

    st.header("Data Set Summary")
    st.write("The link for the dataset key is located beneath.")
    # link for data set key
    st.page_link("https://www.football-data.co.uk/notes.txt", 
                label="Dataset Key", icon="⚽")
    st.write(
        """
        The data set is compiled from football statistics collected from over 
        the last few years. they consist of various metrics, including:
        - Goals scored
        - Goals conceded
        - Corners for, and corners against
        - Fouls for and against
        - Booking points

        The list is extensive and there are many more. See more information and 
        see the abbreviations used broken down and explained in the above link. 
        """
    )

    # Handle missing values
    data.ffill(inplace=True)

    st.write(data.head())
    
    st.subheader("Overall Win Percentage Bar Plot")  
    data_org, win_percentage = myutils.calculateOverallPerformance(data)

    fig = plt.figure(figsize=(12, 6))
    win_percentage.plot(kind='bar', color='skyblue', alpha=0.8, edgecolor='black')
    plt.title('Team Win Percentage')
    plt.xlabel('Team Name')
    plt.ylabel('Win Percentage (%)')
    plt.xticks(rotation=90, fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("""
    This is an overall win percentage from premier league teams over the last 5 
    years. Immediately you can make a fairly obvious prediction based on this 
    stat alone. You can bet that City, Liverpool and Arsenal will all feature 
    highly as predicted winners with each having a percentage of 70%, 65% & 58% 
    respectivley.

    """)

