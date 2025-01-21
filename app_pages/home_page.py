import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report

def app():
    st.title("Football betting notebook")

    st.image("./images/premlogo.png")

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
    
    # Load the dataset
    file_path = "./jupyter_notebooks/data/full_dataset.csv"  # Replace with your dataset path
    data = pd.read_csv(file_path)

    # Data Preprocessing
    # Handle missing values
    data.fillna(method='ffill', inplace=True)
    
    st.write(data.head())

    # Define a function for performing EDA
    print("Dataset Head:\n", data.head())
    print("Dataset Info:\n")
    data.info()
    print("Summary Statistics:\n", data.describe())

    # Plot histograms for numeric columns
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns].hist(figsize=(15, 10), bins=20, color='skyblue', edgecolor='black')
    plt.suptitle('Histograms of Numeric Features', fontsize=16)
    plt.tight_layout()
    st.pyplot(plt.gcf())
