import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report

def app():
    st.title("Football betting notebook")

    st.write("Data driven betting predictions")
    
    # Load the dataset
    file_path = "./data/full_dataset.csv"  # Replace with your dataset path
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
