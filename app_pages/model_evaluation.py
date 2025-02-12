import streamlit as st
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import joblib
import myutils

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

save_path = "./model/"

def app():

    # Load Data
    st.title("Football Match Outcome Prediction")
    st.write("This app evaluates machine learning models (Random Forest & Neural Network) for football match predictions.")

    file_path = './jupyter_notebooks/data/full_dataset.csv'
    raw_data = pd.read_csv(file_path)
    normalised_data, label_encoder = myutils.preprocess_data(raw_data)

    features = [
        'HomeTeam', 'AwayTeam', 'HomeGoalsAvg', 'HomeWinPct', 'AwayWinPct',
        'GoalDifference', 'Implied_Prob_A', 'Expected_Goals', 'Market_Expectation'
    ]

    X = normalised_data[features]
    y = normalised_data['FTR']  # 0=H, 1=D, 2=A

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, save_path + 'scaler.pkl')

    st.subheader("Random Forest Model Training")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    st.write(f"Random Forest Accuracy: {rf_accuracy:.3f}")

    # Display classification report as a DataFrame
    def classification_report_to_df(report):
        report_dict = classification_report(y_test, y_pred_rf, output_dict=True)
        return pd.DataFrame(report_dict).transpose()

    st.subheader("Classification Report")
    st.dataframe(classification_report_to_df(classification_report(y_test, y_pred_rf)))

    # Feature Importance
    importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False)
    st.subheader("Feature Importance")
    fig, ax = plt.subplots()
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Random Forest Feature Importances")
    st.pyplot(fig)

    # Show Neural Network Architecture
    num_classes = len(np.unique(y))
    y_train_oh = to_categorical(y_train, num_classes=num_classes)
    y_test_oh = to_categorical(y_test, num_classes=num_classes)

    nn_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    st.subheader("Neural Network Architecture")

    # Extract layer names and parameters
    layer_names = [layer.name for layer in nn_model.layers]
    layer_params = [layer.count_params() for layer in nn_model.layers]
    st.write("Sequential Model:")
    st.write("Layers:")
    st.write(layer_names)
    st.write("Parameters:")
    st.write(layer_params)    
        
    # Neural Network Training
    if st.button("Train Neural Network"):
        st.subheader("Neural Network Training:")
        nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse', 'mae'])

        epochs = 50
        st.write(f"Neural Network training epochs: {epochs}")
        # Display a progress bar
        accuracy_results = []
        loss_results = []
        progress_bar = st.progress(0)
        for epoch in range(epochs):
            history = nn_model.fit(X_train_scaled, y_train_oh, validation_split=0.2, epochs=1, batch_size=8, verbose=0)
            progress_bar.progress((epoch + 1) / epochs)
            accuracy_results.append(history.history['accuracy'])
            loss_results.append(history.history['loss'])
        
        loss, acc, mse, mae = nn_model.evaluate(X_test_scaled, y_test_oh, verbose=0)

        st.write(f"Neural Network Accuracy: {acc:.3f}")
        st.write(f"Validation loss: {loss:.3f}")

        st.write(f"Mean Squared Error: {mse:.3f}")
        st.write(f"Mean Average Error: {mae:.3f}")
        
        nn_model.save(save_path + 'nn_model.h5')

        # Plot Training History
        epochs = range(1, len(accuracy_results) + 1)
        fig, ax = plt.subplots()
        ax.plot(epochs, accuracy_results, label='Train Accuracy')
        ax.plot(epochs, loss_results, label='Validation Accuracy')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Neural Network Accuracy per Epoch")
        ax.legend()
        st.pyplot(fig)
