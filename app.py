import streamlit as st
from app_pages.multipage import MultiPage
from app_pages import data_analysis, home_page, model_evaluation, conclusion, prediction, hypothesis

# Initialize the multipage app
app = MultiPage("Football Betting Predictor")

# Add pages to the app
app.add_page("Project Summary 🏠", home_page.app)
app.add_page("Data Study 📝", data_analysis.app)
app.add_page("Hypothesis and Analysis 📌", hypothesis.app)
app.add_page("Model Performance 📈", model_evaluation.app)
app.add_page("Prediction 🔮", prediction.app)
app.add_page("Conclusion 🎯", conclusion.app)

# Run the app
app.run()