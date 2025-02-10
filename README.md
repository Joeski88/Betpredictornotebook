
## Project Overview
The **Sports Betting Predictor Dashboard** is a Streamlit-based application that leverages predictive analytics to provide betting insights for sports enthusiasts. This dashboard allows users to upload data, perform exploratory analysis, generate predictions for match outcomes, and evaluate model performance.

---

## Features
- **Data Upload:** Upload and preprocess sports datasets (CSV/Excel).
- **Exploratory Data Analysis (EDA):** Interactive data visualization and analysis tools.
- **Predict Outcomes:** Generate single-event or batch predictions.
- **Model Evaluation:** Analyze model performance with metrics and visualizations.

---

## User Stories

### 1. Upload and Clean Data
**As a** user, **I want** to upload and clean my dataset **so that** I can ensure the data is ready for analysis.
- **Acceptance Criteria:**
  - Upload required datasets.
  - Users can preview and clean the data by removing missing values.
  - Error messages are displayed for invalid files.

### 2. Explore Data
**As a** user, **I want** to explore trends and insights in my data **so that** I can understand its structure and key statistics.
- **Acceptance Criteria:**
  - Users can visualise data using bar charts, line charts, and plots.
  - Filters allow users to select specific columns for analysis.

### 3. Predict Match Outcomes
**As a** user, **I want** to predict the outcome of matches **so that** I can make informed betting decisions.
- **Acceptance Criteria:**
  - Users can input team names for single-event predictions.
  - Users can upload batch files for bulk predictions.
  - Predictions are displayed with probabilities.

### 4. Evaluate Model Performance
**As a** user, **I want** to review the model's performance metrics **so that** I can assess its reliability.
- **Acceptance Criteria:**
  - Users can view accuracy, precision, recall, and ROC curve visualizations.

---

## Dashboard Design (Streamlit UI)

### Sidebar Navigation
- HomeProject Summary/Home
- Data Study
- Hypothesis and Analysis
- Model Performance
- Prediction
- Conclusion

### Pages Overview

#### **HomeProject Summary/Home**
- Welcome message and app overview.
- Data set summary.
- Business requirements.

#### **Data Study**
- Correlation Matrix
- Data cleaning options and preview.
- Outliers
- Team Comparisons.

#### **Hypothesis and Analysis**
- What the data shows.
- Charts and Plots for analysis.
- Single team analysis.

#### **Model Performance**
- Performance metrics (accuracy, precision, recall).
- ROC curve visualization

#### **Prediction**
- Input for single-match predictions (Team A vs. Team B).
- Batch prediction uploader with changeable metrics, RAB or request a bet
predictions

#### **Conclusion**
- What Have we found out, what have we learnt.

---

## Business Case Assessment

### Objectives
- Enable users to make data-driven decisions for sports betting.
- Provide accessible tools for exploring and analysing sports data.
- Offer an interactive and user-friendly interface for predictions and insights.

### Benefits
1. **Enhanced Decision-Making:** Users gain insights into data trends and model predictions.
2. **User Engagement:** The interactive interface fosters exploration and better understanding of predictions.
3. **Efficiency:** Automates data cleaning, visualization, and predictive analytics, saving time.

### Risks
- **Data Quality:** Dataset may not be big enough or thorough enough to give a good enough prediction.
- **Model Accuracy:** Predictions depend on the quality and relevance of training data.
- **Technical Knowledge:** Some users may require guidance to interpret visualizations and predictions.

### Proposed Solution
Develop a robust, easy-to-use dashboard with clear instructions, interactive visualizations, and reliable predictive models. Provide error handling and informative feedback to users for a seamless experience.

---

## Sources

## Libraries/Dependencies

- **Pandas:**
  - Used for data manipulation and analysis, especially for handling structured data in DataFrames.
  
- **NumPy:**
  - Essential for numerical computing in Python, used for efficient data storage and manipulation.
  - Provides support for large, multi-dimensional arrays and datasets.
  
- **Matplotlib:**
  - A popular library for creating static, interactive, and animated visualizations in Python.
  - Used primarily for generating plots like scatter plots, histograms, and line graphs in this project.

- **Seaborn:**
  - A data visualization library based on Matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics.
  - Used for generating heatmaps and correlation plots to understand relationships between features.

- **Scikit-learn:**
  - Used for splitting the dataset, preprocessing data, and evaluating machine learning models.
  
- **XGBoost:**
  - A powerful, efficient, and scalable implementation of gradient boosting framework.
  - Due to its size, it is excluded from the deployed app to avoid increasing slug size but is used during model training and development. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#### Images

##### All images are taken from google image search, source provided below.

- Premier League Logo - https://www.nairaland.com/3683655/official-premier-league-news-thread
- Betting images - Google search

### To use this repository, follow these steps:

1. **Fork or Clone the Repository**

   First, fork or clone this repository to your local machine:

   ```bash
   git clone github.com/Joeski88/Betpredictornotebook
   cd betpredictornotebook
   ```

2. **Install Dependencies**

   Install the required dependencies by running one of the following commands, depending on your environment:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up the Datasets**

   Ensure the dataset paths are correct.

4. **Run the Streamlit Dashboard**

   Start the Streamlit web app by running the following command:

   ```bash
   streamlit run app.py
   ```
   This will launch the interactive dashboard where you can visualise data, analyse predictions, and see model performance.

---

## Bugs

- ![features](/images/pyplotfigbug.png)

- ![features](/images/utilsbug.png)
---

## Credits

This project was made possible by the support and guidance of several key resources and contributors. Special thanks to the following:

- **[Football dataset](https://www.football-data.co.uk/englandm.php)**: For providing the dataset used in this project.
- **[Streamlit documentation](https://docs.streamlit.io/)**: For guiding the development of the interactive web app.
- **[Pandas documentation](https://pandas.pydata.org/docs/)**: For guiding the development of the interactive web app.
- **[XGBoost documentation](https://xgboost.readthedocs.io/)**: For providing resources on the XGBoost machine learning library, which was used for model training and evaluation.
- **[Heroku documentation](https://devcenter.heroku.com/)**: For deploying the Streamlit app
- **[Code Institute](https://learn.codeinstitute.net/ci_program/sppredan2024_4)**: Lessons and notebooks on course content essential to progress. 
 

All third-party libraries and frameworks used in the project have been credited in the `requirements.txt`



