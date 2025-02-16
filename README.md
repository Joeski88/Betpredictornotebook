# Bet Predictor App Readme

![Responsive Mockup](/images/mockup.png)

*View the live project here: [BetPredictorApp](https://betpredictornotebook.onrender.com/)*

## Project Overview
The **Sports Betting Predictor** is a Streamlit-based application that leverages predictive analytics to provide betting insights for sports enthusiasts. This dashboard allows users to upload data, perform exploratory analysis, generate predictions for match outcomes, and evaluate model performance.

#### I must stress, due to having to use render to deploy, it can take a while to load the page. (Roughly 3 minutes when I load it)

---

## Features
- **Data Upload:** Upload and preprocess sports datasets (CSV/Excel).
- **Exploratory Data Analysis (EDA):** Interactive data visualization and analysis tools.
- **Predict Outcomes:** Generate single-event or batch predictions.
- **Model Evaluation:** Analyse model performance with metrics and visualizations.

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
averages to base your prediction.

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
  - Used for splitting the dataset, pre processing data, and evaluating machine learning models.
  
- **XGBoost:**
  - A powerful, efficient, and scalable implementation of gradient boosting framework.

#### Images

##### All images are taken from google image search, source provided below.

- Premier League Logo - https://www.nairaland.com/3683655/official-premier-league-news-thread

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

## Deploying Your App to Render

Render is a cloud platform that allows you to deploy web apps, static sites, and more with ease. This guide will walk you through the process of deploying your app to Render.

### Prerequisites
Before you begin, make sure you have:

A Render account. 
A GitHub or GitLab account with your project repository.
Your app's dependencies defined (e.g., requirements.txt for Python projects).
Your app should be compatible with Render's environment (e.g., HTTP web service or background worker).
  - #### Step 1: Push Your Project to GitHub/GitLab
If you have not already, push your project to a GitHub or GitLab repository.
Make sure your repository contains all the necessary files for deployment, such as:
requirements.txt (for Python projects).
Any other configuration files specific to your app.
  - #### Step 2: Create a New Web Service on Render
Go to Render and log into your account.
Click on the New button in the upper-left corner of the dashboard and select Web Service.
Select Connect Account and link your GitHub or GitLab account to Render.
Once your account is connected, Render will show your repositories. Select the repository you want to deploy.
Render will ask you to configure the web service. Fill in the details:
Name: Give your web service a name.
Environment: Choose your environment (e.g., Python for Python projects).
Branch: Select the branch you want to deploy (typically main).
Build Command: The build command to set up the environment (e.g., pip install -r requirements.txt for Python projects).
Start Command: The command that runs your application (e.g., python app.py or flask run).
Region: Choose the region closest to your users.
  - #### Step 3: Set Up Environment Variables (Optional)
If your app requires environment variables (like API keys or secrets), you can set them up during deployment:
In the Environment Variables section of the Render configuration page, click Add Environment Variable.
Enter the name and value for each environment variable.
  - #### Step 4: Deploy Your App
After filling out the configuration, click Create Web Service.
Render will start building and deploying your app.
It will clone your Git repository, install dependencies, and run the build command.
You can track the progress of the deployment in the Logs section.
  - #### Step 5: Verify Your App
Once the deployment is complete, Render will provide you with a URL to access your app.
Click on the provided URL to verify that your app is running correctly.
  - #### Step 6: Update Your Deployment
If you make any changes to your code (e.g., bug fixes, new features), push those changes to your Git repository.
Render will automatically rebuild and redeploy your app whenever changes are pushed to the connected repository.

## Bugs

Throughout this project, I was exposed to many new and more familiar bugs. It would have been impossible for me to track them all. So I will focus on the bigger bugs and problems I encountered.

1 - The first was when creating the myutils.py page. Struggled to incorporate that page with the rest of the app. This was because I failed to import at the top of the page. 

![features](/images/utilsbug.png)
2 - The second, I managed to get the plots to display in data analysis, but no data was being displayed. My file paths were in the wrong place, located inside the main app function. Organising the datasets better and making sure they were properly indented was key to fixing the issue.

3 - The biggest one I encountered was by far the deployment to heroku. My slug size was way to large and it made it impossible to deploy. Tried everything, pruning, clearing un needed files and folders. The support team at CI on slack pointed me in the direction of Render. This fixed the deployment issue, however i was unable to run the streamlit app.I would get this error message, the issue here was me following the instructions to literally, and entered the wrong python version when deploying to render.

![features](/images/renderbug.png)



---

## Unfixed Bugs

There were a couple of things I just could not work out how to fix/change, tried all I could. Here are the culprits:

1 - The first issue was in the prediction page, where on the lower radar plot, if you add all of the metrics into the plot at once you get a "pop" error. However if you leave a second or 2 between each addition it works fine. Even when googling the error message its still not really clear to me what the issue is.

![features](/images/popbug.png)

2 - The second being on the same radar plot. As I am taking an average number over the last 5 seasons, the results are displayed as a percentage instead of an integer.
The result ends in roughly 10% = 1. 100% = 10. As stated on the streamlit app's prediction page.

3 - Lastly, theres a parallel coordinate plot in the data analysis page. For some strange reason I have to define the data sets again before it. Again, unknown reason as to why.
This is noted in a comment in the python script.


## Credits

This project was made possible by the support and guidance of several key resources and contributors. Special thanks to the following:

- **[Football dataset](https://www.football-data.co.uk/englandm.php)**: For providing the dataset used in this project.
- **[Streamlit documentation](https://docs.streamlit.io/)**: For guiding the development of the interactive web app.
- **[Pandas documentation](https://pandas.pydata.org/docs/)**: For guiding the development of the interactive web app.
- **[XGBoost documentation](https://xgboost.readthedocs.io/)**: For providing resources on the XGBoost machine learning library, which was used for model training and evaluation.
- **[Code Institute](https://learn.codeinstitute.net/ci_program/sppredan2024_4)**: Lessons and notebooks on course content essential to progress. 
 
## Librarys

### Main Data Analysis Libraries:

- **Pandas:**
  - Used for data manipulation and analysis, especially for handling structured data in DataFrames.
  - Key functions: handling missing values and generating summary statistics
  
- **NumPy:**
  - Essential for numerical computing in Python, used for efficient data storage and manipulation.
  - Provides support for large, multi-dimensional arrays and matrices.
  
- **Matplotlib:**
  - A popular library for creating static, interactive, and animated visualizations in Python.
  - Used primarily for generating plots like scatter plots, histograms, and line graphs in this project.

- **Seaborn:**
  - A data visualization library based on Matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics.
  - Used for generating heatmaps and correlation plots to understand relationships between features.

### Main Machine Learning Libraries:

- **Scikit-learn:**
  - Used for splitting the dataset, pre processing data, and evaluating machine learning models.
  
- **XGBoost:**
  - A powerful, efficient, and scalable implementation of gradient boosting framework.
  - Utilized for training the regression model to predict house sale prices.

- **Random Forest:**
  - Random Forest is an ensemble learning method that can be used for both classification and regression tasks.
  - Handles both categorical and numerical data, random feature selection & ensemble learning.

### Other Libraries:
  
- **Streamlit:**
  - Used to build the interactive web dashboard for visualizing data, making predictions, and displaying model performance metrics.

These libraries were crucial in building an end-to-end machine learning solution, from data pre processing to model deployment.

###  Acknowledgments

- Mentor Luke, been very calming and encouraging presence. Much needed. 

- My family, have helped me get through this challenge, and given me all the support possible. Including with childcare and moral support. 

All third-party libraries and frameworks used in the project have been credited in the `requirements.txt`



