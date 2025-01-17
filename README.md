
## Cloud IDE Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to _Account Settings_ in the menu under your avatar.
2. Scroll down to the _API Key_ and click _Reveal_
3. Copy the key
4. In the terminal, run `heroku_config`
5. Paste in your API key when asked


You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you so do not share it. If you accidentally make it public then you can create a new one with _Regenerate API Key_.


## Dataset Content
* Describe your dataset. Choose a dataset of reasonable size to avoid exceeding the repository's maximum size and to have a shorter model training time. If you are doing an image recognition project, we suggest you consider using an image shape that is 100px × 100px or 50px × 50px, to ensure the model meets the performance requirement but is smaller than 100Mb for a smoother push to GitHub. A reasonably sized image set is ~5000 images, but you can choose ~10000 lines for numeric or textual data. 


## Business Requirements
* Describe your business requirements


## Hypothesis and how to validate?
* List here your project hypothesis(es) and how you envision validating it (them) 


## The rationale to map the business requirements to the Data Visualizations and ML tasks
* List your business requirements and a rationale to map them to the Data Visualizations and ML tasks


## ML Business Case
* In the previous bullet, you potentially visualized an ML task to answer a business requirement. You should frame the business case using the method we covered in the course 


## Dashboard Design
* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other item that your dashboard library supports.
* Later, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but subsequently you used another plot type).



## Unfixed Bugs
* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable to consider, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment
### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* Set the runtime.txt Python version to a [Heroku-24](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.


## Main Data Analysis and Machine Learning Libraries
* Here you should list the libraries you used in the project and provide an example(s) of how you used these libraries.


## Credits 

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open-Source site
- The images used for the gallery page were taken from this other open-source site



## Acknowledgements (optional)
* Thank the people who provided support through this project.

## Project Overview
The **Sports Betting Predictor Dashboard** is a Streamlit-based application that leverages predictive analytics to provide betting insights for sports enthusiasts. This dashboard allows users to upload data, perform exploratory analysis, generate predictions for match outcomes, and evaluate model performance.

---

## Features
- **Data Upload:** Upload and preprocess sports datasets (CSV/Excel).
- **Exploratory Data Analysis (EDA):** Interactive data visualization and analysis tools.
- **Predict Outcomes:** Generate single-event or batch predictions.
- **Model Evaluation:** Analyze model performance with metrics and visualizations.
- **Settings:** Customize preferences like model selection and appearance.

---

## User Stories

### 1. Upload and Clean Data
**As a** user, **I want** to upload and clean my dataset **so that** I can ensure the data is ready for analysis.
- **Acceptance Criteria:**
  - Users can upload CSV/Excel files.
  - Users can preview and clean the data by removing missing values.
  - Error messages are displayed for invalid files.

### 2. Explore Data
**As a** user, **I want** to explore trends and insights in my data **so that** I can understand its structure and key statistics.
- **Acceptance Criteria:**
  - Users can visualize data using bar charts, line charts, and heatmaps.
  - Filters allow users to select specific columns for analysis.

### 3. Predict Match Outcomes
**As a** user, **I want** to predict the outcome of matches **so that** I can make informed betting decisions.
- **Acceptance Criteria:**
  - Users can input team names for single-event predictions.
  - Users can upload batch files for bulk predictions.
  - Predictions are displayed with probabilities and downloadable as CSV files.

### 4. Evaluate Model Performance
**As a** user, **I want** to review the model's performance metrics **so that** I can assess its reliability.
- **Acceptance Criteria:**
  - Users can view accuracy, precision, recall, and ROC curve visualizations.

### 5. Customize Settings
**As a** user, **I want** to configure the app's appearance and default model **so that** it aligns with my preferences.
- **Acceptance Criteria:**
  - Users can select a default predictive model.
  - Users can toggle between light and dark themes.

---

## Dashboard Design (Streamlit UI)

### Sidebar Navigation
- Home
- Data Upload
- Exploratory Data Analysis (EDA)
- Predict Outcomes
- Model Evaluation
- Settings

### Pages Overview

#### **Home**
- Welcome message and app overview.
- Key statistics like total predictions, accuracy, and confidence range.

#### **Data Upload**
- File uploader for CSV/Excel files.
- Data cleaning options and preview.

#### **Exploratory Data Analysis (EDA)**
- Column selector for visualizations.
- Charts and heatmap for correlation analysis.

#### **Predict Outcomes**
- Input for single-match predictions (Team A vs. Team B).
- Batch prediction uploader with downloadable results.

#### **Model Evaluation**
- Performance metrics (accuracy, precision, recall).
- ROC curve visualization.

#### **Settings**
- Model selection (e.g., Logistic Regression, Random Forest).
- Appearance options (light/dark theme).

---

## Business Case Assessment

### Objectives
- Enable users to make data-driven decisions for sports betting.
- Provide accessible tools for exploring and analyzing sports data.
- Offer an interactive and user-friendly interface for predictions and insights.

### Benefits
1. **Enhanced Decision-Making:** Users gain insights into data trends and model predictions.
2. **User Engagement:** The interactive interface fosters exploration and better understanding of predictions.
3. **Efficiency:** Automates data cleaning, visualization, and predictive analytics, saving time.

### Risks
- **Data Quality:** Users may upload incomplete or incorrect datasets.
- **Model Accuracy:** Predictions depend on the quality and relevance of training data.
- **Technical Knowledge:** Some users may require guidance to interpret visualizations and predictions.

### Proposed Solution
Develop a robust, easy-to-use dashboard with clear instructions, interactive visualizations, and reliable predictive models. Provide error handling and informative feedback to users for a seamless experience.

---

## Sources

#### Images

##### All images are taken from google image search, source provided below.

- Premier League Logo - https://www.nairaland.com/3683655/official-premier-league-news-thread
- Betting images - 

## How to Run the Project
1. Install the required Python packages:
   ```bash
   pip install streamlit pandas matplotlib seaborn scikit-learn
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Navigate through the app using the sidebar to explore its features.

---

## Contributing
We welcome contributions! Please open an issue or submit a pull request to improve the project.


