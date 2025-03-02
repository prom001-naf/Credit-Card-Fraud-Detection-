# Credit Card Fraud Detection Dashboard

This project provides a web-based dashboard for analyzing credit card transactions and detecting fraud. It integrates data analysis via Plotly, geospatial visualization using Folium, and machine learning predictions using a pre-trained model.

## Project Structure

- **.gitignore**  
  Specifies files and directories to ignore in version control.

- **credit-card-transactions-dataset.zip**  
  Compressed dataset downloaded and extracted in the **Data/** folder.

- **Data/**  
  Contains the extracted dataset file: `credit_card_transactions.csv`.

- **FraudDetection_Colab.ipynb**  
  A Jupyter Notebook demonstrating EDA and model evaluation approaches.

- **README.md**  
  This file.

- **UI/**  
  Contains application source files:
  - [`app.py`](UI/app.py): The Flask backend that serves the dashboard.
  - `city_population.json`: JSON file with population data used in predictions.
  - `credit_card_transactions.csv`: A copy of the transactions data (if needed).
  - `fraud_model.joblib`: The pre-trained machine learning model.
  
  - **UI/static/**  
    Contains downloadable files:
    - `files/analysis_notebook.ipynb`: Jupyter notebook for analysis.
    - `files/fraud_model.joblib`: Exported fraud detection model file.
    - `files/training_notebook.ipynb`: Jupyter notebook used to train the model.
    
  - **UI/templates/**  
    Contains HTML templates:
    - [`dashboard.html`](UI/templates/dashboard.html): The main dashboard view.

## Features

- **Data Analysis Visualizations**  
  Line charts, box plots, and histograms created with Plotly to visualize trends in fraudulent activities and demographic distributions.

- **Geospatial Visualizations**  
  Interactive maps using Folium to display locations of fraudulent transactions.

- **Fraud Prediction**  
  A machine learning model (RandomForestClassifier) that predicts the likelihood of a transaction being fraudulent.

## Requirements

- Python 3.x
- Flask
- Pandas
- Plotly
- Folium
- Joblib
- NumPy
- Other dependencies may be installed using `pip`

## Setup and Installation

1. **Clone the Repository**

   ```sh
   git clone <repository_url>
   cd Credit-Card-Fraud-Detection-
