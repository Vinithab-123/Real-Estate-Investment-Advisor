# 🏡 Real Estate Investment Advisor: Predicting Property Profitability & Future Value

## Project Overview

This project delivers a machine learning solution designed to assist real estate investors in India. The application employs dual models—Classification and Regression—to provide data-backed investment intelligence via a user-friendly Streamlit web application.

### Key Objectives:
1.  **Classification:** Predict whether a property qualifies as a "**Good Investment**" based on price and infrastructure accessibility.
2.  **Regression:** Forecast the estimated property **Future Value (after 5 years)**.
3.  **Deployment:** Deploy an interactive analytical dashboard using **Streamlit**.

## 📊 Methodology & Models

The solution utilizes two XGBoost models trained on cleaned and engineered features from the provided dataset.

| Model Type | Purpose | Key Metric | Performance (Approximate) |
| :--- | :--- | :--- | :--- |
| **XGBoost Classifier** | Good Investment (Binary) | ROC AUC | ~0.88 |
| **XGBoost Regressor** | Future Price Forecast | R² Score | ~0.94 |

### Top 5 Feature Drivers
The analysis highlighted these features as the strongest predictors of future property price:
1.  **Price per SqFt:** Primary indicator of current property market standing.
2.  **City (One-Hot Encoded):** Location, particularly major metros, dictates value.
3.  **Size in SqFt:** Total size influences the base price.
4.  **Age of Property:** The age relative to the present year.
5.  **BHK:** The number of bedrooms, hall, and kitchen.

## 🚀 Deployment and Setup

The application is built using Streamlit and requires the model files (`.pkl`) and the data file to run locally.

### Prerequisites

* Python 3.8+
* `pip` package manager

### Installation and Run

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Vinithab-123/Real-Estate-Investment-Advisor.git](https://github.com/Vinithab-123/Real-Estate-Investment-Advisor.git)
    cd Real-Estate-Investment-Advisor
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You may need to create a `requirements.txt` file listing all libraries used, such as `streamlit`, `pandas`, `numpy`, `scikit-learn==1.3.2`, `xgboost`, and `joblib`.)*

3.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser at `http://localhost:8501`.

## 🛠 Technologies Used

* **Data Analysis:** Python, Pandas, NumPy
* **Machine Learning:** Scikit-learn Pipelines, XGBoost (Regressor and Classifier)
* **Deployment:** Streamlit
* **Version Control:** Git
 
* **Experiment Tracking:** MLflow

* **Experiment Tracking:** MLflow (used in the notebook)

 
