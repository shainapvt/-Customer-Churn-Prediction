
# Customer Churn Prediction

## Project Overview
Customer Churn Prediction is a machine learning project that predicts whether a customer is likely to leave (churn) a company based on their usage behavior and account information. Retaining existing customers is often more cost-effective than acquiring new ones, making churn prediction an important business application.

This project uses a **Random Forest classifier** trained on the **Telco Customer Churn dataset** and includes an **interactive Gradio interface** for testing predictions.

---

## Features

- **Data Preprocessing:** Handles missing values, categorical encoding, and feature selection.  
- **Model Training:** Random Forest Classifier with evaluation metrics like accuracy, precision, recall, and confusion matrix.  
- **Feature Importance:** Visualizes which features are most influential in predicting churn.  
- **Interactive Prediction:** Simplified Gradio interface that allows users to enter key customer details and get churn predictions with probabilities.  
- **Colab-Friendly:** Fully runs in Google Colab without heavy local computation.

---

## Dataset

- **Source:** [Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- **Contents:**  
  - Customer demographics (age, gender, senior citizen, partner, dependents)  
  - Account information (tenure, contract type, payment method)  
  - Service usage (internet, phone, tech support, streaming)  
  - Churn status (Yes/No)

---

## Installation & Usage

### Requirements
- Python 3.x
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `gradio`

### Run in Google Colab
1. Open the notebook in [Google Colab](https://colab.research.google.com/).  
2. Mount Google Drive and upload the dataset.  
3. Run all cells to train the model, evaluate it, and launch the Gradio interface.  
4. Use the simplified Gradio form to test churn predictions interactively.

---

## How It Works

1. **Load Dataset:** Load customer data from CSV.  
2. **Preprocess Data:** Encode categorical variables and fill missing values.  
3. **Train Model:** Use Random Forest to classify customers as churned or not.  
4. **Evaluate Model:** Check model performance with accuracy, classification report, and confusion matrix.  
5. **Feature Importance:** Visualize the most important features affecting churn.  
6. **Interactive Prediction:** Enter customer details in Gradio interface to get predictions.

---

## Key Features Used in Simplified Interface
- Tenure (months)  
- Monthly Charges  
- Contract Type (One Year / Two Year)  
- Internet Service (Fiber Optic)  
- Tech Support  

---

## Results
- Model Accuracy: Typically around **79-82%** on test set (can vary with train/test split).  
- Feature Importance highlights **tenure, contract type, tech support, and internet service** as the most predictive features.  
- Interactive Gradio interface provides a user-friendly way to predict churn with probability scores.

---

## Future Improvements
- Test with **XGBoost or LightGBM** for potentially higher accuracy.  
- Handle **class imbalance** with SMOTE or class weighting.  
- Deploy as a **web app** for real-time prediction in production.  
- Integrate more **customer behavioral data** for better predictions.

---


