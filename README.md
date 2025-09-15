# 🚢 Titanic Survival Predictor

A machine learning project that predicts passenger survival on the RMS Titanic with **83.24% accuracy**. This project demonstrates the full end-to-end data science pipeline: from data cleaning and exploratory analysis to feature engineering, model training, and deployment.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2.2-orange)

## 📊 Project Overview

The sinking of the Titanic is one of the most infamous shipwrecks in history. This project builds a classifier to predict whether a passenger survived the disaster based on attributes like age, gender, socio-economic class, etc.

The final model is a **RandomForestClassifier** tuned via GridSearch, achieving strong performance by leveraging key engineered features like passenger titles and cabin presence.

## 🏆 Model Performance

The final model was evaluated on a held-out test set (20% of the data).

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **83.24%** |
| Precision (Survived) | 81% |
| Recall (Survived) | 76% |
| F1-Score (Survived) | 78% |

![Confusion Matrix](images/confusion_matrix.png) <!-- (Upload your plot first) -->
*Confusion Matrix showing model predictions vs. actual outcomes.*

## 🔍 Key Insights from EDA

The correlation heatmap and feature importance analysis revealed the strongest predictors of survival:

1.  **Sex**: The strongest predictor (Correlation: 0.54). Females had a dramatically higher survival rate.
2.  **Passenger Class (Pclass)**: Strong negative correlation (-0.34). 1st-class passengers had priority to lifeboats.
3.  **Fare**: Weak positive correlation (0.26). Higher fares correlated with higher survival.
4.  **Title (Engineered Feature)**: Titles like 'Master' (young boys) and 'Mrs' had high survival rates, while 'Mr' had a very low rate.
5.  **Has_Cabin (Engineered Feature)**: Passengers with a recorded cabin number (likely higher socio-economic status) had a better chance of survival.

![Feature Correlation Heatmap](images/heatmap.png) <!-- (Upload your plot first) -->
*Correlation heatmap of the main numerical features.*

## 🛠️ Tech Stack & Dependencies

- **Python 3.8+**
- **Pandas** & **NumPy** (Data manipulation)
- **Matplotlib** & **Seaborn** (Data visualization)
- **Scikit-Learn** (Machine Learning)

Install all required dependencies:
```bash
pip install -r requirements.txt
📁 Project Structure
text
titanic-survival-predictor/
├── models/                              # Serialized trained model
├── notebooks/                           # Jupyter notebook with full analysis
├── src/                                 # Source code for demo prediction
├── data/                                # Original dataset (optional)
├── requirements.txt                     # Project dependencies
└── README.md
🚀 How to Use
1. Explore the Analysis
Open and run the Jupyter Notebook (notebooks/titanic_survival_analysis.ipynb) to see the complete data science workflow.

2. Run a Prediction Demo
Use the src/app.py script to make a sample prediction on a hypothetical passenger.

bash
python app.py
Example Output:

text
Prediction: Survived
Probability of Survival: 93.81%
The function takes the following parameters:

pclass (int): Passenger class (1, 2, 3)

sex (int): 0 for Male, 1 for Female

age (float): Age of the passenger

sibsp (int): Number of siblings/spouses aboard

parch (int): Number of parents/children aboard

fare (float): Passenger fare

title (str): Passenger title ('Mr', 'Mrs', 'Miss', 'Master', 'Rare')

embarked (str): Port of embarkation ('C', 'Q', 'S')

📝 Methodology
Data Cleaning: Handled missing values in Age (median imputation) and Embarked (mode imputation).

Feature Engineering:

Created Title feature from the Name column and one-hot encoded it.

Created Has_Cabin binary feature from the Cabin column.

Dropped irrelevant features (PassengerId, Name, Ticket, Cabin).

Modeling: Trained a RandomForestClassifier optimized via GridSearchCV for hyperparameter tuning.

Evaluation: Used accuracy, precision, recall, F1-score, and a confusion matrix for robust evaluation.

👨‍💻 Author
Victory Daberechi

LinkedIn: https://www.linkedin.com/in/victory-daberechi-7783382a9?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app

GitHub: @DABERECHI-AI
