
import joblib
import pandas as pd


model = joblib.load('titanic_survival_predictor.pkl')

# Get the feature names the model was trained on
expected_features = model.feature_names_in_
print(f"Model expects these features: {list(expected_features)}")

def predict_survival(pclass, sex, age, sibsp, parch, fare, title, embarked):
    """
    Predict survival on the Titanic.
    sex: 0 for male, 1 for female
    pclass: 1, 2, or 3
    title: 'Mr', 'Mrs', 'Miss', 'Master', 'Rare'
    embarked: 'C', 'Q', 'S'
    """
    
    
    input_data = pd.DataFrame(0, index=[0], columns=expected_features)
    
    
    input_data['Pclass'] = pclass
    input_data['Sex'] = sex
    input_data['Age'] = age
    input_data['SibSp'] = sibsp
    input_data['Parch'] = parch
        input_data['Fare'] = fare
    
    
    title_col = f"Title_{title}"
    if title_col in input_data.columns:
        input_data[title_col] = 1
        
 
    embarked_col = f"Embarked_{embarked}"
    if embarked_col in input_data.columns:
        input_data[embarked_col] = 1


    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    survival_status = "Survived" if prediction == 1 else "Did Not Survive"
    survival_prob = probability[1] * 100 

    print(f"\nPrediction: {survival_status}")
    print(f"Probability of Survival: {survival_prob:.2f}%")
    return prediction


predict_survival(pclass=1, sex=1, age=30, sibsp=0, parch=0, fare=100, title='Mrs', embarked='S')