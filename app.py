import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("titanic_survival_predictor.pkl")

# Expected features
expected_features = model.feature_names_in_

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details below to predict survival:")

# User inputs
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
sex = 0 if sex == "Male" else 1

age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=50.0)

title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Predict button
if st.button("Predict Survival"):
    # Prepare input
    input_data = pd.DataFrame(0, index=[0], columns=expected_features)
    input_data["Pclass"] = pclass
    input_data["Sex"] = sex
    input_data["Age"] = age
    input_data["SibSp"] = sibsp
    input_data["Parch"] = parch
    input_data["Fare"] = fare

    # One-hot encodings
    title_col = f"Title_{title}"
    if title_col in input_data.columns:
        input_data[title_col] = 1

    embarked_col = f"Embarked_{embarked}"
    if embarked_col in input_data.columns:
        input_data[embarked_col] = 1

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    survival_status = "🟢 Survived" if prediction == 1 else "🔴 Did Not Survive"
    survival_prob = probability[1] * 100

    st.subheader("Prediction Result")
    st.write(f"**Prediction:** {survival_status}")
    st.write(f"**Probability of Survival:** {survival_prob:.2f}%")
