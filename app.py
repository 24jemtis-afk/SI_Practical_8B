import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺")


# Load saved files only once
@st.cache_resource
def load_files():
    model = XGBClassifier()
    model.load_model("diabetes_model.json")   # best model = XGBoost
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    model_name = joblib.load("model_name.pkl")
    accuracy = joblib.load("accuracy.pkl")
    return model, scaler, feature_columns, model_name, accuracy


model, scaler, feature_columns, model_name, accuracy = load_files()

# Sidebar
st.sidebar.title("About")
st.sidebar.write("Type 2 Diabetes Risk Prediction")
st.sidebar.write("Best Model:", model_name)
st.sidebar.write("Accuracy:", f"{accuracy * 100:.2f}%")
st.sidebar.write("Total Features:", len(feature_columns))

# Title
st.title("🩺 Type 2 Diabetes Risk Prediction")

# Model accuracy at the top
m1, m2 = st.columns(2)
m1.metric("Best Model", model_name)
m2.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

st.write("Enter the patient details below and click Predict.")

# Inputs in two columns
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    age = st.number_input("Age", 1, 120, 30)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

with col2:
    smoking = st.selectbox(
        "Smoking History",
        ["No Info", "current", "ever", "former", "never", "not current"]
    )
    bmi = st.number_input("BMI", 10.0, 100.0, 25.0)
    hba1c = st.number_input("HbA1c Level", 3.0, 10.0, 5.5)
    glucose = st.number_input("Blood Glucose Level", 50, 400, 120)

if st.button("Predict"):
    # Start with all columns as 0
    data = dict.fromkeys(feature_columns, 0)

    data["age"] = age
    data["hypertension"] = 1 if hypertension == "Yes" else 0
    data["heart_disease"] = 1 if heart_disease == "Yes" else 0
    data["bmi"] = bmi
    data["HbA1c_level"] = hba1c
    data["blood_glucose_level"] = glucose

    # One-hot columns (dropped category stays 0)
    if "gender_" + gender in data:
        data["gender_" + gender] = 1
    if "smoking_history_" + smoking in data:
        data["smoking_history_" + smoking] = 1

    input_df = pd.DataFrame([data])[feature_columns]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Result
    st.subheader("Result")
    if prediction == 1:
        st.error("High Risk: Diabetes Detected")
    else:
        st.success("Low Risk: No Diabetes")

    st.write(f"Diabetes Risk Probability: {probability * 100:.2f}%")
    st.progress(float(probability))

    # Entered details
    st.subheader("Entered Details")
    details = pd.DataFrame({
        "Feature": ["Gender", "Age", "Hypertension", "Heart Disease",
                    "Smoking History", "BMI", "HbA1c Level", "Blood Glucose"],
        "Value": [gender, age, hypertension, heart_disease,
                  smoking, bmi, hba1c, glucose]
    })
    st.dataframe(details, hide_index=True)

st.caption("This app is made only for educational purpose.")