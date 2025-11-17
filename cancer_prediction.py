import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Cancer Prediction Using KNN", layout="wide")
st.title("🩺 Cancer Prediction Using KNN")
st.write("Enter patient details to predict if cancer is **Benign (0)** or **Malignant (1)**.")

# Load model, scaler & features
model = joblib.load("knn_cancer_model.pkl")
scaler = joblib.load("knn_scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.header("🔍 Patient Attributes")

inputs = []
cols = st.columns(2)  # 2 columns for input fields

for i, feature in enumerate(feature_names):
    col = cols[i % 2]
    # Set realistic default ranges
    if feature in ["age", "bmi", "physical_activity", "alcohol_intake"]:
        value = col.number_input(feature, min_value=0.0, max_value=100.0, value=0.0)
    elif feature in ["gender", "smoking", "genetic_risk", "cancer_history"]:
        value = col.number_input(feature, min_value=0, max_value=5, value=0)
    else:
        value = col.number_input(feature, value=0.0)
    inputs.append(value)

if st.button("Predict"):
    input_array = np.array(inputs).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ The patient is predicted to have **Malignant Cancer**")
    else:
        st.success("✅ The patient is predicted to have **Benign/No Cancer**")