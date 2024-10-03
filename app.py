import pandas as pd
import streamlit as st
import joblib  # For loading the model
import numpy as np

# Load the trained model and scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app
st.title("Heart Disease Prediction App")

# Input fields for the user to enter their data
age = st.number_input("Age", min_value=1, max_value=120, value=55)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
trtbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=300, value=130)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=250)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results (0-2)", options=[0, 1, 2])
thalachh = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exng = st.selectbox("Exercise Induced Angina", options=[0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", value=2.5)
slp = st.selectbox("Slope of the Peak Exercise ST Segment (0-2)", options=[0, 1, 2])
caa = st.selectbox("Number of Major Vessels (0-3) Colored by Fluoroscopy", options=[0, 1, 2, 3])
thall = st.selectbox("Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)", options=[1, 2, 3])

# Create a DataFrame for the input values
input_values = {
    'age': age,
    'sex': sex,
    'cp': cp,
    'trtbps': trtbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalachh': thalachh,
    'exng': exng,
    'oldpeak': oldpeak,
    'slp': slp,
    'caa': caa,
    'thall': thall
}

# Create a DataFrame for the input values
new_data = pd.DataFrame([input_values])

# Scale the new data
new_data_scaled = scaler.transform(new_data)

# Make predictions
if st.button("Predict"):
    predictions = model.predict(new_data_scaled)
    prediction_result = "Heart Disease" if predictions[0] == 1 else "No Heart Disease"
    st.success(f"Prediction for the input values: {prediction_result}")

# Run the Streamlit app
if __name__ == "__main__":
    st.write("Model Loaded Successfully. You can input your data above.")
