import streamlit as st
import numpy as np
import joblib

scaler = joblib.load("Scaler.plk")
model = joblib.load("best_model.pkl")

st.title("Churn Prediction App")
st.divider()
st.write("Fill the form to predict the churn rate")
st.divider()

age = st.number_input("Enter Age :",min_value=10, max_value=100,value=30)
tenure = st.number_input("Enter Tenure :",min_value=0, max_value=130,value=10)
monthlyCharge = st.number_input("Enter Monthly charge :",min_value=30, max_value=150)
gender = st.selectbox("Enter the gender", ["Male","Female"])

st.divider()

predictButton= st.button("Predict")

if predictButton:
    gender_sel = 1 if gender == "Female" else 0 
    
    X= [age,gender_sel,tenure,monthlyCharge]
    
    X1 = np.array(X)
    
    X_array = scaler.transform([X1])
    
    prediction = model.predict(X_array)[0]
    
    predicted = "Yes" if prediction == 1 else "No"
    
    st.write(f"Prediction :{predicted}")
else:
    st.write("enter value and press predict")    
