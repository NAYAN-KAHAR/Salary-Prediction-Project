# file: app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd


with open("prediction_model.pkl", "rb") as f1:
    model = pickle.load(f1)

with open("accuary_model.pkl", "rb") as f2:
    model_accuracy = pickle.load(f2)



st.markdown("<h1 style='text-align: center;'>Salary Prediction App</h1>", unsafe_allow_html=True)

# defines cols
col1, col2, col3 = st.columns(3)

# input fields
with col1:
    age = st.number_input("Enter Employee Age", min_value=18, max_value=60, value=22)
    experience = st.number_input("Enter Employee Experience", min_value=0, max_value=60, value=0)


with col2:
    education = st.selectbox( label="Select Your Education",
                             options=['High School','Bachelor','Master','PhD'], index=1)
    job_title = st.selectbox( label="Select Your Job Title",
                             options=['Director','Analyst','Manager','Engineer'], index=0)


with col3:
    location = st.selectbox( label="Select Your Location",
                             options=['Suburban','Rural','Urban'], index=0)
    gender = st.selectbox( label="Select Your Gender",
                             options=['Male','Female'], index=0)

#button
if st.button("Predict Placement"):
    
    if experience > (age - 18): 
        st.error("Experience cannot exceed the number of working years since age 18.")

    else:
        input_data = pd.DataFrame([{
            'Education': education,
            'Experience': experience,
            'Location': location,
            'Job_Title': job_title,
            'Age': age,
            'Gender': gender
        }])

        predicted_salary = model.predict(input_data)[0]

        st.success(f'Estimated Salary: ${predicted_salary:.2f}')
        st.write(f'Model Accuracy : {model_accuracy * 100:.2f} %')


st.divider()