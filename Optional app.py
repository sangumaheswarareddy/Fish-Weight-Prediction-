import streamlit as st
import joblib
import numpy as np

model = joblib.load("fish_weight_model.pkl")

st.title("Fish Weight Prediction App")

length1 = st.number_input("Length 1")
length2 = st.number_input("Length 2")
length3 = st.number_input("Length 3")
height = st.number_input("Height")
width = st.number_input("Width")

if st.button("Predict"):
    data = np.array([[length1, length2, length3, height, width]])
    prediction = model.predict(data)
    st.success(f"Predicted Weight: {prediction[0]:.2f} grams")
