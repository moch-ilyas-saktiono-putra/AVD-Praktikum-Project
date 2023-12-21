# app.py

import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model_rf = joblib.load('random_forest_model.sav')

# Function to predict cluster based on user input
def predict_cluster(user_input):
    new_data = pd.DataFrame({
        'Paling sering membeli di tenant berapa?': [user_input['paling_sering_membeli']],
        'Rata-rata pengeluaran ketika beli di kantin?': [user_input['rata_pengeluaran']],
        'Rating': [user_input['rating']],
        'Antri': [user_input['antri']],
    })

    predicted_label = model_rf.predict(new_data)
    return predicted_label[0]

# Streamlit app
st.title('Prediksi Cluster Kantin Kampus')

# User input
user_input = {
    'paling_sering_membeli': st.radio('Paling sering membeli di tenant berapa?', [1, 2, 3, 4, 5]),
    'rata_pengeluaran': st.radio('Rata-rata pengeluaran ketika beli di kantin?', [1, 2, 3, 4, 5]),
    'rating': st.radio('Rating?', [1, 2, 3, 4, 5]),
    'antri': st.radio('Antri?', [0, 1]),
}

# Prediction and display
if st.button('Prediksi'):
    predicted_cluster = predict_cluster(user_input)
    st.success(f'Hasil Prediksi: Cluster {predicted_cluster}')
