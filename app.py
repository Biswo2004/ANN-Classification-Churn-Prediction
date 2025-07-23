import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load model and artifacts
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit UI
st.title("Customer Churn Prediction")
st.write("This app predicts customer churn based on various features.")
st.sidebar.header("Input Features")

# Inputs
geography = st.sidebar.selectbox("Geography", ['France', 'Spain', 'Germany'])
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
age = st.sidebar.slider("Age", 18, 100, 30)
balance = st.number_input("Balance", 0.0, 100000.0, 5000.0)
credit_score = st.number_input("Credit Score", 300, 850, 600)
estimated_salary = st.number_input("Estimated Salary", 0.0, 1000000.0, 50000.0)
tenure = st.slider("Tenure (Years)", 0, 10, 2)
number_of_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card", ['Yes', 'No'])
is_active_member = st.selectbox("Is Active Member", ['Yes', 'No'])

# Prepare input
gender_encoded = label_encoder_gender.transform([gender])[0]
has_cr_card_bin = 1 if has_cr_card == 'Yes' else 0
is_active_member_bin = 1 if is_active_member == 'Yes' else 0

base_input = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [has_cr_card_bin],
    'IsActiveMember': [is_active_member_bin],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform(base_input[['Geography']])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Final input with ordered columns
base_input.drop('Geography', axis=1, inplace=True)
final_input = pd.concat([base_input, geo_encoded_df], axis=1)

# Align column order to match scaler expectation
expected_order = scaler.feature_names_in_
final_input = final_input[expected_order]

# Scale and predict
final_input_scaled = scaler.transform(final_input)
prediction = model.predict(final_input_scaled)
probability = prediction[0][0]

# Result
if probability > 0.5:
    st.write(f"**Prediction:** Customer is likely to churn (Probability: {probability:.2f})")
else:
    st.write(f"**Prediction:** Customer is likely to stay (Probability: {probability:.2f})")


# Sidebar buttons

if st.sidebar.button("Clear Inputs"):
    st.cache_data.clear()
    st.rerun()

if st.sidebar.button("Make New Prediction"):
    st.rerun()

# Footer
st.markdown("---")
st.write("Developed by Biswojit Bal.")
