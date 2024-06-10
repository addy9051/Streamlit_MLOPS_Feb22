import pandas as pd

import streamlit as st
import datetime
import pickle

#streamlit header
st.header('Cars24 Price Prediction App')

df = pd.read_csv("C:/Users/ankit/PycharmProjects/StreamlitTutorial/cars24-car-price.csv")

st.dataframe(df)


seller_type = st.selectbox(
    "Select Seller Type: ",
    ("Trustmark Dealer", "Individual", "Dealer"))


engine = st.slider("Set the Engine Power", 0, 6752)
km_driven = st.slider("What is the Kilometers driven?", 100, 3800000)

#year	seller_type	km_driven	fuel_type	transmission_type	mileage	engine	max_power	seats

col1, col2, col3 = st.columns(3)
with col1:
    fuel_type = st.selectbox(
        "Select Fuel Type: ",
        ("Petrol", "Diesel", "CNG", "LPG", "Electric"))
with col2:
    transmission_type = st.selectbox(
        "Select Transmission Type: ",
        ("Manual", "Automatic"))

with col3:
    seats = st.selectbox(
        "Select number of seats: ",
        (2, 4, 5, 6, 7, 8, 9, 10, 14))

col4, col5, col6 = st.columns(3)
with col4:
    year = st.slider(
        "Select Year: ", 2022, 2050, step=1)
with col5:
    mileage = st.slider(
        "Select Mileage: ", 0, 120, step=1)

with col6:
    max_power = st.slider(
        "Select Maximum Power: ", 5, 626, step=1)

encode_dict = {
    "fuel_type": {"Diesel": 1, "Petrol": 2, "CNG": 3, "LPG": 4, "Electric": 5},
    "seller_type": {"Trustmark Dealer": 1, "Individual": 2, "Dealer": 3},
    "transmission_type": {"Manual": 1, "Automatic": 2}
}


def model_pred(fuel_encoded, seller_encoded, transmission_encoded):
    with open("car_pred_model", "rb") as file:
        reg_model = pickle.load(file)

        input_features = [
            [year, seller_encoded, km_driven,	fuel_encoded, transmission_encoded, mileage, engine, max_power, seats]]

        return reg_model.predict(input_features)

if st.button("Predict"):
    fuel_encoded = encode_dict["fuel_type"][fuel_type]
    seller_encoded = encode_dict["seller_type"][seller_type]
    transmission_encoded = encode_dict["transmission_type"][transmission_type]


    price = model_pred(fuel_encoded, seller_encoded, transmission_encoded)
    st.text("Predicted Price is : "+ str(price[0].round(2)) + " Lakhs")