import numpy as np
import streamlit as st
import pandas as pd

import seaborn as sns

import joblib

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

housing = pd.read_csv('USA_Housing.csv')
x = housing.drop(['Address','Price'], axis=1)
y = housing['Price']

x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=101)

st.title('US Housing Prediction Model')

lr_model = LinearRegression()  #instantiating algorithm
lr_model.fit(x_train,y_train)

from joblib import dump, load
dump(lr_model, 'lr_model.joblib') 
load_model = load('lr_model.joblib')
load_model.predict(x_train)

Avg_Area_Income = st.sidebar.number_input('Avg. Area ,Income', min_value=0.0, value=0.0)
Avg_Area_House_Age = st.sidebar.number_input('Avg. Area House Age', min_value=0.0, value=0.0)	
Avg_Area_Number_of_Rooms = st.sidebar.number_input('Avg. Area Number of Rooms', min_value=0.0, value=0.0)	
Avg_Area_Number_of_Bedrooms = st.sidebar.number_input('Avg. Area Number of Bedrooms', min_value=0.0, value=0.0)	 
Area_Population =  st.sidebar.number_input('Area Population', min_value=0.0, value=0.0)

input_data = np.array([[Avg_Area_Income,Avg_Area_House_Age,Avg_Area_Number_of_Rooms,Avg_Area_Number_of_Bedrooms,Area_Population ]])

if st.button("Prediction"):
    prediction = lr_model.predict(input_data)
    st.success(f"Prediction: {prediction[0]:.2f}")

