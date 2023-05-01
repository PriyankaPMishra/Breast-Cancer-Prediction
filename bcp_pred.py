import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st

import tensorflow as tf #for creating neural networks
from tensorflow import keras #APIs of tensorflow
from keras.models import load_model


sc = StandardScaler() #instance of StandardScaler

def func(input_data):
    #loading model
    loaded_model = load_model('new_model.h5')

    #change the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshaping the np array as we predict for only 1 data point
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    sc.fit(input_data_reshaped)
    input_data_std = sc.transform(input_data_reshaped)
    
    input_data_reshaped_lstm = input_data_std.reshape(input_data_std.shape[0],1,input_data_std.shape[1])

    prediction = loaded_model.predict(input_data_reshaped_lstm) #probabilities and not 1/0 class

    prediction_label = [np.argmax(prediction)] #probs to 1/0

    if(prediction_label[0] == 0):
        return 'M'

    else:
        return 'B'


def app():
    st.title('Breast Cancer Prediction')
    
    # Get user input
    col1, col2, col3  = st.columns(3)
    with col1:
        radius_mean = st.number_input('Radius Mean')
    with col1: 
        texture_mean = st.number_input('Texture Mean')
    with col1:
        perimeter_mean = st.number_input('Perimter Mean')
    with col1:
        area_mean = st.number_input('Area Mean')
    with col1: 
        smoothness_mean = st.number_input('Smoothness Mean')
    with col1:
        compactness_mean = st.number_input("Compactness Mean")
    with col1:
        concavity_mean = st.number_input("Conacvity Mean")
    with col1:
        concave_pts_mean = st.number_input('Concave Points Mean')
    with col1:
        symmetry_mean = st.number_input('Symmetry Mean')
    with col1:
        fractal_dimension_mean = st.number_input('Fractal Dimension Mean')
    with col2:
        radius_se = st.number_input('Radius Squared Error')
    with col2: 
        texture_se = st.number_input('Texture Squared Error')
    with col2:
        perimeter_se = st.number_input('Perimter Squared Error')
    with col2:
        area_se = st.number_input('Area Squared Error')
    with col2: 
        smoothness_se = st.number_input('Smoothness Squared Error')
    with col2:
        compactness_se = st.number_input("Compactness Squared Error")
    with col2:
        concavity_se = st.number_input("Conacvity Squared Error")
    with col2:
        concave_pts_se = st.number_input('Concave Points Squared Error')
    with col2:
        symmetry_se = st.number_input('Symmetry Squared Error')
    with col2:
        fractal_dimension_se = st.number_input('Fractal Dimension Squared Error')        
    with col3:
        radius_worst = st.number_input('Radius Worst Mean')
    with col3: 
        texture_worst = st.number_input('Texture Worst Mean')
    with col3:
        perimeter_worst = st.number_input('Perimeter Worst Mean')
    with col3:
        area_worst = st.number_input('Area Worst Mean')
    with col3: 
        smoothness_worst = st.number_input('Smoothness Worst Mean')
    with col3:
        compactness_worst = st.number_input("Compactness Worst Mean")
    with col3:
        concavity_worst = st.number_input("Conacvity Worst Mean")
    with col3:
        concave_pts_worst = st.number_input('Concave Points Worst Mean')
    with col3:
        symmetry_worst = st.number_input('Symmetry Worst Mean')
    with col3:
        fractal_dimension_worst = st.number_input('Fractal Dimension Worst Mean')

    # Create an empty list to store the inputs
    inputs = [radius_mean , texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_pts_mean, symmetry_mean, fractal_dimension_mean,
              radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_pts_se, symmetry_se, fractal_dimension_se,
              radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_pts_worst, symmetry_worst, fractal_dimension_worst]
    
    result = ''
    color = ''
    prediction = func(inputs)

    #output display logic
    if st.button("PREDICT"):
        
        if prediction == 'B':
            color = 'Green'
            result = 'The Tumor Is Benign.'
        else:
            color = 'Red'
            result = 'The Tumor Is Malignant.'

    st.markdown(f'<h3 style="color: {color};">{result}</h3>', unsafe_allow_html=True)

    st.text('Made By Priyanka and Chandan.')

# Run the app
if __name__ == '__main__':
    app()
