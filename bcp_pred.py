import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st

import tensorflow as tf #for creating neural networks
from tensorflow import keras #APIs of tensorflow
from keras.models import load_model

from streamlit_option_menu import option_menu
from PIL import Image

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

    # sidebar for navigation
    with st.sidebar:
        selected = option_menu('Breast Cancer Prediction',  
                            ['Home','About','Predict a Cancer'],
                            icons=['house-fill','bi-info-circle-fill','person'],
                            default_index=0)
    
    # for Breast Cancer Prediction System
    if selected == 'Predict a Cancer':
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

    # for Home
    if selected == 'Home':
        st.header('WELCOME TO BREAST CANCER DETECTION SYSTEM!')
        st.subheader('This is a StreamLit web application for Breast Cancer Detection Process.')
        st.markdown("""
        - CANCER – a 6 lettered word, which is common to the world nowadays, is a major cause of death among the population. Cancer is a disease that occurs when cells in the body grow and divide uncontrollably, leading to the formation of abnormal masses of tissue called tumors. 
        - These tumors can be either benign (non-cancerous) or malignant (cancerous) and can grow and spread to other parts of the body. Cancer has identiﬁed a diverse condition of several various subtypes. The timely screening and course of treatment of a cancer form is important.
        - One such forms of cancer is the Breast Cancer, which is a common mostly among women. Breast Cancer symptoms include changed genes, excruciating pain, size and shape, variations in the color (redness) of the breasts, and changes in the texture of the skin. 
        
        """)

    # for about 
    if selected == 'About':
        st.header('Breast Cancer Prediction')
        st.markdown("""
        - A Streamlit Web Application for Breast Cancer Prediction. 
        - DATASET: Wisconsin Breast Cancer Dataset (WBCD)
        - MODEL: Stacked LSTM (Long Short Term Memory) 
            - It is a type of Recurrent Neural Network (RNN) that is designed to address the limitations of traditional RNNs, which have difficulty retaining long-term dependencies in sequential data. 
            - LSTM achieves this by using a memory cell that can store information for an extended period, along with a set of gates that control the flow of information into and out of the cell. These gates allow LSTM to selectively forget or remember information from the past, and update the memory cell with new information as needed. 
        """)

    st.text('Made By Priyanka and Chandan.')

# Run the app
if __name__ == '__main__':
    app()
