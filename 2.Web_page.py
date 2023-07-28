import numpy as np
import pickle
import streamlit as st
# Loading the saved model

loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# Creating a function for prediction

def diabetes_prediction(input):
    
    input = np.asarray(input).reshape(1,-1)
    
    prediction = loaded_model.predict(input)

    if prediction[0]==0:
        return 'The person is predicted to be non-diabetic.'
    else:
        return 'The person is predicted to be diabetic.'
    
def main():
   
    # Giving a title
    
    st.title('# Diabetes Prediction #')
    
    st.markdown(
        """
        <div style="font-size: 20px; color: red; text-align: center; margin-top: 10px;">
            ⚠️ CAUTION : This is a machine learning model and not a substitute for professional medical advice. 
            Please consult a healthcare professional for precise diagnosis and treatment. ⚠️
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Getting the input from the user
    
    Pregnancies = st.text_input('Number of pregnancies :')
    Glucose = st.text_input('Glucose level :')
    BloodPressure = st.text_input('Blood pressure value :')
    SkinThickness = st.text_input('Skin thickness value :')
    Insulin = st.text_input('Insulin level :')
    BMI = st.text_input('Body Mass Index value :')
    DiabetesPedigreefunction = st.text_input('Diabetes Pedigree Function value :')
    Age = st.text_input('Enter the Age :')
    
    
    # Code for prediction
    diagnosis = ''
    
    
    # Creating a button for preiction
    if st.button('Predict the result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreefunction, Age])
        
    
    st.success(diagnosis)
    
    
    
    
if __name__ == '__main__':
    main()
    
        
    