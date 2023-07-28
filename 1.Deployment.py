import pandas as pd
import numpy as np
import pickle

loaded_model = pickle.load(open('C:/CAUTION/Code/Machine learning/11.Projects/2.Diabetes/Deployment/trained_model.sav', 'rb'))

input = (5,166,72,19,175,25.8,0.587,51)
input = np.asarray(input).reshape(1,-1)
prediction = loaded_model.predict(input)

if prediction[0]==0:
    print('The person is non-diabetic.')
else:
    print('The person is diabetic.')