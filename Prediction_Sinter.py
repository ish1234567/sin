# -*- coding: utf-8 -*-
import pandas as pd
featuresAI = pd.read_csv('testAI.csv')
print(featuresAI.head(5))
print(featuresAI.shape)
print(round(featuresAI.describe(),3))
import numpy as np
labelsAI = np.array(featuresAI['AI'])
featuresAI= featuresAI.drop('AI',axis=1)
featureAI_list = list(featuresAI.columns)
featuresAI= np.array(featuresAI)
from sklearn.model_selection import train_test_split
train_featuresAI, test_featuresAI, train_labelsAI, test_labelsAI = train_test_split(featuresAI, labelsAI, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_featuresAI.shape)
print('Training Labels Shape:', train_labelsAI.shape)
print('Testing Features Shape:', test_featuresAI.shape)
print('Testing Labels Shape:', test_labelsAI.shape)
from sklearn.ensemble import RandomForestRegressor
rfAI = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rfAI.fit(train_featuresAI, train_labelsAI);
predictionsAI= rfAI.predict(test_featuresAI)
errorsAI = abs(predictionsAI - test_labelsAI)
print('Mean Absolute Error:', round(np.mean(errorsAI), 2), 'degrees.')
mapeAI = 100 * (errorsAI / test_labelsAI)
accuracyAI = 100 - np.mean(mapeAI)
print('Accuracy Of AI prediction:', round(accuracyAI, 2), '%.')
import pandas as pa
features = pa.read_csv('test.csv')
print(features.head(5))
print(features.shape)
print(round(features.describe(),3))
import numpy as nu
labels= nu.array(features['TI'])
features= features.drop('TI',axis=1)
feature_list = list(features.columns)
features= nu.array(features)
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_features, train_labels);
predictions= rf.predict(test_features)
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
import streamlit as st
st.title("Sinter AI TI Prediction Model")
st.markdown("                           An Initiative by PI QA Team")
st.write("Enter the predicted sinter chemistry")
Tfe = st.number_input("Enter T Fe in sinter ")
CaO = st.number_input("Enter CaO in sinter ")
MgO  = st.number_input("Enter MgO in sinter ")
SiO2  = st.number_input("Enter SiO2 in sinter ")
Al2O3  = st.number_input("Enter Al2O3 in sinter ")
MnO = st.number_input("Enter MnO in sinter ")
P = st.number_input("Enter P in sinter ")
Cr2O3 = st.number_input("Enter Cr2O3 in sinter ")
newinput=[[Tfe,CaO,MgO,SiO2,Al2O3,MnO,P,Cr2O3]]
Predicted_TI=rf.predict(newinput)
Predicted_AI=rfAI.predict(newinput)
st.write("Predicted AI according to the input chemistry is: ", np.round(Predicted_AI,2))
st.write("Predicted TI according to the input chemistry is: ", np.round(Predicted_TI,2))
  