# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:36:06 2020

@author: miasya
"""
import numpy as np
import pandas as pd
import pickle
from real_time_class import Prediction

model_filename = 'model_features_2020-03-22_windows_date_all_subject_all_mode_1_2_4-03_26_2020_09_38_43.pkl'
features_filename = 'features_2020-03-22_windows_date_all_subject_all_mode_1_2_4.pkl'
destination_filename = 'features_with_prediction_2020-03-22_windows_date_all_subject_all_mode_1_2_4.pkl'

with open(features_filename, 'rb') as f:
    df = pickle.load(f)

Model = Prediction(model_filename=model_filename)
#%%

labels = []
for index, row in df.iterrows():

    x = row
    x = x.drop(['keypressed', 'hand', 'finger', 'id', 'mode',
                'channel 1', 'channel 2', 'channel 3', 'channel 4',
                'channel 5', 'channel 6', 'channel 7', 'channel 8'])
    
    labels.append(Model.clf.predict_proba(np.squeeze(x.to_numpy()).reshape(1,-1)))
    
    if index % 10000 == 0:
        print('10000 down, more to go!')

df['prediction'] = labels

#%%

df.to_pickle(destination_filename)