#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:16:38 2020

@author: marley
"""
from featurize import compute_features
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix



# features to use
# options are: iemg, mav, mmav, var, rms, zc, wamp, wl
feature_names = ['mav', 'var', 'rms']
channels = [1,2,3,4]
channel_names = ['channel {}'.format(i) for i in channels]
filename = 'windows-2020-02-16.pkl'
label_map = {'k': 3, ';':5, 'j': 2, 'l': 4, 'p': 5, 'u': 2, 'o':4, '.': 4,
          'm':2, 'n': 2, '[':5, ']': 5, "'": 5, 'h': 2, '/':5, '\\':5}
labels = set(label_map.values())
# reads file
df = pd.read_pickle(filename)
df['keypressed'] = df['keypressed'].map(label_map)
df['keypressed'].fillna(0, inplace=True)
print("Key press values: {}".format(df['keypressed'].unique()))

features = compute_features(df, channel_names, feature_names, mutate=True)



cols = all_names(channel_names, feature_names) + ['keypressed']

activation = features[features['keypressed'] > 0]
activation = activation[cols].to_numpy()
n_baseline_samples = int(activation.shape[0]/len(labels))
baseline = features[features['keypressed'] == 0 ].sample(n=n_baseline_samples, replace=False)
baseline = baseline[cols].to_numpy()

dataset = np.vstack((activation, baseline))

np.random.shuffle(dataset)
X = dataset[:,:-1]
Y = dataset[:,-1]
print("size of dataset:", X.shape)


validation_size = 0.20
seed = 7

# Test options and evaluation metric
scoring = 'accuracy'
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    
    
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size)

# classifier = svm.SVC(kernel='linear').fit(X_train, Y_train)
classifier = LinearDiscriminantAnalysis().fit(X_train, Y_train)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_validation, Y_validation,
                                 display_labels=[2,3,4,5,'base'],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)






