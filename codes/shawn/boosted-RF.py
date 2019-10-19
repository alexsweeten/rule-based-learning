 # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from collections import Counter 
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt

training_data = pd.read_csv('C:/Users/Chung/rule-based-learning/datasets/humanSP1/humanSP1_train.csv', sep= ',', header=None)
test_data = pd.read_csv('C:/Users/Chung/rule-based-learning/datasets/humanSP1/humanSP1_test.csv', sep= ',', header=None)

X = training_data.values[:,0]
Y = training_data.values[:, 1:2]
X_real_test = test_data.values[:,0]

updated_X = []
for line in X:
    tmp= np.zeros((4, 14))
    for i in range(len(line)):
        if  line[i] == 'A':
            tmp[0][i] = 1
        elif line[i] == 'C':
            tmp[1][i] = 1
        elif line[i] == 'G':
            tmp[2][i] = 1
        elif line[i] == 'T':
            tmp[3][i] = 1
    tmp = tmp.flatten()
    updated_X.append(tmp)
#Y = Y=='binding site'

X_train, X_test, y_train, y_test = train_test_split(updated_X, Y.T[0], test_size = 0.3, random_state = 100)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

updated_X = []
for line in X_real_test:
    tmp= np.zeros((4, 14))
    for i in range(len(line)):
        if  line[i] == 'A':
            tmp[0][i] = 1
        elif line[i] == 'C':
            tmp[1][i] = 1
        elif line[i] == 'G':
            tmp[2][i] = 1
        elif line[i] == 'T':
            tmp[3][i] = 1
    tmp = tmp.flatten()
    updated_X.append(tmp)
X_real_test = updated_X
X_real_test = np.array(X_real_test)

tfbs_classifier = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 100)
tfbs_classifier.fit(X_train, y_train)
y_pred = tfbs_classifier.predict(X_test)
scores = f1_score(y_test, y_pred, average='macro')
ATCG_identity = ['A','C','G','T']
features = ["1"]*56
for i in range(len(X_train[0])):
    base = i//14
    features[i] = 'if position ' + str(i%14) +' is ' + ATCG_identity[base]

#save results
y_real_pred = tfbs_classifier.predict(X_real_test)
df = pd.DataFrame(y_real_pred)
df.to_csv('../result/boosted-RF.csv', header=0, index=0)

##plot importance
tfbs_classifier.get_booster().feature_names = features
xgb.plot_importance(tfbs_classifier)
plt.rcParams['figure.figsize'] = [10, 10]
plt.show()

for d in [1]:
    tfbs_classifier = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 100)
    scores = np.zeros(10)
    for i in range(10):
        tfbs_classifier.fit(X_train, y_train)
        y_pred = tfbs_classifier.predict(X_test)
        scores = f1_score(y_test, y_pred, average='macro')
    print('gini: ', np.mean(scores), '+-', np.std(scores))


