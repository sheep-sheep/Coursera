# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

# Q1
cancerdf = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
cancerdf['target'] = pd.Series(data = cancer.target)

# Q2
numTotal = len(cancerdf)
numBenign = sum(cancerdf['target'])
numMalignant = numTotal-numBenign

data_target = pd.Series(data=[numMalignant, numBenign], index=cancer.target_names)

# Q3
X= cancerdf[cancer.feature_names]
y = cancerdf['target']

# Q4
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Q5
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)

# Q6
means = cancerdf.mean()[:-1].values.reshape(1, -1)
knn.predict(means)

# Q7
knn.predict(X_test)

# Q8
knn.score(X_test, y_test)
