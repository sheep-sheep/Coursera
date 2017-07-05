Reusability matters

1. Separate the data into distinct groups by similarity
2. Trees often require less preprocessing of data
   Trees are easy to interpret and visualize
3. To improve generalization by reducing correlation among the trees and making the model more robust to bias.
4. Neural Networks wrong
   Naive Bayes wrong
5. For a model that won’t overfit a training set, Naive Bayes would be a better choice than a decision tree.
   For predicting future sales of a clothing line, Linear regression would be a better choice than a decision tree regressor
6. Neural Network
   KNN (k=1)
   Decision Tree
7. 
8. compliance_detail - More information on why each ticket was marked compliant or non-compliant
   collection_status - Flag for payments in collections
9. Remove variables that a model in production wouldn’t have access to wrong
   Sanity check the model with an unseen validation set wrong
   
If time is a factor, remove any data related to the event of interest that doesn’t take place prior to the event.
Ensure that data is preprocessed outside of any cross validation folds.
10. 0110


import numpy as np
import pandas as pd


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
cols = [ 'violation_street_name', 'violation_code','total_fine']
# preprocess the data
def preprocess(df):
    cols = ['violation_street_name', 'violation_code', 'total_fine']
    if 'compliance' not in df.columns:
        cols = cols+['ticket_id']
    df.loc[:, 'city']= df['city'].str.upper()
    df = df[df.city.isin(['DETROIT', 'DET'])] if 'compliance' in df.columns else df
    df.loc[:,'total_fine'] = df['fine_amount']+df['admin_fee']+df['state_fee']
    df.loc[:,'ticket_issued_date'] =pd.to_datetime(df['ticket_issued_date']).dt.date
    df.loc[:,'hearing_date'] =pd.to_datetime(df['hearing_date']).dt.date
    df = df[df.compliance.notnull()] if 'compliance' in df.columns else df
    df.loc[:, ( 'violation_street_name','violation_code')]= df.loc[:, ( 'violation_street_name', 'violation_code')].apply(LabelEncoder().fit_transform)
    cols = cols+['compliance'] if 'compliance' in df.columns else cols
    return df[cols]

df = pd.read_csv('C:/Users/Yang/Downloads/train.csv' )
train_data = preprocess(df)

df = pd.read_csv('C:/Users/Yang/Downloads/test.csv' )
test_data = preprocess(df)
target_data = test_data.iloc[:,:-1]
target_index = test_data.iloc[:,-1]

X = train_data.iloc[:,:-1]
Y = train_data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
clf = SVC(kernel='rbf', probability=True)
grid_values = {'gamma': [0.001],
               'C': [1, 10, 100]}
grid_clf_auc = GridSearchCV(clf, param_grid = grid_values, scoring = 'roc_auc')
grid_clf_auc.fit(X_train, y_train)
res = grid_clf_auc.decision_function(X_test) 
#    knn = KNeighborsClassifier(n_neighbors = 5)
#    clf = DecisionTreeClassifier(max_depth=5)
#    clf.fit(X_train, y_train)
#
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, res))
res = pd.Series([data[0] for data in grid_clf_auc.predict_proba(target_data)], index = target_index)




#grid_values = {'gamma': [0.01, 1.0, 10.0],
#               'C':[0.01, 0.1, 1, 10, 100]}
#grid_clf = GridSearchCV(clf, param_grid=grid_values)
#grid_clf.fit(X_train, y_train)
#res = grid_clf.predict(test_data)
