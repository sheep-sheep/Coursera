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
# preprocess the data
def preprocess(df):
    cols = ['ticket_id', 'total_fine', 'violation_code', 'zip_code', 'city']
    df.loc[:, 'city']= df['city'].str.upper()
#    df = df[df.city.isin(['DETROIT', 'DET'])] if 'compliance' in df.columns else df
    df.loc[:,'total_fine'] = df['fine_amount']+df['admin_fee']+df['state_fee']
    df['ticket_issued_date'] =pd.to_datetime(df['ticket_issued_date']).dt.date
    df['hearing_date'] =pd.to_datetime(df['hearing_date']).dt.date
    df['time_delta'] =df['hearing_date']- df['ticket_issued_date']
    df = df[df.compliance.notnull()] if 'compliance' in df.columns else df
    df.loc[:, ('violation_code', 'zip_code', 'time_delta', 'city')]= df.loc[:, ('violation_code', 'zip_code', 'time_delta', 'city')].apply(LabelEncoder().fit_transform)
    cols = cols+['time_delta']
    cols = cols+['compliance'] if 'compliance' in df.columns else cols
    return df[cols]

df = pd.read_csv('C:/Users/Yang/Downloads/train.csv', encoding='latin1')
train_data = preprocess(df)
df = pd.read_csv('C:/Users/Yang/Downloads/test.csv', encoding='latin1')
test_data = preprocess(df)
address = pd.read_csv('C:/Users/Yang/Downloads/addresses.csv', encoding='latin1')
geo = pd.read_csv('C:/Users/Yang/Downloads/latlons.csv', encoding='latin1')
geo_data = pd.merge(address, geo, how='left', left_on=['address'], right_on = ['address']).loc[:,('ticket_id', 'lat', 'lon')]
train_data = pd.merge(train_data, geo_data, how='left', left_on=['ticket_id'], right_on = ['ticket_id']).iloc[:,1:]#Remove ticketID
test_data = pd.merge(test_data, geo_data, how='left', left_on=['ticket_id'], right_on = ['ticket_id'])
target_data = test_data.iloc[:,1:]
target_index = test_data.iloc[:,0]

train_data = train_data[train_data.lat.notnull()]
X = train_data[['total_fine', 'violation_code', 'zip_code', 'city', 'time_delta', 'lat', 'lon']]
Y = train_data.iloc[:,-3]# get compliance
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 1)
#
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#parameters = {'n_estimators':(5, 10, 15, 20), 
#              'max_depth':[3, 5, 10],}
#
#parameters = {'learning_rate':[0.1, 0.05, 0.01],
#              'n_estimators':[10, 20, 30, 40]}
#
#clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators = 80, max_depth=13, min_samples_leaf=2,subsample=0.8, max_features = 6, random_state=0).fit(X_train, y_train)
#grid_search = GridSearchCV(clf, param_grid=parameters, cv=3, scoring='roc_auc')
#
#grid_search.fit(X_train, y_train)
#res = grid_search.predict(X_test)
#knn = KNeighborsClassifier(n_neighbors = 5).fit(X_train, y_train)
#parameters = {'n_neighbors':(5, 10, 15, 20), }
#grid_search = GridSearchCV(knn, param_grid=parameters, cv=3, scoring='roc_auc')
#clf = SVC().fit(X_train, y_train)
clf = RandomForestClassifier().fit(X_train, y_train)
#parameters = {'penalty': ['l1','l2'],
#              'C':[0.01, 0.1, 1, 10, 100],
#              'random_state':[0, 1],}
parameters = {
              "min_samples_split": [2, 4, 6],
              "max_depth": [None, 5, 10],
              "min_samples_leaf": [2, 5, 10],
              "max_leaf_nodes": [None],
              }#"criterion": ["gini", "entropy"],
grid_search = GridSearchCV(clf, param_grid=parameters, cv=3, scoring='roc_auc')#scoring='roc_auc'cv=3,
grid_search.fit(X_train, y_train)
res = grid_search.predict(X_test)
#    clf = DecisionTreeClassifier(max_depth=5)
#    clf.fit(X_train, y_train)
#
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, res))
res = pd.Series([data[0] for data in grid_search.predict_proba(target_data)], index = target_index)



