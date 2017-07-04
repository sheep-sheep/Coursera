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
from sklearn.model_selection import train_test_split

# preprocess the data
def preprocess(df):
    df.loc[:, 'city']= df['city'].str.upper()
    df = df[df.city.isin(['DETROIT', 'DET'])]
    df.loc[:,'total_fine'] = df['fine_amount']+df['admin_fee']+df['state_fee']
    df.loc[:,'ticket_issued_date'] =pd.to_datetime(df['ticket_issued_date']).dt.date
    df.loc[:,'hearing_date'] =pd.to_datetime(df['hearing_date']).dt.date
    df = df[df.compliance.notnull()] if 'compliance' in df.columns else df
    cols = ['violator_name', 'violation_street_name', 'city', 'violation_code', 'total_fine']
    cols = cols+['compliance'] if 'compliance' in df.columns else cols
    return df[cols]

## pick name with:
#df = pd.read_csv('C:/Users/Yang/Downloads/train.csv' )
#train_data = preprocess(df)
#
#df = pd.read_csv('C:/Users/Yang/Downloads/test.csv' )
#test_data = preprocess(df)
