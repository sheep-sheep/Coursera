import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
def part1_scatter():
#    %matplotlib notebook
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4);
    
    
# NOTE: Uncomment the function below to visualize the data, but be sure 
# to **re-comment it before submitting this assignment to the autograder**.   
#part1_scatter()

def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    res = []
    pred = np.linspace(0,10,100)[:, np.newaxis]
    X_train_data = X_train.reshape(-1,1)
    for degree in [1, 3, 6, 9]:
        poly = PolynomialFeatures(degree=degree)
        pred_poly = poly.fit_transform(pred)
        X_poly = poly.fit_transform(X_train_data)
        linreg = LinearRegression().fit(X_poly, y_train)
        pred_res = linreg.predict(pred_poly)
        res.append(pred_res)
    return np.array(res)

def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score
    r2_train, r2_test= [],[]
    X_train_data = X_train.reshape(-1,1)
    X_test_data = X_test.reshape(-1,1)
    for degree in range(0, 10):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_train_data)
        X_test_poly = poly.fit_transform(X_test_data)
        linreg = LinearRegression().fit(X_poly, y_train)
        r2_train.append(r2_score(y_train, linreg.predict(X_poly)))
        r2_test.append(r2_score(y_test, linreg.predict(X_test_poly)))
    return r2_train,r2_test

def answer_three():
    #import matplotlib.pyplot as plt    
    #res = answer_two()
    #x = [x for x in range(10)]
    #plt.plot(x, res[0], x, res[1])
    return 2, 8, 6

def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score
    X_train_data = X_train.reshape(-1,1)
    X_test_data = X_test.reshape(-1,1)
    
    poly = PolynomialFeatures(degree=12)
    X_poly = poly.fit_transform(X_train_data)
    X_test_poly = poly.fit_transform(X_test_data)
    linreg = LinearRegression().fit(X_poly, y_train)
    linlasso = Lasso(alpha=0.01, max_iter = 10000).fit(X_poly, y_train)
    
    return r2_score(y_test, linreg.predict(X_test_poly)), r2_score(y_test, linlasso.predict(X_test_poly))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv('C:\OOO\XXX\mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2


def answer_five():
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(random_state=0).fit(X_train2, y_train2)
    ind = np.argpartition(clf.feature_importances_, -5)[-5:]
    x = {}
    for i in range(5):
        x[X_train2.columns.values[ind][i]] = clf.feature_importances_[ind][i]
    import operator
    sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse = True)
    return [item[0] for item in sorted_x]

def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve
    param_range = np.logspace(-4,1,6)
    train_scores, test_scores = validation_curve(SVC(kernel = 'rbf', C=1, random_state=0), X_subset, y_subset,
                                            param_name='gamma',
                                            param_range=param_range, cv=3, scoring = 'accuracy')
    train_scores = [item.mean() for item in train_scores]
    test_scores = [item.mean() for item in test_scores]
    return np.array(train_scores), np.array(test_scores)
