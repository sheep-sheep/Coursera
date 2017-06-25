import numpy as np
import pandas as pd
# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:\Users\Yang\Downloads\creditcard.csv')
df = df[:10000]
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


def answer_one():
    credit_df = pd.read_csv('C:\Users\Yang\Downloads\creditcard.csv')
    targetlist = list(credit_df['Class'])
    res = sum(targetlist)
    return float(res)/len(targetlist)


def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score
    # Nclass (0) is most frequent
    dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
    y_dummy_predictions = dummy_majority.predict(X_test)
    accuracy_score = dummy_majority.score(X_test, y_test)
    recall_score = recall_score(y_test, y_dummy_predictions)
    return accuracy_score, recall_score


def answer_three():
    from sklearn.metrics import recall_score, precision_score
    from sklearn.svm import SVC
    svm = SVC(kernel='rbf', C=1).fit(X_train, y_train)
    y_predict = svm.predict(X_test)
    return svm.score(X_test, y_test), recall_score(y_test, y_predict), precision_score(y_test, y_predict)
   

def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
    svm = SVC(kernel='rbf', C=1e9, gamma = 1e-07).fit(X_train, y_train)
    y_predict = svm.decision_function(X_test) > -220
    confusion = confusion_matrix(y_test, y_predict)
    return confusion

def answer_five():        
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    lr = LogisticRegression().fit(X_train, y_train)
    y_scores_lr = lr.fit(X_train, y_train).decision_function(X_test)

    precision, recall, thresholds = precision_recall_curve(y_test, y_scores_lr)
    closest_zero = np.argmin(np.abs(thresholds))
    closest_zero_r = recall[closest_zero]
    
    from sklearn.metrics import roc_curve, auc
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_scores_lr)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve (1-of-10 digits classifier)', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.axes().set_aspect('equal')
    plt.show()
    print fpr_lr
    print tpr_lr
    print roc_auc_lr
    return closest_zero_r, 1.0

def answer_six():    
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression().fit(X_train, y_train)
    grid_values = {'penalty': ['l1', 'l2'],
                   'C':[0.01, 0.1, 1, 10, 100]}
    grid_clf = GridSearchCV(clf, param_grid=grid_values, scoring='recall')
    grid_clf.fit(X_train, y_train)
    res = grid_clf.cv_results_['mean_test_score']
    return res.reshape(5,2)

# Use the following function to help visualize results from the grid search
def GridSearch_Heatmap(scores):
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
    plt.yticks(rotation=0);

print answer_five()
