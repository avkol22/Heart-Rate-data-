
import numpy as np
import sklearn.linear_model as lm
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.proportion as smprop
from sklearn import model_selection
from dtuimldmtools import rlr_validate
import matplotlib.pyplot as plt


# Path to file 
file = 'HR_data.csv'


# Load data
D = pd.read_csv(file)
df = D.values
cols= range(1,7)


# Define model and cross-validation
CV = model_selection.GroupKFold(n_splits = 5)
model = lm.LogisticRegression()
groups = df[:,9]

# 
residuals = []
missclass = []
missclasstr = []

attributeNames = np.asarray(D.columns[cols]) 
X = df[ : , cols]
y = df[:, 11]
for i in range(168):
    if y[i] >= 5:
        y[i] = 1
    else:
        y[i] = 0
y = y.astype(int)

# Logistic regression 

#%%

for i, (train_index, test_index) in enumerate(CV.split(X, y, groups)):
    X_train = X[train_index, :]
    y_train = y[train_index]
    X_test = X[test_index, :]
    y_test = y[test_index]
    
    #CLassifier 
    model = model.fit(X_train, y_train)
    # train loss
    y_trainest = model.predict(X_train)
    miss_classtrain = np.sum(y_trainest != y_train) / float(len(y_trainest))
    missclasstr.append(miss_classtrain)
    # test 
    y_est = model.predict(X_test)
    misclass_rate = np.sum(y_est != y_test) / float(len(y_est))
    missclass.append(misclass_rate)
    residuals.append(y_est - y_test)


#%%

# Display classification results
avr_miss = sum(missclass)/ len(missclass)


print("\nOverall misclassification rate: {0:.3f}".format(avr_miss))



