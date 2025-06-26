#%%
import numpy as np
import sklearn.linear_model as lm
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.proportion as smprop
from sklearn import model_selection
from dtuimldmtools import rlr_validate, dbplotf, train_neural_net, visualize_decision_boundary
import matplotlib.pyplot as plt
import torch
from collections import Counter

# Path to file 
file = 'HR_data.csv'

# Load data
D2 = pd.read_csv(file)
df = D2.values
cols= range(1,7)


X = df[:,cols]
X = (X - np.ones((X.shape[0],1))* np.mean(X,0))
y = df[:, 11]
# Define Cross-validation
CV = model_selection.GroupKFold(n_splits = 5)
groups = df[:,9]
counter = Counter(df[:,11])
categories = list(counter.keys())
N, M = X.shape
C = len(categories)

missclass2 = [] 
missclasstr2 = []
# Define the model structure
n_hidden_units = 15  # number of hidden units in the signle hidden layer
model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units),  # M features to H hiden units
        torch.nn.GELU(),  # 1st transfer function
        # N hidden units to C classes
        torch.nn.Linear(n_hidden_units, C),  # C logits
        # we use the softmax-funtion along the "class" dimension
        torch.nn.Softmax(dim=1),  # final tranfer function, normalisation of logit output
    )
    # Since we're training a multiclass problem, we cannot use binary cross entropy,
    # but instead use the general cross entropy loss:
loss_fn = torch.nn.CrossEntropyLoss()


for i, (train_index, test_index) in enumerate(CV.split(X, y, groups)):
    X_train = X[train_index, :]
    y_train = y[train_index]
    X_test = X[test_index, :]
    y_test = y[test_index]
    
    # Train the network:
    
    # Making sure the arrays all contain the correct data type 
    X_train = X_train.astype(np.float32) 
    y_train = y_train.astype(np.int64)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.int64)
    #Modelfitting
    net, _, _ = train_neural_net(
        model,
        loss_fn,
        X = torch.tensor(X_train, dtype=torch.float),
        y = torch.tensor(y_train, dtype=torch.long),
        n_replicates=3,
        max_iter=10000,
    )

    #to calculate generalization error
    softmax_logits = net(torch.tensor(X_train, dtype=torch.float))    
    y_tr_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy()
    train_msc = sum(y_tr_est != y_train)/ len(y_tr_est)
    missclasstr2.append(train_msc)

    # Determine probability of each class using trained network
    softmax_logits = net(torch.tensor(X_test, dtype=torch.float))
    # Get the estimated class as the class with highest probability (argmax on softmax_logits)
    y_test_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy()

    # Determine errors
    test_len = 0
    missclasscount = 0 
    e = y_test_est != y_test
    for bool in e: 
        test_len += 1
        if bool == True:
            missclasscount += 1
    mcrate = missclasscount/test_len
    missclass2.append(mcrate)
    
    


#%%

print(missclass2, missclasstr2)



