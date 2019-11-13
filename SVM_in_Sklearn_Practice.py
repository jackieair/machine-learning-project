# import packages necessary
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

# Read the data
data = np.asarray(pd.read_csv('data.csv', header = None)) #if the file exists

# store the features to X, the labels to y
X = data[:,:2]
y = data[:, 2]

# create the svc model and Find the right parameters for this model
model = SVC(kernel= 'rbf', gamma = 5) # trial

# fit the model
model.fit(X, y)

# predict the results
y_pred = model.predict(X)

# calculate the accuracy

acc = accuracy_score(y, y_pred)


