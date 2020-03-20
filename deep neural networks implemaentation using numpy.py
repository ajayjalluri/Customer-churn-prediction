

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
#The one hot encoder does not accept 1-dimensional array or a pandas series, the input should always be 2 Dimensional.
#The data passed to the one hot encoder should not contain strings.
# Encoding categorical data ###geeks for geeks
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])

X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]# there should be correlation blw france coloumn,germany coloumn ,sprin coloumn 
#so we are removing any one coloumn
# if we are able to predict one feautre based on 2 other feauters
# then there is no use of that feautre
# Splitting the dataset into the Training set and Test set
#use make colomn transformer instead of all preprocessing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# fit transorm find mean and variance and stores in object and then applies
# transform collects and then applies 
# Part 2 - Now let's make the ANN!
X_train=X_train.T
y_train=y_train.reshape(1,8000)
from dnnfunctions import*
layer_dim=[11,6,6,1]
parameters = L_layer_model(X_train,y_train , layer_dim, num_iterations = 2500, print_cost = True)
p= predict(X_train, y_train, parameters)