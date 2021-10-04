# -*- coding: utf-8 -*-

# Classification problem
# Whether the customer leaves the company or stays in the company 

# Part-1 Data Preprocessing

#importing relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,-1].values

# Catagorical variable encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
x[:,1] = labelencoder_X_1.fit_transform(x[:,1])
labelencoder_X_2 = LabelEncoder()
x[:,2] = labelencoder_X_2.fit_transform(x[:,2])

ct_1 = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder = 'passthrough')
x = ct_1.fit_transform(x)
x = x[:, 1:]



#Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=0) 


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)#the object must be fitted into the training set class before tansforming
x_test = sc_x.transform(x_test) #For the test set we dont need to fit the StandardScaler object as it is already fitted. we just transfom directly

# Part-2 Making the ANN
# Importing the Keras libraries and packages
import keras
# Importing appropriate modules
from keras.models import Sequential #initialize our NN
from keras.layers import Dense #build the layers of our ANN
from keras.layers import Dropout#overwrite and disable some neurons to prevent over fitting

# Initializing the ANN -- with seccessive layers
classifier = Sequential()

# Adding the input layer and the first hidden layer with dropout (the input layer is added by passing the input_dim parameter)
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim= 11))
classifier.add(Dropout(rate = 0.1))#try not to go over 0.5

# Adding a second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(rate = 0.1))#try not to go over 0.5

# Adding the output layer (because we are interested in the probability)
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN (Applying Stochastic Gradient Descent('adam'-> a type of SGD algorithm(very efficient)) on the ANN layers)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train, batch_size = 5, epochs = 100)

# Part-3 Making the predictions and executing the model

# Predicting the Test set results
y_pred = classifier.predict(x_test)
# Setting a threshold for our predicted probabilities
y_pred = (y_pred > 0.5)
# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Part-4 Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim= 11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 5, epochs = 100, use_multiprocessing=True)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout regularization to reduce overfitting if needed


#Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim= 11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, use_multiprocessing=True)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500, 700],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv=10)
grid_search = grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

