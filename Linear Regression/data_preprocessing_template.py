# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling is not needed as linear Regression library will take of care of it.
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Basic Idea is to teach the model using the training set and testing the model using the test set

from sklearn.linear_model import LinearRegression
Regressor=LinearRegression()
Regressor.fit(X_train,y_train)
Y_Pred=Regressor.predict(X_test)

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,Regressor.predict(X_train),color='blue')
plt.title("Salary vs EXP")
plt.xlabel("EXP")
plt.ylabel("Salary")
plt.show()
# the above graph shows the regression model for the trained data
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,Regressor.predict(X_test),color='blue')
plt.title("Salary vs EXP")
plt.xlabel("EXP")
plt.ylabel("Salary")
plt.show()
#the above graph shows the regression model for the testing data
# through visualisation one can check the accuracy of the model
# to view the accuracy of the model
print(Regressor.score(X_test,y_test)) 