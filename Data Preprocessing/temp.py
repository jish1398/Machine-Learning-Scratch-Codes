##import numpy as np
##import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Data.csv') #File_Name for Convience;
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values

##df=pd.DataFrame(Y) if any object error comes by

from sklearn.preprocessing import Imputer
## It is used to replace all the missing values in the dataframe
## Imputer returns an intialised object of itself or self
imputer= Imputer(missing_values="NaN",strategy="mean",axis=0)
## fit method also returns self  which is used to fit the data into the imputer 
imputer= imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
## transform method is used to change all the occurence of Nan in the dataset.

## Categorical data are those which contains certain categories unlike numerical data
## Here country and decision are two 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Intialising the object
labelEncoder=LabelEncoder()
#The first row of the dataset is transformed
X[:, 0]=labelEncoder.fit_transform(X[:,0])
#Since this categorise data in terms of some precedence or value which is not required
#So buiding new col. depending upon no of categories in the col,
#Passing the col. no
onehotencoder=OneHotEncoder(categorical_features=[0])
# fitting and transforming together
X=onehotencoder.fit_transform(X).toarray()
labelencoderY=LabelEncoder()
Y=labelencoderY.fit_transform(Y)

# Now from the given datasets it necessary to split them into two different sets
# train set is used to develop the logic and test to test that logic
# it is necessary  to make sure that the model doesnt mug up the data without any development of logic

from sklearn.cross_validation import train_test_split
# arrays are the intput to the method and test_size is the percent value generally 20 or 30
# no need to put train size as both add up to 1 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Now if the input values and output values here be Age and Salary, the range differs a lot
# as a result it is necessary to scale them to comparable amount between 1 and -1
#It is done through two methods i.e standardisation and normalisation 
# this is what scaling is all about
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
# the train value is already fitted by default the test value will also be fitted.
X_test=sc_X.transform(X_test)

# Since here y is classification problem as it has 0 and 1 only no need of scaling
# But in general scaling is done for regression things.