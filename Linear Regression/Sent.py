# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 01:15:21 2018

@author: jishn
"""

import pandas as pd
import quandl as qd
import math
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
# This command is used to get the required datasets from the quandl website
df= qd.get('WIKI/GOOGL')
#Since the dataset contain a lot of values therefore selecting whats actually needed from the rest
df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['Hl_PCT']=(df['Adj. High']-df['Adj. Close']/df['Adj. Close'])*100
df['PCT_change']=(df['Adj. Close']-df['Adj. Open']/df['Adj. Open'])*100
df=df[['Adj. Close','Hl_PCT','PCT_change','Adj. Volume']]
forecast_col='Adj. Close'
df.fillna(-9999,inplace=True)
forecast_out=int(math.ceil(0.01*len(df)))
df['Label']=df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
X=np.array(df.drop(['Label'],1))
Y=np.array(df['Label'])
X=preprocessing.scale(X)
Y=np.array(df['Label'])
x_train,x_text,y_train,y_test=cross_validation.train_test_split(X,Y,test_size=0.2)
Reg=LinearRegression()
Reg.fit(x_train,y_train)
Acc=Reg.score(x_text,y_test)
print(Acc)
