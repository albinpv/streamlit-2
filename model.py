import pandas as pd
import numpy as np

data=pd.read_csv('Iris.csv')
print(data.head())
x=data[['SepalLengthCm','SepalWidthCm','PetalLengthCm', 'PetalWidthCm']]
y=data[['Species']]
print(x.columns)
print(y.columns)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
x_train=min_max.fit_transform(x_train)
x_test=min_max.fit_transform(x_test)

from sklearn.tree import DecisionTreeClassifier
dc= DecisionTreeClassifier()
dc.fit(x_train,y_train)
y=dc.predict(x_test)
from sklearn.metrics import accuracy_score
print("Accuracy on Decision Tree Model: ", accuracy_score(y_test, y))


import joblib
joblib.dump(dc,'sample_model')