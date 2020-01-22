import numpy as np
from sklearn import preprocessing ,neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
#importing the required modules for the k nearest algorithm.

df = pd.read_csv('breast-cancer-wisconsin.data')		#gettingthe data set.
df.replace('?', -99999, inplace = True)					#cleaning the data set.
df.drop(['id'],1,inplace = True)						#removing the id column.

X = np.array(df.drop(['class'],1))						#creating the features.
y = np.array(df['class'])								#creating the labels.

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)		#splitting the data for training and testing.

clf = neighbors.KNeighborsClassifier()					#creating the classifiers.
clf.fit(X_train, y_train)								#training the classifiers.

accuracy = clf.score(X_test, y_test)					#getting the accuracy.
print(accuracy)

#example measures.
example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])

example_measures = example_measures.reshape(len(example_measures),-1)

#predicting the example measures.
prediction = clf.predict(example_measures)
print(prediction)