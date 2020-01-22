import sklearn
import quandl
import math,datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
#imported required 

style.use('ggplot')			#style used for plotting.

df = quandl.get("WSE/TSGAMES", authtoken="ruRydyUwqLVegTszKtxZ")		#note: you must create an account in the quandl site to access this data directly
df.head()																#or you can get the data in the data folder.

forecast_col = 'High'				#setting the cloumn that is going to be forecasted.
df.fillna(-99999,inplace = True)	#filling the full values.

forecast_out = int(math.ceil(0.1 * len(df)))			#getting the forecast out values.
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)		#making the label column.

#getting the X features.
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace = True)

#getting the y labels.
y = np.array(df['label'])

#shuffling and splitting the values.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 2)

classifier = LinearRegression()		#setting the linear regression algorithm in a classifier
classifier.fit(X_train,y_train)		#training features against labels.

#saving the classifiers in a file.
with open('linearregression.pickle','wb') as f:
    pickle.dump(classifier,f)
    
pickle_in = open('linearregression.pickle','rb')
classifier = pickle.load(pickle_in)

accuracy = classifier.score(X_test,y_test)		#getting the accuracy after testing.

forecast_set = classifier.predict(X_lately)		#getting the set of forecast values.
print(forecast_set,accuracy,forecast_out)

#getting dates for the graph.
df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    
#plotting the data in a graph.    
df['High'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()