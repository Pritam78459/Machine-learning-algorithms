import numpy as np
from math import sqrt
from collections import Counter
import warnings
import pandas as pd
import random
#importing the required modules.

dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}     #making a dataset for testing.
new_features = [5,7]                                            #adding a new feature.

def k_nearest_neighbors(data, predict ,k = 3):
    #this function gets the votes for the nearest neighbors.

    if len(data) >= k:
        warnings.warn('K is set to a value less than total votnigs groups')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance,group])
            
    votes = [i[1] for i in sorted(distances) [:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
            
    return vote_result

result = k_nearest_neighbors(dataset,new_features,k = 3)        #gets the total votes.

df = pd.read_csv('breast-cancer-wisconsin.data')                #getting the data set for implementation.
df.replace('?',-99999,inplace = True)                           #cleaning the data set.
df.drop(['id'],1,inplace = True)                                #dropping the id column.

full_data = df.astype(float).values.tolist()                    #storing the full data in a variable.
random.shuffle(full_data)                                       #shuffling the data.

#creating the training and testing data
test_size = 0.2
train_set = {2:[],4:[]}
test_set = {2:[],4:[]}
train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]

#filling up the training and testing data.
for i in train_data:
    train_set[i[-1]].append(i[:-1])
    
for i in test_data:
    test_set[i[-1]].append(i[:-1])
    
correct = 0     #flag for correct values.
total = 0       #flag for total values

#getting votes.
for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data , k = 5)
        if group == vote:
            correct += 1
        total += 1

print('Accuracy: ',correct / total)     #printing accuracy.