import csv
import pickle

import pandas as pd

import numpy as np

data = pd.read_csv("test.csv", sep=",")
values = {}
for name in list(data):
    values.update({name: data[name].mean()})
data = data.fillna(value=values)
x_validation = np.array(data.drop(["index"], 1))

print(x_validation)
print(data.head())

pickle_in = open("tempPrediction.pickle", "rb")
regression = pickle.load(pickle_in)

predictions = regression.predict(x_validation)

print(predictions)
names = data.columns
mylist=[i for i in range(len(x_validation))]



arr1 = np.array(mylist).reshape(len(predictions),1)

arr2 = predictions.reshape(len(predictions),1)
predictionstest = np.concatenate((arr2,arr1),axis=1)

# t = pd.DataFrame(data= predictionstest, index = mylist,columns=["critical_temp","index"])
#
# print(t.head())
# t.to_csv("../code-ml-2021-challenge-1/out.csv")
print("test" ,predictionstest[3])

with open('out.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(["critical_temp", "index"])

    # write the data
    for i in range(len(predictions)):
         writer.writerow([predictions[i],i])

