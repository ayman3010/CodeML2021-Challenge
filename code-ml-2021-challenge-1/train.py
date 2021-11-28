import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing as pr
import pickle

data = pd.read_csv("train.csv", sep=",")

#data = data[names]
print(data.head())
values = {}
for name in list(data):
    values.update({name : data[name].median()})

data = data.fillna(value=values)

# print(data.head())
# scaler = pr.MinMaxScaler()
# names = data.columns
# d = scaler.fit_transform(data)
# scaled_df = pd.DataFrame(d, columns=names)
#
# data = scaled_df

# names = np.random.choice(names, 5, replace=False)
# names = np.append(names,'critical_temp')


names = ["wtd_range_atomic_mass","wtd_std_ThermalConductivity", "wtd_std_ElectronAffinity", "wtd_entropy_ThermalConductivity", "mean_Density", "range_atomic_radius","wtd_mean_Valence","wtd_gmean_Valence","std_Density","wtd_gmean_ThermalConductivity","range_ThermalConductivity",]

predict = "critical_temp"
x = np.array(data[names])
y = np.array(data[predict])


best=0
for i in range(10):
    print("experience:" , i)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    regr_1 = RandomForestRegressor(max_depth=22, n_estimators=119, min_impurity_decrease=0.0007)
    regr_1.fit(x_train, y_train)
    print("Mean squared error: %.2f" % np.mean((regr_1 .predict(x_test) - y_test) ** 2))
    acc = regr_1.score(x_test, y_test)
    print("R2 :", acc)
    print("best : " , best)
    if acc>best:
        best=acc
        if acc > best:
            with open("tempPrediction.pickle", "wb") as f:
                pickle.dump(regr_1, f)  # sauvegarde le modele

# print(regr_1.predict(x_test))

















