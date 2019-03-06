#Import modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

#load data
data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(data.head())

#missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))

#dealing with missing data
data = data.drop((missing_data[missing_data['Total'] > 1]).index,1)
data = data.drop(data.loc[data['Electrical'].isnull()].index)
print(data.isnull().sum().max()) #just checking that there's no missing data missing...

#missing data for test
total = test.isnull().sum().sort_values(ascending=False)
percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))

#dealing with missing data
test = test.drop((missing_data[missing_data['Total'] > 1]).index,1)
test = test.drop(data.loc[data['Electrical'].isnull()].index)
print(data.isnull().sum().max())

data.drop(['MSZoning', 'Utilities', 'BsmtFullBath', 'BsmtHalfBath', 'Functional' ], axis=1, inplace=True)
#test.drop(['KitchenQual', 'Functional'], axis=1, inplace=True)
print(data.head())
print(test.head())

#print(data.columns.values, test.columns.values)
data = data.values
test = test.values

#splitting into test-train sets
X = data[:, :-1]
y = data[:, -1]

#Label Encoding

indices = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20, 21, 22, 23, 28, 29, 30, 31, 40, 45, 55, 56]
print(test[0])
for i in indices:
    labelencoder1 = LabelEncoder()
    X[:, i] = labelencoder1.fit_transform(X[:, i])
    test[:, i] = labelencoder1.transform(test[:, i])
#print(X[0])
print(test[0])

#One Hot Encoding and removing dummy trap
onehotencoder = OneHotEncoder(categorical_features=indices)
X = onehotencoder.fit_transform(X).toarray()
#test = onehotencoder.transform(test).toarray()
X = X[:, 1:]
#test = test[:, 1:]
#print(X[0])


#standardscaler = StandardScaler()
#X = standardscaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X_train, y_train)


print(regressor.score(X_test, y_test))
print(regressor.score(X_train, y_train))

#Fitting the model with whole data


regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X, y)
y_pred = regressor.predict(test)

'''
output = pd.DataFrame()
output['Id'] = test['Id']
print(output)
'''