import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression,Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_Id = train['Id']
test_Id = test['Id']
train.drop(['Id'], inplace=True, axis=1)
test.drop(['Id'], inplace=True, axis=1)

n_train = train.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)


all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print(missing_data.head(20))

all_data['PoolQC'] = all_data['PoolQC'].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

#Check remaining missing values if any
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print(missing_data.head())

#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


all_data = all_data.values
indices = [3, 5, 6, 7, 10, 11, 14, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 28, 30, 33, 34, 35, 36, 40, 41, 42, 44, 45, 46, 48, 50, 52,53,55,56,58,59,61,63, 65, 66, 67, 68, 69, 71, 77]
# process columns, apply LabelEncoder to categorical features
for i in indices:
    lbl = LabelEncoder()
    all_data[:, i] = lbl.fit_transform(all_data[:, i])

indices = [3, 5, 6, 7, 10, 11, 14, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 28, 30, 33, 34, 35, 36, 40, 41, 42, 44, 45, 46, 48, 50,53,55,56,59,63, 65, 66, 67, 68, 69, 71, 77]

onehotencoder = OneHotEncoder(categorical_features=indices)
all_data = onehotencoder.fit_transform(all_data).toarray()

all_data = all_data[:, 1:]

standardscaler = StandardScaler()
all_data = standardscaler.fit_transform(all_data)
train = all_data[:n_train]
test = all_data[n_train:]


regressor = RandomForestRegressor(n_estimators=500, criterion="mse")
regressor.fit(train, y_train)
y_pred = regressor.predict(test)
print(regressor.score(train, y_train))


sub = pd.DataFrame()
sub['Id'] = test_Id
sub['SalePrice'] = y_pred
sub.to_csv('submission.csv',index=False)

