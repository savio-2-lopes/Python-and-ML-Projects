# importar bibliotecas
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics

# Carregando o dataset
df = pd.read_csv('car.csv')
df.shape

# Checando valores
df.isnull().sum()
df.describe()

sec_df=df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
print(sec_df.head())

sec_df['Current Year']=2021
print(sec_df.head())

sec_df['no_year']=sec_df['Current Year'] - sec_df['Year']
sec_df.drop(['Year'],axis=1,inplace=True)

sec_df=pd.get_dummies(sec_df,drop_first=True)
sec_df=sec_df.drop(['Current Year'],axis=1)
sec_df.corr()

X=sec_df.iloc[:,1:]
y=sec_df.iloc[:,0]

X['Owner'].unique()

model = ExtraTreesRegressor()
model.fit(X,y)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

regressor=RandomForestRegressor()

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid
random_grid = {
  'n_estimators': n_estimators,
  'max_features': max_features,
  'max_depth': max_depth,
  'min_samples_split': min_samples_split,
  'min_samples_leaf': min_samples_leaf
}

# Primeiro crie o modelo básico para ajustar
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)
rf_random.best_params_
rf_random.best_score_
predictions=rf_random.predict(X_test)
plt.scatter(y_test,predictions)

# Criação de um arquivo pickle para o classificador
pickle.dump(rf_random, open('model/model.pkl', 'wb'))