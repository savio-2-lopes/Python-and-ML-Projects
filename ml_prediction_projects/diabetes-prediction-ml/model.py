# importar bibliotecas
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Carregando o dataset
df = pd.read_csv('diabete.csv')
sec_df=df.copy(deep=True)
sec_df[[
  'Glicose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'
]] = sec_df[[
  'Glicose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'
]]

sec_df['Glicose'].fillna(sec_df['Glicose'].mean(), inplace=True)
sec_df['BloodPressure'].fillna(sec_df['BloodPressure'].mean(), inplace=True)
sec_df['SkinThickness'].fillna(sec_df['SkinThickness'].median(), inplace=True)
sec_df['Insulin'].fillna(sec_df['Insulin'].median(), inplace=True)
sec_df['BMI'].fillna(sec_df['BMI'].median(), inplace=True)

# Construção do modelo
X = df.drop(columns='Outcome')
y = df['Outcome']

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.20, random_state=0)

# Criando Modelo Random Forest
classifier=RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Criação de um arquivo pickle para o classificador
pickle.dump(classifier, open('model/model.pkl', 'wb'))