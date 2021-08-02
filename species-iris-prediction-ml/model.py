import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df=pd.read_json('flor.json')
print(df.head())

X=df[["comprimento_sepala", "tamanho_sepala", "comprimento_petala", "tamanho_petala"]]
y=df["Flor"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

classifier=RandomForestClassifier()
classifier.fit(X_train, y_train)

pickle.dump(classifier, open('model/model.pkl', 'wb'))