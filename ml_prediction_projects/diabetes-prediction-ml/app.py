# importar bibliotecas
from flask import Flask, render_template, request
import pickle
import numpy as np

# Carregando o dataset
classifier = pickle.load(open('model/model.pkl', 'rb'))

app=Flask(__name__)

# Index
@app.route('/')
def home():
  return render_template('home.html')

# Realizar predicao
@app.route('/predicao', methods=['POST'])
def predicao():
  if request.method=='POST':
    pregnancies=request.form['pregnancies']
    glicose=request.form['glicose']
    bloodpressure=request.form['bloodpressure']
    skinthickness=request.form['skinthickness']
    insulin=request.form['insulin']
    bmi=request.form['bmi']
    diabetesfunction=request.form['diabetesfunction']
    age=request.form['age']

    data_array=np.array([[pregnancies, glicose, bloodpressure, skinthickness, insulin, bmi, diabetesfunction, age ]])

    print(data_array)

    prediction_text=classifier.predict(data_array)
    return render_template('prediction.html', prediction=prediction_text, insulin=insulin, pg=pregnancies, gl=glicose, bp=bloodpressure, st=skinthickness, bmi=bmi, db=diabetesfunction, age=age)

if __name__ == '__main__':
	app.run(debug=True)