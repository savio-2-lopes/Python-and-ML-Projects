# importar bibliotecas
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

import pickle
import numpy as np

# Carregando o dataset
classifier=pickle.load(open('model/model.pkl', 'rb'))

app=Flask(__name__)

# Index
@app.route('/')
def home():
  return render_template('home.html')

standard_to=StandardScaler

# Realizar predicao
@app.route('/predicao', methods=['POST'])
def predicao():
  
  fuel_type_diesel=0

  if request.method=='POST':
    year=int(request.form['year'])
    present_price=float(request.form['present_price'])
    kms_driven=int(request.form['kms_driven'])
    kms_driven2=np.log(kms_driven)
    owner=int(request.form['owner'])
    fuel_type_petrol=request.form['fuel_type_petrol']

    if fuel_type_petrol=='Petrol':
      fuel_type_petrol=1
      fuel_type_diesel=0

    else:
      fuel_type_petrol=0
      fuel_type_diesel=1
    
    year=2021-year
    seller_type_individual=request.form['seller_type_individual']

    if(seller_type_individual=='Individual'):
        seller_type_individual=1

    else:
        seller_type_individual=0	
    
    transmission_mannual=request.form['transmission_mannual']
    if(transmission_mannual=='Mannual'):
        transmission_mannual=1

    else:
        transmission_mannual=0
    
    prediction=classifier.predict([[present_price, kms_driven2, owner, year, fuel_type_diesel, fuel_type_petrol, seller_type_individual, transmission_mannual]])
    output=round(prediction[0],2)

    return render_template('prediction.html', prediction=output, present_price=present_price, kms_driven2=kms_driven2, owner=owner, fuel_type_diesel=fuel_type_diesel, seller_type_individual=seller_type_individual, transmission_mannual=transmission_mannual)
    
if __name__ == '__main__':
	app.run(debug=True)