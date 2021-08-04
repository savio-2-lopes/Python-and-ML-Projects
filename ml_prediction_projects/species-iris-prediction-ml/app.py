from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model/model.pkl', 'rb'))

# Index
@app.route('/')
def Home():
    return render_template('home.html')

# Predição
@app.route('/predicao', methods=['POST'])
def predicao():
    comprimento_sepala=request.form['comprimento_sepala']
    tamanho_sepala=request.form['tamanho_sepala']
    comprimento_petala=request.form['comprimento_petala']
    tamanho_petala=request.form['tamanho_petala']
    
    float_features=[float(x) for x in request.form.values()]
    features=[np.array(float_features)]
    prediction=model.predict(features)

    print(prediction)

    return render_template('prediction.html', comprimento_sepala=comprimento_sepala, tamanho_sepala=tamanho_sepala, comprimento_petala=comprimento_petala, tamanho_petala=tamanho_petala, iris=prediction, item_predicao='Irís é da especie {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)