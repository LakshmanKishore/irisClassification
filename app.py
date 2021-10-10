from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('iris.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        data=request.form.to_dict()
        print(data)
        prediction = model.predict([[float(data["sl"]),float(data["sw"]),float(data["pl"]),float(data["pw"])]])
        classes = ["SETOSA","VERSICOLOR","VIRGINICA"]
        answer = classes[int(prediction)]
        filename = answer+".png"
        return render_template('prediction.html',prediction = answer,filename=filename)

if __name__=="__main__":
    app.run(debug=True)