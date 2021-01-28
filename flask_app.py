import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():

    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if prediction[0] == 0:
        output = 'No Pluvial Flood'
    elif prediction[0] == 1:
        output = 'Low Pluvial Flood'
    elif prediction[0] == 2:
        output = 'Moderate Pluvial Flood'
    elif prediction[0] == 3:
        output = 'High Pluvial Flood'
    else:
        output = 'Very High Pluvial Flood'


    return render_template('index.html', prediction_text=' {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)