from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle

# load the pre-trained model
with open('realestate.pkl', 'rb') as f:
    model = pickle.load(f)

# create a Flask app
app = Flask(__name__)
CORS(app)

# define a route for the API


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])
    df['feature1'] = df['feature1'].astype(float)
    df['feature2'] = df['feature2'].astype(float)
    df['feature3'] = df['feature3'].astype(float)
    df['feature4'] = df['feature4'].astype(float)
    df['feature5'] = df['feature5'].astype(float)
    # make the prediction using the pre-trained model
    prediction = model.predict(df)[0]

    # return the prediction as a JSON response
    response = {'prediction': prediction}
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
