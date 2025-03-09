from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and preprocessor
with open('../models/churn_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('../models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    df_preprocessed = preprocessor.transform(df)
    prediction = model.predict(df_preprocessed)[0]
    return jsonify({'churn_prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
