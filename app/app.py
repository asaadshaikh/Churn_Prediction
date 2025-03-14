from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get the correct paths for model files
app_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(app_dir)
model_path = os.path.join(project_dir, 'models', 'churn_model.pkl')
preprocessor_path = os.path.join(project_dir, 'models', 'preprocessor.pkl')

# Load the trained model and preprocessor
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Model loaded successfully from {model_path}")
    
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    logger.info(f"Preprocessor loaded successfully from {preprocessor_path}")
except Exception as e:
    logger.error(f"Error loading model or preprocessor: {e}")
    logger.warning("API will not be able to make predictions without model files")
    model = None
    preprocessor = None

@app.route('/')
def home():
    # A simple home page
    return """
    <html>
        <head>
            <title>Churn Prediction API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
                .container { max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                pre { background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }
                .endpoint { margin-bottom: 20px; }
                .endpoint h2 { color: #0066cc; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Customer Churn Prediction API</h1>
                <p>Use this API to predict customer churn. Send customer data to the prediction endpoint to get a churn prediction.</p>
                
                <div class="endpoint">
                    <h2>Prediction Endpoint: POST /predict</h2>
                    <p>Send a POST request with JSON customer data to get a prediction.</p>
                    <p>Example request:</p>
                    <pre>
{
  "tenure": 24,
  "MonthlyCharges": 65.5,
  "TotalCharges": 1556.7,
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check"
}
                    </pre>
                </div>
                
                <div class="endpoint">
                    <h2>Health Check: GET /health</h2>
                    <p>Check if the API is working properly.</p>
                </div>
            </div>
        </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict customer churn based on input features"""
    try:
        if model is None or preprocessor is None:
            return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
            
        data = request.get_json()
        logger.info(f"Received prediction request with data: {data}")
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess the data
        df_preprocessed = preprocessor.transform(df)
        
        # Make prediction
        prediction = model.predict(df_preprocessed)[0]
        
        # Get prediction probability
        probabilities = model.predict_proba(df_preprocessed)[0]
        probability = max(probabilities)
        
        # Create response
        result = {
            'churn_prediction': int(prediction) if isinstance(prediction, (int, float)) else prediction,
            'probability': float(probability),
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"Prediction result: {result}")
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"Error making prediction: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    health_status = {
        'status': 'OK' if model is not None and preprocessor is not None else 'ERROR',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    }
    return jsonify(health_status)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    logger.info(f"Starting Flask app on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
