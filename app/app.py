from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os
import json
import logging
import traceback
import uuid
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
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
    """Home page with project information"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard with visualizations and analytics"""
    return render_template('dashboard.html')

@app.route('/predict-form')
def predict_form():
    """Form for making predictions via the web interface"""
    return render_template('predict.html')

@app.route('/api-docs')
def api_docs():
    """API documentation page"""
    return render_template('api_docs.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict customer churn based on input features"""
    try:
        if model is None or preprocessor is None:
            return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided. Please send customer data as JSON.'}), 400
            
        logger.info(f"Received prediction request with data: {data}")
        
        # Check if all required fields are present
        required_fields = [
            'tenure', 'MonthlyCharges', 'TotalCharges', 
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
            'PhoneService', 'InternetService', 'Contract', 
            'PaperlessBilling', 'PaymentMethod'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'required_fields': required_fields
            }), 400
        
        # Add customerID if it's missing (it's used during training but not for prediction)
        if 'customerID' not in data:
            data['customerID'] = f'TEST-{str(uuid.uuid4())[:8]}'
            
        # Add other missing non-required fields with default values if they were in the training data
        optional_fields = {
            'MultipleLines': 'No',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No', 
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No'
        }
        
        for field, default_value in optional_fields.items():
            if field not in data:
                data[field] = default_value
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess the data
        try:
            df_preprocessed = preprocessor.transform(df)
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': 'Error preprocessing data. Please check that your input data matches the expected format.',
                'details': str(e)
            }), 400
        
        # Make prediction
        try:
            prediction = model.predict(df_preprocessed)[0]
            
            # Get prediction probability
            probabilities = model.predict_proba(df_preprocessed)[0]
            
            # For string labels, get the probability corresponding to positive class
            if isinstance(prediction, str):
                if prediction == 'Yes':
                    probability = probabilities[list(model.classes_).index('Yes')]
                else:
                    probability = probabilities[list(model.classes_).index('No')]
                    # Invert probability for "No" predictions to show likelihood of churning
                    probability = 1 - probability
            else:
                # For numeric labels, get the highest probability
                probability = max(probabilities)
            
            # Create response
            result = {
                'churn_prediction': prediction,
                'churn_probability': float(probability),
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"Prediction result: {result}")
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': 'Error making prediction.',
                'details': str(e)
            }), 500
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify({'error': error_msg}), 500

@app.route('/health')
def health_check():
    """Health check endpoint to verify the application is working correctly"""
    status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    logger.info(f"Health check: {status}")
    return jsonify(status)

@app.route('/sample', methods=['GET'])
def sample():
    """Return a sample input for testing the API"""
    sample_data = {
        "customerID": "SAMPLE-001",
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
    return jsonify(sample_data)

if __name__ == '__main__':
    # Log that we're starting the app
    logger.info("Starting Flask app on 0.0.0.0:5000 (debug=True)")
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
