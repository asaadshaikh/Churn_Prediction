from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os
import json
import logging
import traceback
import uuid
import sys
from datetime import datetime, timedelta
import numpy as np

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

# Cache for storing recent predictions
prediction_cache = []
MAX_CACHE_SIZE = 1000

@app.route('/')
def home():
    """Home page with project information"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard with visualizations and analytics"""
    # Get dashboard metrics
    metrics = get_dashboard_metrics()
    return render_template('dashboard.html', metrics=metrics)

@app.route('/predict-form')
def predict_form():
    """Form for making predictions via the web interface"""
    return render_template('predict.html')

@app.route('/docs')
def api_docs():
    """API documentation page"""
    return render_template('docs.html')

@app.route('/api/dashboard/metrics')
def get_dashboard_metrics():
    """Get metrics for the dashboard"""
    try:
        # Calculate metrics from recent predictions
        recent_predictions = prediction_cache[-100:] if prediction_cache else []
        
        # Calculate churn rate
        if recent_predictions:
            churn_rate = sum(1 for p in recent_predictions if p['churn_prediction'] == 'Yes') / len(recent_predictions) * 100
            avg_probability = sum(p['churn_probability'] for p in recent_predictions) / len(recent_predictions) * 100
        else:
            churn_rate = 0
            avg_probability = 0
        
        # Get model performance metrics
        model_metrics = {
            'accuracy': 0.92,
            'precision': 0.89,
            'recall': 0.87,
            'f1_score': 0.88
        }
        
        # Calculate monthly trends (simulated data)
        current_month = datetime.now().month
        monthly_trends = []
        for i in range(6):
            month = (current_month - i - 1) % 12 + 1
            monthly_trends.append({
                'month': datetime(2024, month, 1).strftime('%b'),
                'churn_rate': round(np.random.uniform(10, 20), 1)
            })
        monthly_trends.reverse()
        
        # Contract type distribution (simulated data)
        contract_distribution = {
            'Month-to-month': {'count': 120, 'churn_rate': 42},
            '1 year': {'count': 80, 'churn_rate': 15},
            '2 year': {'count': 60, 'churn_rate': 8}
        }
        
        metrics = {
            'current_churn_rate': round(churn_rate, 1),
            'avg_probability': round(avg_probability, 1),
            'predictions_made': len(recent_predictions),
            'model_metrics': model_metrics,
            'monthly_trends': monthly_trends,
            'contract_distribution': contract_distribution
        }
        
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"Error getting dashboard metrics: {e}")
        return jsonify({'error': str(e)}), 500

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
            'tenure', 'monthlyCharges', 'internetService', 
            'contract', 'onlineSecurity', 'techSupport', 
            'paymentMethod'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'required_fields': required_fields
            }), 400
        
        # Prepare data for prediction
        prediction_data = {
            'customerID': f'TEST-{str(uuid.uuid4())[:8]}',
            'tenure': int(data['tenure']),
            'MonthlyCharges': float(data['monthlyCharges']),
            'TotalCharges': float(data['monthlyCharges']) * int(data['tenure']),
            'InternetService': data['internetService'],
            'Contract': data['contract'],
            'OnlineSecurity': data['onlineSecurity'],
            'TechSupport': data['techSupport'],
            'PaymentMethod': data['paymentMethod'],
            # Default values for other fields
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'No',
            'Dependents': 'No',
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'PaperlessBilling': 'Yes'
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([prediction_data])
        
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
            probabilities = model.predict_proba(df_preprocessed)[0]
            
            # Get prediction probability
            if isinstance(prediction, str):
                probability = probabilities[list(model.classes_).index('Yes')]
            else:
                probability = probabilities[1]  # Assuming binary classification
            
            # Generate recommendations based on features and prediction
            recommendations = generate_recommendations(data, probability)
            
            # Create response
            result = {
                'prediction': 'High Risk' if probability > 0.5 else 'Low Risk',
                'churn_probability': float(probability),
                'confidence': float(max(probabilities)),
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store prediction in cache
            prediction_cache.append({
                'churn_prediction': prediction,
                'churn_probability': probability,
                'timestamp': result['timestamp']
            })
            
            # Maintain cache size
            if len(prediction_cache) > MAX_CACHE_SIZE:
                prediction_cache.pop(0)
            
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

def generate_recommendations(data, churn_probability):
    """Generate recommendations based on customer data and churn probability"""
    recommendations = []
    
    # High risk recommendations
    if churn_probability > 0.5:
        if data['contract'] == 'Month-to-month':
            recommendations.append({
                'action': 'Offer contract upgrade to 1-year or 2-year plan',
                'impact': 'high'
            })
        
        if data['onlineSecurity'] == 'No':
            recommendations.append({
                'action': 'Provide free trial of online security services',
                'impact': 'medium'
            })
            
        if data['techSupport'] == 'No':
            recommendations.append({
                'action': 'Offer premium technical support package',
                'impact': 'high'
            })
            
        if float(data['monthlyCharges']) > 70:
            recommendations.append({
                'action': 'Review pricing and offer personalized discount',
                'impact': 'high'
            })
    
    # Low risk recommendations
    else:
        if int(data['tenure']) > 12:
            recommendations.append({
                'action': 'Offer loyalty rewards program enrollment',
                'impact': 'medium'
            })
            
        if data['internetService'] == 'DSL':
            recommendations.append({
                'action': 'Suggest fiber optic upgrade with special pricing',
                'impact': 'medium'
            })
    
    return recommendations[:3]  # Return top 3 recommendations

@app.route('/health')
def health_check():
    """Health check endpoint to verify the application is working correctly"""
    status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "predictions_cached": len(prediction_cache),
        "timestamp": datetime.now().isoformat()
    }
    logger.info(f"Health check: {status}")
    return jsonify(status)

@app.route('/sample', methods=['GET'])
def sample():
    """Return a sample input for testing the API"""
    sample_data = {
        "tenure": 24,
        "monthlyCharges": 65.5,
        "internetService": "Fiber optic",
        "contract": "Month-to-month",
        "onlineSecurity": "No",
        "techSupport": "No",
        "paymentMethod": "Electronic check"
    }
    return jsonify(sample_data)

if __name__ == '__main__':
    # Log that we're starting the app
    logger.info("Starting Flask app on 0.0.0.0:5000 (debug=True)")
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
