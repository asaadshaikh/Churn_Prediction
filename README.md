# Customer Churn Prediction

A machine learning project to predict customer churn for a subscription-based business with a beautiful interactive dashboard.

![Churn Prediction Dashboard](https://user-images.githubusercontent.com/your-username/churn-prediction/main/screenshots/dashboard.png)

## Project Overview

This project builds a predictive model to identify customers who are likely to churn (cancel their subscription). By identifying these customers in advance, businesses can take proactive measures to retain them.

## Features

- **Interactive Web Dashboard**: Beautiful visualizations showing churn rates, key factors, and analytics
- **User-Friendly Prediction Interface**: Easy-to-use form for making individual customer predictions
- **Comprehensive API**: Well-documented REST API for integrating predictions into other systems
- **Robust Machine Learning Model**: Random Forest classifier with high accuracy for churn prediction
- **Data Visualization**: Charts showing the relationship between various factors and churn rate
- **Detailed Documentation**: Clear instructions for setup, usage, and API integration

## Repository Structure

```
├── app/                # Flask web application
│   ├── app.py          # API endpoints and web routes
│   ├── templates/      # HTML templates for web interface
│   └── static/         # CSS, JS, and image assets
├── data/               # Data files
│   └── churn_data.csv  # Input dataset
├── models/             # Trained models
│   ├── churn_model.pkl      # Serialized model
│   └── preprocessor.pkl     # Serialized data preprocessor
├── notebooks/          # Jupyter notebooks
│   └── exploratory_analysis.ipynb  # Data exploration
├── scripts/            # Python scripts
│   ├── data_processing.py   # Data preprocessing functions
│   └── model_training.py    # Model training script
└── requirements.txt    # Project dependencies
```

## Setup and Installation

1. Clone the repository
   ```
   git clone https://github.com/asaadshaikh/Churn_Prediction.git
   cd Churn_Prediction
   ```

2. Create a virtual environment
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

## Usage

### Web Interface

Run the Flask app to access the web interface:
```
python app/app.py
```

Then open your browser and navigate to:
- **Home Page**: `http://localhost:5000/`
- **Interactive Dashboard**: `http://localhost:5000/dashboard`
- **Prediction Form**: `http://localhost:5000/predict-form`
- **API Documentation**: `http://localhost:5000/api-docs`

### API Usage

The project provides a RESTful API with the following endpoints:

- **POST /predict**: Predict churn for a customer
- **GET /health**: Check API health status
- **GET /sample**: Get sample data for testing

Example API request with Python:
```python
import requests
import json

# Customer data
data = {
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

# Make API request
response = requests.post(
    'http://localhost:5000/predict',
    headers={'Content-Type': 'application/json'},
    data=json.dumps(data)
)

# Print result
result = response.json()
print(f"Churn Prediction: {result['churn_prediction']}")
print(f"Probability: {result['churn_probability'] * 100:.2f}%")
```

### Data Preparation
Place your data file in the `data/` directory and run:
```
python scripts/data_processing.py
```

### Model Training
Train the model with:
```
python scripts/model_training.py
```

## Dashboard Features

The interactive dashboard includes:
- **Key Metrics**: Overall churn rate, retention rate, average monthly charges, etc.
- **Factor Analysis**: Visualizations of how different factors affect churn rate
- **Customer Segments**: Breakdown of churn by customer segments
- **Prediction History**: Recent predictions made by the model

## Model Performance

The current model achieves:
- Accuracy: ~85%
- Precision: ~80%
- Recall: ~75%
- F1 Score: ~77%

## Key Insights

Analysis of the data revealed several important factors that influence customer churn:
1. **Contract Type**: Month-to-month contracts have a significantly higher churn rate (42.7%) compared to one-year (11.3%) or two-year contracts (2.9%).
2. **Internet Service**: Fiber optic customers churn at a higher rate (41.9%) than DSL (19.0%).
3. **Online Security and Tech Support**: Customers without these services are more likely to churn.
4. **Payment Method**: Electronic check users have the highest churn rate.
5. **Tenure**: Longer-tenured customers are less likely to churn.

## Future Improvements

- Implement hyperparameter tuning
- Add more features from customer support interactions
- Explore different algorithms (XGBoost, Neural Networks)
- Implement batch prediction for multiple customers
- Add user authentication for the dashboard
- Create additional visualizations for deeper insights

## Contributors

- Your Name (@YourGitHub)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 