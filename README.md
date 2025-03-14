# Customer Churn Prediction

A machine learning project to predict customer churn for a subscription-based business.

## Project Overview

This project builds a predictive model to identify customers who are likely to churn (cancel their subscription). By identifying these customers in advance, businesses can take proactive measures to retain them.

## Repository Structure

```
├── app/                # Flask web application
│   └── app.py          # API endpoints for predictions
├── data/               # Data files
│   └── churn_data.csv  # Input dataset (not included in repo)
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

### Making Predictions
Start the Flask API:
```
python app/app.py
```

Then send POST requests to `http://localhost:5000/predict` with customer data to get churn predictions.

## Model Performance

The current model achieves:
- Accuracy: ~85%
- Precision: ~80%
- Recall: ~75%
- F1 Score: ~77%

## Future Improvements

- Implement hyperparameter tuning
- Add more features
- Explore different algorithms (XGBoost, Neural Networks)
- Create a dashboard for visualizing predictions 