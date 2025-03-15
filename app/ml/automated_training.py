import mlflow
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectFromModel
import shap
from celery import shared_task
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutomatedModelTraining:
    def __init__(self, data_path, model_save_path):
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.best_model = None
        self.feature_selector = None
        self.scaler = StandardScaler()
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        try:
            df = pd.read_csv(self.data_path)
            X = df.drop('Churn', axis=1)
            y = df['Churn']
            
            # Handle categorical variables
            X = pd.get_dummies(X)
            
            return X, y
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def automated_feature_selection(self, X, y):
        """Perform automated feature selection"""
        base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = SelectFromModel(base_model, prefit=False)
        selector.fit(X, y)
        
        # Get selected feature mask and names
        selected_features = X.columns[selector.get_support()].tolist()
        logger.info(f"Selected features: {selected_features}")
        
        self.feature_selector = selector
        return X.loc[:, selected_features]
        
    def objective(self, trial, X_train, X_test, y_train, y_test):
        """Optuna objective function for hyperparameter optimization"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
        
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return f1_score(y_test, y_pred)
        
    def train_model(self):
        """Train the model with automated feature selection and hyperparameter tuning"""
        try:
            # Start MLflow run
            with mlflow.start_run() as run:
                # Load and preprocess data
                X, y = self.load_and_preprocess_data()
                
                # Perform feature selection
                X_selected = self.automated_feature_selection(X, y)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_selected, y, test_size=0.2, random_state=42
                )
                
                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Hyperparameter optimization
                study = optuna.create_study(direction='maximize')
                study.optimize(lambda trial: self.objective(
                    trial, X_train_scaled, X_test_scaled, y_train, y_test
                ), n_trials=50)
                
                # Train final model with best parameters
                best_params = study.best_params
                self.best_model = RandomForestClassifier(**best_params, random_state=42)
                self.best_model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                y_pred = self.best_model.predict(X_test_scaled)
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred)
                }
                
                # Log metrics and parameters
                mlflow.log_params(best_params)
                mlflow.log_metrics(metrics)
                
                # Generate SHAP values
                explainer = shap.TreeExplainer(self.best_model)
                shap_values = explainer.shap_values(X_test_scaled)
                
                # Save model and artifacts
                self.save_model()
                
                logger.info(f"Model training completed. Metrics: {metrics}")
                return metrics
                
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
            
    def save_model(self):
        """Save the trained model and related artifacts"""
        if self.best_model is None:
            raise ValueError("No model to save. Please train the model first.")
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.model_save_path, timestamp)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model and artifacts
        mlflow.sklearn.save_model(self.best_model, model_dir)
        
@shared_task
def trigger_automated_training(data_path, model_save_path):
    """Celery task for automated model training"""
    trainer = AutomatedModelTraining(data_path, model_save_path)
    return trainer.train_model() 