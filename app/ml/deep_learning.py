import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow.tensorflow
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepLearningModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self._build_model()
        self.scaler = StandardScaler()
        
    def _build_model(self):
        """Build the neural network architecture"""
        model = models.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
        
    def train(self, X, y, validation_split=0.2, epochs=50, batch_size=32):
        """Train the deep learning model"""
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Start MLflow run
            with mlflow.start_run():
                # Log model parameters
                mlflow.log_params({
                    'input_dim': self.input_dim,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'validation_split': validation_split
                })
                
                # Train model
                history = self.model.fit(
                    X_scaled, y,
                    validation_split=validation_split,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=5,
                            restore_best_weights=True
                        )
                    ]
                )
                
                # Log metrics
                metrics = {
                    'final_loss': history.history['loss'][-1],
                    'final_accuracy': history.history['accuracy'][-1],
                    'final_auc': history.history['auc'][-1],
                    'val_loss': history.history['val_loss'][-1],
                    'val_accuracy': history.history['val_accuracy'][-1],
                    'val_auc': history.history['val_auc'][-1]
                }
                mlflow.log_metrics(metrics)
                
                # Log model
                mlflow.tensorflow.log_model(self.model, "model")
                
                logger.info(f"Model training completed. Metrics: {metrics}")
                return history
                
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
            
    def predict(self, X):
        """Make predictions using the trained model"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def evaluate(self, X, y):
        """Evaluate the model on test data"""
        X_scaled = self.scaler.transform(X)
        return self.model.evaluate(X_scaled, y)
        
def create_and_train_deep_model(data_path, save_path):
    """Create and train a deep learning model"""
    try:
        # Load data
        df = pd.read_csv(data_path)
        X = pd.get_dummies(df.drop('Churn', axis=1))
        y = df['Churn']
        
        # Create and train model
        model = DeepLearningModel(input_dim=X.shape[1])
        history = model.train(X, y)
        
        # Save model
        model.model.save(save_path)
        
        return model, history
        
    except Exception as e:
        logger.error(f"Error in creating and training deep model: {str(e)}")
        raise 