from celery import Celery
from celery.schedules import crontab
import os
from datetime import datetime
import logging
from .ml.automated_training import trigger_automated_training
from .ml.deep_learning import create_and_train_deep_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery
celery = Celery(
    'churn_prediction',
    broker=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
)

# Celery Configuration
celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True
)

# Schedule periodic tasks
celery.conf.beat_schedule = {
    'retrain-model-weekly': {
        'task': 'app.celery_worker.scheduled_model_retraining',
        'schedule': crontab(day_of_week='monday', hour=0, minute=0),
    },
    'deep-learning-training-monthly': {
        'task': 'app.celery_worker.scheduled_deep_learning_training',
        'schedule': crontab(0, 0, day_of_month='1'),
    }
}

@celery.task
def scheduled_model_retraining():
    """Scheduled task for model retraining"""
    try:
        data_path = os.path.join('data', 'processed', 'churn_data.csv')
        model_path = os.path.join('models', 'trained')
        
        logger.info("Starting scheduled model retraining")
        metrics = trigger_automated_training(data_path, model_path)
        logger.info(f"Model retraining completed. Metrics: {metrics}")
        
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
    except Exception as e:
        logger.error(f"Error in scheduled model retraining: {str(e)}")
        return {
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }

@celery.task
def scheduled_deep_learning_training():
    """Scheduled task for deep learning model training"""
    try:
        data_path = os.path.join('data', 'processed', 'churn_data.csv')
        model_path = os.path.join('models', 'deep_learning', 
                                 f'model_{datetime.now().strftime("%Y%m%d")}.h5')
        
        logger.info("Starting scheduled deep learning model training")
        model, history = create_and_train_deep_model(data_path, model_path)
        
        metrics = {
            'final_accuracy': history.history['accuracy'][-1],
            'final_loss': history.history['loss'][-1]
        }
        
        logger.info(f"Deep learning model training completed. Metrics: {metrics}")
        
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
    except Exception as e:
        logger.error(f"Error in scheduled deep learning training: {str(e)}")
        return {
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }

@celery.task
def process_batch_predictions(data):
    """Process batch predictions asynchronously"""
    try:
        # Implementation for batch prediction processing
        # This would load the model and process multiple records
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'predictions': []  # Add actual predictions here
        }
    except Exception as e:
        logger.error(f"Error in batch predictions: {str(e)}")
        return {
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        } 