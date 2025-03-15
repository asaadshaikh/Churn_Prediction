# model_training.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import pickle
from data_processing import load_data, preprocess_data
from sklearn.utils import class_weight
import numpy as np

def train_model(X_train, y_train, class_weights=None):
    """
    Train a Random Forest model on the given data
    
    Args:
        X_train: Preprocessed feature data
        y_train: Target variable data
        class_weights: Dictionary of class weights for handling imbalanced data
        
    Returns:
        Trained model
    """
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight=class_weights
    )
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    print("Starting model_training.py...")
    
    # Get the correct path for the data file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_path = os.path.join(project_dir, 'data', 'churn_data.csv')
    
    # Load and preprocess the data
    df = load_data(data_path)
    if df is None:
        print("Error: Could not load data. Please make sure the churn_data.csv file exists in the data directory.")
        exit(1)

    print("Data loaded.")
    columns = df.columns.tolist()
    print("Columns in DataFrame:", columns)

    # Replace "Churn" with the EXACT column name from the printed list.
    target_column = "Churn"  # <--- VERY IMPORTANT: Replace with the correct column name

    if target_column not in columns:
        print(f"Error: Target column '{target_column}' not found in DataFrame.")
        print("Available columns:", columns)
        print("Please check the column name and update target_column.")
        exit(1)

    X = df.drop(target_column, axis=1)
    y = df[target_column]
    print("Features and target separated.")
    X_preprocessed, preprocessor = preprocess_data(X)
    print("Data preprocessed.")
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42, stratify=y)
    print("Data split into train and test (stratified).")

    # Check class distribution
    print("\nClass distribution in training set:")
    y_train_counts = pd.Series(y_train).value_counts(normalize=True)
    print(y_train_counts)
    
    # Calculate class weights to handle imbalance
    unique_classes = np.unique(y_train)
    if len(unique_classes) > 1:
        print("\nCalculating class weights to handle imbalance...")
        if isinstance(unique_classes[0], str):
            # For string labels
            class_weights = class_weight.compute_class_weight('balanced', classes=unique_classes, y=y_train)
            class_weights_dict = {unique_classes[i]: class_weights[i] for i in range(len(unique_classes))}
        else:
            # For numeric labels
            class_weights_dict = 'balanced'
        print(f"Class weights: {class_weights_dict}")
    else:
        class_weights_dict = None
        print("Warning: Only one class found in training data!")

    model = train_model(X_train, y_train, class_weights=class_weights_dict)
    print("Model trained.")

    # Evaluate the model
    y_pred = model.predict(X_test)
    
    # Handle case where target is not binary with 'Yes'/'No' values
    try:
        pos_label = 'Yes'
        precision = precision_score(y_test, y_pred, pos_label=pos_label)
        recall = recall_score(y_test, y_pred, pos_label=pos_label)
        f1 = f1_score(y_test, y_pred, pos_label=pos_label)
    except:
        # Fall back to default behavior if 'Yes' is not in the labels
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    try:
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print("ROC AUC Score:", roc_auc)
    except:
        print("ROC AUC Score could not be calculated (non-binary problem or issue with probabilities)")

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        # Get feature names
        feature_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if hasattr(transformer, 'get_feature_names_out'):
                try:
                    # For newer sklearn versions
                    transformed_names = transformer.get_feature_names_out(columns)
                    feature_names.extend(transformed_names)
                except:
                    # Fallback for older versions or other transformers
                    for col in columns:
                        feature_names.append(f"{name}_{col}")
            else:
                for col in columns:
                    feature_names.append(f"{name}_{col}")
                    
        # Display top 10 feature importances
        importances = pd.DataFrame({
            'feature': feature_names[:len(model.feature_importances_)],
            'importance': model.feature_importances_
        })
        importances = importances.sort_values('importance', ascending=False).head(10)
        print("\nTop 10 features by importance:")
        print(importances)

    # Save the model and preprocessor
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'churn_model.pkl')
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print(f"Model saved to {model_path}")
    print(f"Preprocessor saved to {preprocessor_path}")
    print("Training complete.")
