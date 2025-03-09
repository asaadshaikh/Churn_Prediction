# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import pickle
from data_processing import load_data, preprocess_data

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    print("Starting model_training.py...")
    df = load_data('data/churn_data.csv')

    print("Data loaded.")
    columns = df.columns.tolist()
    print("Columns in DataFrame:", columns)

    # Replace "Churn" with the EXACT column name from the printed list.
    target_column = "Churn"  # <--- VERY IMPORTANT: Replace with the correct column name

    if target_column not in columns:
        print(f"Error: Target column '{target_column}' not found in DataFrame.")
        print("Please check the column name and update target_column.")
        exit()

    X = df.drop(target_column, axis=1)
    y = df[target_column]
    print("Features and target separated.")
    X_preprocessed, preprocessor = preprocess_data(X)
    print("Data preprocessed.")
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)
    print("Data split into train and test.")

    model = train_model(X_train, y_train)
    print("Model trained.")

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, pos_label='Yes'))

    print("Recall:", recall_score(y_test, y_pred, pos_label='Yes'))

    print("F1 Score:", f1_score(y_test, y_pred, pos_label='Yes'))

    print("ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1], multi_class='ovr'))

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the model and preprocessor
    with open('../models/churn_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('../models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print("Model trained and saved.")
    print("Training complete.")
