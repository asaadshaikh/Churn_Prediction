import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(filepath):
    """
    Load data from a CSV file
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with the loaded data or None if error
    """
    try:
        print("Loading data from:", filepath)
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the data by scaling numerical features and one-hot encoding categorical features
    
    Args:
        df: DataFrame to preprocess
        
    Returns:
        Tuple of preprocessed data and preprocessor pipeline
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or not loaded.")
        return None, None

    print("Preprocessing data...")
    # Handle missing values
    print("Checking for missing values...")
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("Missing values found:")
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found.")
    
    # Define preprocessing for numerical columns (scaling and imputing)
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    print(f"Numerical features: {len(numerical_features)} columns")
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Define preprocessing for categorical columns (one-hot encoding)
    categorical_features = df.select_dtypes(include=['object']).columns
    print(f"Categorical features: {len(categorical_features)} columns")
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Preprocess the data
    print("Applying preprocessing pipeline...")
    df_preprocessed = preprocessor.fit_transform(df)
    print(f"Data preprocessed. Result shape: {df_preprocessed.shape}")
    return df_preprocessed, preprocessor

def save_preprocessed_data(df_preprocessed, filepath):
    """
    Save preprocessed data to a CSV file
    
    Args:
        df_preprocessed: Preprocessed data
        filepath: Path to save the CSV file
    """
    try:
        pd.DataFrame(df_preprocessed).to_csv(filepath, index=False)
        print(f"Preprocessed data saved to {filepath}")
    except Exception as e:
        print(f"Error saving preprocessed data: {e}")

if __name__ == "__main__":
    # Get the correct path for the data file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_path = os.path.join(project_dir, 'data', 'churn_data.csv')
    output_path = os.path.join(project_dir, 'data', 'preprocessed_churn_data.csv')
    
    print(f"Looking for data file at: {data_path}")
    df = load_data(data_path)
    if df is not None:
        df_preprocessed, preprocessor = preprocess_data(df)
        save_preprocessed_data(df_preprocessed, output_path)
        print("Preprocessing complete.")
    else:
        print("Preprocessing failed: Could not load data.")
