import pandas as pd
import os
import numpy as np
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
        
        # Basic data cleaning
        # Replace any missing values in numerical columns with median
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                print(f"Replacing missing values in {col} with median")
                df[col] = df[col].fillna(df[col].median())
        
        # Replace missing values in categorical columns with mode
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isnull().any():
                print(f"Replacing missing values in {col} with mode")
                df[col] = df[col].fillna(df[col].mode()[0])
        
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
    
    # Check for and handle invalid values in numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        # Convert any non-numeric values to NaN
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass
        
        # Replace NaN with median
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Define preprocessing for numerical columns (scaling and imputing)
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    print(f"Numerical features: {len(numerical_features)} columns")
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
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

def create_sample_data(output_path, num_samples=100):
    """
    Create a sample dataset if none exists
    
    Args:
        output_path: Path to save the sample data
        num_samples: Number of samples to generate
    """
    print(f"Creating sample dataset with {num_samples} samples...")
    
    # Generate random data
    np.random.seed(42)
    
    # Define the data structure
    data = {
        'customerID': [f'CUST{i:05d}' for i in range(num_samples)],
        'gender': np.random.choice(['Male', 'Female'], size=num_samples),
        'SeniorCitizen': np.random.choice([0, 1], size=num_samples),
        'Partner': np.random.choice(['Yes', 'No'], size=num_samples),
        'Dependents': np.random.choice(['Yes', 'No'], size=num_samples),
        'tenure': np.random.randint(0, 72, size=num_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], size=num_samples),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], size=num_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], size=num_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], size=num_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], size=num_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], size=num_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], size=num_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], size=num_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], size=num_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=num_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], size=num_samples),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ], size=num_samples),
        'MonthlyCharges': np.random.uniform(20, 120, size=num_samples).round(2),
        'TotalCharges': np.random.uniform(0, 8000, size=num_samples).round(2)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add churn column (make it realistic based on features)
    # Higher churn for month-to-month contract, shorter tenure, higher monthly charges
    churn_prob = (
        (df['Contract'] == 'Month-to-month') * 0.3 +
        (df['tenure'] < 12) * 0.3 +
        (df['MonthlyCharges'] > 70) * 0.2 +
        (df['InternetService'] == 'Fiber optic') * 0.1 +
        (df['PaymentMethod'] == 'Electronic check') * 0.1
    )
    # Normalize to 0-1 range
    churn_prob = (churn_prob - churn_prob.min()) / (churn_prob.max() - churn_prob.min()) * 0.7
    
    # Apply probability to generate Yes/No
    df['Churn'] = np.random.binomial(1, churn_prob, size=num_samples).astype(str)
    df['Churn'] = df['Churn'].replace({'1': 'Yes', '0': 'No'})
    
    # Save the data
    df.to_csv(output_path, index=False)
    print(f"Sample dataset created and saved to {output_path}")
    print(f"Churn distribution: {df['Churn'].value_counts(normalize=True)}")
    
    return df

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
    if os.path.exists(data_path):
        df = load_data(data_path)
    else:
        print(f"Data file not found at {data_path}. Creating sample data...")
        data_dir = os.path.dirname(data_path)
        os.makedirs(data_dir, exist_ok=True)
        df = create_sample_data(data_path, num_samples=7000)
    
    if df is not None:
        df_preprocessed, preprocessor = preprocess_data(df)
        save_preprocessed_data(df_preprocessed, output_path)
        print("Preprocessing complete.")
    else:
        print("Preprocessing failed: Could not load or create data.")
