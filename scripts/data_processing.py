import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(filepath):
    try:
        print("Loading data from:", filepath)
        df = pd.read_csv(filepath)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def preprocess_data(df):
    if df is None or df.empty:
        print("Error: DataFrame is empty or not loaded.")
        return None, None

    print("Preprocessing data...")
    # Define preprocessing for numerical columns (scaling and imputing)
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Define preprocessing for categorical columns (one-hot encoding)
    categorical_features = df.select_dtypes(include=['object']).columns
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
    df_preprocessed = preprocessor.fit_transform(df)
    print("Data preprocessed.")
    return df_preprocessed, preprocessor

def save_preprocessed_data(df_preprocessed, filepath):
    pd.DataFrame(df_preprocessed).to_csv(filepath, index=False)

if __name__ == "__main__":
    df = load_data('../data/churn_data.csv')
    df_preprocessed, preprocessor = preprocess_data(df)
    save_preprocessed_data(df_preprocessed, '../data/preprocessed_churn_data.csv')
    print("Preprocessing complete.")
