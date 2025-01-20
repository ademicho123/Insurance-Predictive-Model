import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import pickle
import joblib
import os

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the insurance dataset
    """
    # Loading the data set
    ins = pd.read_csv(file_path)
    
    # Converting the smoker column into numerical format
    ins.dropna(subset=['smoker'], inplace=True)
    ins['smoker'] = ins['smoker'].map({'yes': 1, 'no': 0})
    
    # Converting the sex column into numerical format
    ins['sex'] = ins['sex'].map({'male': 0, 'female': 1})
    
    # Dropping the 'region' column
    ins.drop('region', axis=1, inplace=True)
    
    return ins

def balance_dataset(data):
    """
    Balance the dataset across all features, prioritizing the smoker balance first
    """
    print("\nOriginal dataset shape:", data.shape)
    print("\nInitial distribution:")
    for col in ['sex', 'smoker']:
        print(f"\n{col} distribution:\n", data[col].value_counts())
    
    # Start with smoker balance first
    balanced_data = data.copy()
    smoker_counts = balanced_data['smoker'].value_counts()
    max_smoker_size = smoker_counts.max()
    
    print(f"\nBalancing smoker to {max_smoker_size} samples per category")
    
    # Balance smokers
    balanced_dfs = []
    for smoker_value in [0, 1]:
        subset = balanced_data[balanced_data['smoker'] == smoker_value]
        if len(subset) < max_smoker_size:
            upsampled = resample(subset,
                               replace=True,
                               n_samples=max_smoker_size,
                               random_state=42)
            balanced_dfs.append(upsampled)
        else:
            # If it's the majority class, downsample it
            downsampled = resample(subset,
                                 replace=False,
                                 n_samples=max_smoker_size,
                                 random_state=42)
            balanced_dfs.append(downsampled)
    
    balanced_data = pd.concat(balanced_dfs, ignore_index=True)
    print("After balancing smoker:", balanced_data['smoker'].value_counts())
    
    # Now balance sex while maintaining smoker balance
    balanced_by_smoker_sex = []
    for smoker_value in [0, 1]:
        smoker_data = balanced_data[balanced_data['smoker'] == smoker_value]
        sex_counts = smoker_data['sex'].value_counts()
        max_sex_size = sex_counts.max()
        
        sex_balanced_dfs = []
        for sex_value in [0, 1]:
            subset = smoker_data[smoker_data['sex'] == sex_value]
            if len(subset) < max_sex_size:
                upsampled = resample(subset,
                                   replace=True,
                                   n_samples=max_sex_size,
                                   random_state=42)
                sex_balanced_dfs.append(upsampled)
            else:
                sex_balanced_dfs.append(subset)
        
        smoker_sex_balanced = pd.concat(sex_balanced_dfs, ignore_index=True)
        balanced_by_smoker_sex.append(smoker_sex_balanced)
    
    balanced_data = pd.concat(balanced_by_smoker_sex, ignore_index=True)
    
    print("\nFinal balanced dataset shape:", balanced_data.shape)
    print("\nFinal distribution:")
    for col in ['sex', 'smoker']:
        print(f"\n{col} distribution:\n", balanced_data[col].value_counts())
    
    return balanced_data

def prepare_features(data, increase_factor=5):
    """
    Prepare X and y variables with balanced classes and increased dataset size
    """
    print(f"\nStarting data preparation with increase factor: {increase_factor}")
    print("Initial dataset shape:", data.shape)
    
    # First balance the dataset
    balanced_data = balance_dataset(data)
    
    # Separate features and target
    X = balanced_data[['age', 'sex', 'bmi', 'children', 'smoker']]
    y = balanced_data['charges']
    
    print("\nShape after balancing:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to DataFrame to maintain column names
    X_scaled = pd.DataFrame(X_scaled, columns=['age', 'sex', 'bmi', 'children', 'smoker'])
    
    # Increase the dataset size
    X_increased = pd.concat([X_scaled] * increase_factor, ignore_index=True)
    y_increased = pd.concat([pd.Series(y)] * increase_factor, ignore_index=True)
    
    # Add some random noise to continuous features to prevent exact duplicates
    continuous_cols = ['age', 'bmi', 'children']
    for col in continuous_cols:
        noise = np.random.normal(0, X_increased[col].std() * 0.01, size=len(X_increased))
        X_increased[col] = X_increased[col] + noise
    
    # Add some random noise to target variable
    noise = np.random.normal(0, y_increased.std() * 0.01, size=len(y_increased))
    y_increased = y_increased + noise
    
    print("\nFinal shapes:")
    print("X shape:", X_increased.shape)
    print("y shape:", y_increased.shape)
    
    return X_increased.values, y_increased.values

def verify_balance(X, y, original_data):
    """
    Verify the balance of the dataset after processing
    
    Parameters:
    X : array-like
        The processed feature matrix
    y : array-like
        The processed target values
    original_data : pandas.DataFrame
        The original dataset for comparison
    """
    X_df = pd.DataFrame(X, columns=['age', 'sex', 'bmi', 'children', 'smoker'])
    
    print("\nOriginal vs Final distributions:")
    
    # Compare categorical distributions
    print("\nSmoker distribution:")
    print("Original:", original_data['smoker'].value_counts(normalize=True))
    print("Final:", X_df['smoker'].value_counts(normalize=True))
    
    print("\nSex distribution:")
    print("Original:", original_data['sex'].value_counts(normalize=True))
    print("Final:", X_df['sex'].value_counts(normalize=True))
    
    # Joint distribution
    print("\nJoint distribution (sex, smoker):")
    print("Original:")
    print(pd.crosstab(original_data['sex'], original_data['smoker'], normalize='all'))
    print("\nFinal:")
    print(pd.crosstab(X_df['sex'], X_df['smoker'], normalize='all'))
    
    # Compare continuous features
    print("\nContinuous features summary:")
    for feature in ['age', 'bmi', 'children']:
        print(f"\n{feature}:")
        print("Original quantiles:", original_data[feature].quantile([0.25, 0.5, 0.75]).values)
        print("Final quantiles:", X_df[feature].quantile([0.25, 0.5, 0.75]).values)
    
    print("\nCharges:")
    print("Original quantiles:", original_data['charges'].quantile([0.25, 0.5, 0.75]).values)
    print("Final quantiles:", pd.Series(y).quantile([0.25, 0.5, 0.75]).values)

def plot_visuals(data):
    """
    Create visualizations for the data
    """
    # Scatter plot of age vs charges
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='age', y='charges')
    plt.title('Scatterplot for Charges')
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.show()
    plt.close()

    # Histogram of charges
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='charges', kde=True)
    plt.title('Histogram of Charges')
    plt.xlabel('Charges')
    plt.ylabel('Frequency')
    plt.show()
    plt.close()

    # Bar chart of smoker vs charges
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x='smoker', y='charges')
    plt.title('Bar Chart of Smoker vs Charges')
    plt.xlabel('Smoker')
    plt.ylabel('Charges')
    plt.show()
    plt.close()

def train_model(X, y, test_size=0.2, random_state=42):
    """
    Split data, train the model, and tune hyperparameters
    """
    print("\nStarting model training...")
    print("Input data shapes:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    
    # Convert X to DataFrame if it's not already
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=['age', 'sex', 'bmi', 'children', 'smoker'])
    
    # Create bins for charges to use in stratification
    y_bins = pd.qcut(y, q=5, labels=False)
    
    # Splitting the data with stratification
    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y_bins
    )
    
    print("\nAfter train-test split:")
    print("Training set shape:", x_train.shape)
    print("Test set shape:", x_test.shape)
    
    # Adjust hyperparameters based on dataset size
    n_cv_folds = 3 if len(y) > 10000 else 5
    
    # Define the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(random_state=random_state))
    ])

    # Reduce parameter grid size for larger datasets
    if len(y) > 10000:
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 30],
            'model__min_samples_split': [5, 10],
            'model__min_samples_leaf': [2]
        }
        print("\nUsing reduced parameter grid for large dataset")
    else:
        param_grid = {
            'model__n_estimators': [50, 100, 150, 200, 250],
            'model__max_depth': [None, 10, 20, 30, 40],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
        print("\nUsing full parameter grid")

    print(f"\nUsing {n_cv_folds}-fold cross validation")
    print("Parameter grid:", param_grid)
    
    # Perform Grid Search
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=n_cv_folds,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(x_train, y_train)
    
    # Print best parameters
    print("\nBest parameters:", grid_search.best_params_)
    
    # Best model from grid search
    best_model = grid_search.best_estimator_

    # Evaluate with cross-validation
    cross_val_scores = cross_val_score(
        best_model,
        x_train,
        y_train,
        cv=n_cv_folds,
        scoring='neg_mean_absolute_error'
    )
    mean_cv_mae = -np.mean(cross_val_scores)
    print(f"Mean CV MAE: {mean_cv_mae}")

    return best_model, x_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Make predictions and evaluate the model
    """
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    # Calculate MAPE, handling the case where the actual value is zero
    y_test_non_zero = y_test != 0
    mape = np.mean(np.abs((y_test[y_test_non_zero] - predictions[y_test_non_zero]) / y_test[y_test_non_zero])) * 100
    
    return predictions, mae, mape


def save_model(model, filename):
    # Specify the models folder path
    models_folder = r'C:\Users\ELITEBOOK\OneDrive\Desktop\Projects\Insurance-Predictive-Model-1\models'
    
    # Create the models folder if it doesn't exist
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    
    # Save the model to the file
    model_path = os.path.join(models_folder, filename)
    joblib.dump(model, model_path)

