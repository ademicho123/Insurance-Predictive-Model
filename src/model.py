import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle

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

def prepare_features(data):
    """
    Prepare X and y variables
    """
    X = data['age']
    y = data['charges']
    return X, y

def plot_scatter(data):
    """
    Create a scatter plot of age vs charges
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='age', y='charges')
    plt.title('Scatterplot for Charges')
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.show()
    plt.close()

def train_model(X, y, test_size=0.2, random_state=42):
    """
    Split data and train the model
    """
    # Splitting the data
    x_train, x_test, y_train, y_test = train_test_split(X, y, 
                                                       test_size=test_size, 
                                                       random_state=random_state)
    
    # Creating and training the model
    model = RandomForestRegressor(random_state=random_state)
    model.fit(x_train.values.reshape(-1, 1), y_train)
    
    return model, x_test, y_test

def save_model(model, file_path):
    """
    Save the trained model to a pickle file
    """
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

def evaluate_model(model, X_test, y_test):
    """
    Make predictions and evaluate the model
    """
    predictions = model.predict(X_test.values.reshape(-1, 1))
    mae = mean_absolute_error(y_test, predictions)
    return predictions, mae