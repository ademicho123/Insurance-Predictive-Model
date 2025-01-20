import os
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the saved pipeline
model_path = os.path.join(os.getcwd(), 'models', 'ins_model.pkl')
loaded_pipeline = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the request body
    data = request.get_json()
    age = data['age']
    sex = data['sex']
    bmi = data['bmi']
    children = data['children']
    smoker = data['smoker']
    
    # Create a DataFrame with the user input
    user_input = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker]
    })
    
    # Make a prediction using the loaded model
    prediction = loaded_pipeline.predict(user_input)

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
