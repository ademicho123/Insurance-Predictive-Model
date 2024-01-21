from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the saved model
with open('ins_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    age = float(request.form['age'])
    
    # Make a prediction using the loaded model
    prediction = model.predict([[age]])

    # Display the prediction on the web page
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

