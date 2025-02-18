from flask import Flask, render_template, request
import numpy as np
import pickle

model_path = 'model.pkl'  # Ensure this path is correct

# Try loading the model
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
        print("Model loaded successfully.")  # Optional: print if model is loaded
except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
    print(f"Error loading model: {e}")
    model = None

app = Flask(__name__, static_url_path='/static')

# Your other routes and functions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details')
def pred():
    return render_template('details.html')

@app.route('/crop_predict', methods=['GET', 'POST'])
def crop_predict():
    if request.method == 'POST':
        if not model:
            return "Model is not loaded properly. Please check the model file."

        # Get the input data from the form
        try:
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
        except ValueError as e:
            return f"Error in input data: {e}"

        # Make a prediction using the loaded model
        try:
            prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
            crop = prediction[0]
        except Exception as e:
            return f"Error during prediction: {e}"

        return render_template('crop_predict.html', crop=crop)
    else:
        return render_template('crop_predict.html', crop=None)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
