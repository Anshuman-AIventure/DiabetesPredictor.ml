from flask import Flask, render_template, request
import numpy as np
import joblib
app = Flask(__name__)
# Load the trained model and scaler
model = joblib.load('svm_diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')  # Save your scaler as well
@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        # Get form data and convert to float
        features = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]
        # Scale input and predict
        features_scaled = scaler.transform([features])
        result = model.predict(features_scaled)[0]
        prediction = 'Diabetic' if result == 1 else 'Not Diabetic'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)