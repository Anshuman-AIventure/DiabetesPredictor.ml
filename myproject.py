import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
# Load the dataset
diabetes_data = pd.read_csv('diabetes.csv')  # Pima Indians Diabetes Database
# Note: The file path should be updated to the correct location of your dataset.
# Check for missing values
# Display the first few rows of the dataset
print(diabetes_data.head())
# Split the dataset into features and target variable
X = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Save the scaler for future use
import joblib
joblib.dump(scaler, 'scaler.pkl')
# Create and train the SVM model
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
# Print classification report
print(classification_report(y_test, y_pred))
# Save the model for future use
joblib.dump(model, 'svm_diabetes_model.pkl')
# Load the model and make predictions on new data
loaded_model = joblib.load('svm_diabetes_model.pkl')
new_data = np.array([[3,171,72,33,135,33.3,0.199,24]])  # Example new data
new_data_scaled = scaler.transform(new_data)
new_prediction = loaded_model.predict(new_data_scaled)
print(f'Prediction for new data: {new_prediction[0]}')  # Output the prediction
# The output will be 1 for diabetes and 0 for no diabetes
# End of the script
# This script demonstrates a simple SVM model for diabetes prediction using the Pima Indians Diabetes Database.