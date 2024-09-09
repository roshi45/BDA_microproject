from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the dataset
dataset = pd.read_csv("heart.csv")

# Train the model
predictors = dataset.drop("target", axis=1)
target = dataset["target"]
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.2, random_state=42)
lr = LogisticRegression()
lr.fit(X_train, Y_train)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    input_features = [float(x) for x in request.form.values()]
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_data = pd.DataFrame([input_features], columns=feature_names)

    # Perform prediction
    prediction = lr.predict(input_data)
    output = 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'
    
    return render_template('index.html', prediction_text=f'Prediction: {output}')

if __name__ == "__main__":
    app.run(debug=True)
