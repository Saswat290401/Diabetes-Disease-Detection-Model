from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the XGBoost model
xgb_loaded_model = joblib.load('xgboost_diabetes_model.joblib')

# Load the Neural Network model
from tensorflow.keras.models import load_model
nn_loaded_model = load_model('neural_network_diabetes_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    from sklearn.preprocessing import StandardScaler  # Import inside the function

    data = request.get_json()

    # Preprocess the input data (convert to numpy array)
    sample_data = np.array([[data['pregnancies'], data['glucose'], data['bloodPressure'],
                             data['skinThickness'], data['insulin'], data['bmi'],
                             data['diabetesPedigreeFunction'], data['age']]], dtype=np.float32)

    # Create a new scaler for preprocessing
    numerical_features = [1, 2, 3, 4, 5, 6, 7]  # Indices of numerical features in the order they appear in the data
    scaler = StandardScaler()
    sample_data[:, numerical_features] = scaler.fit_transform(sample_data[:, numerical_features])

    # Calculate the Glucose_BMI_interaction feature
    glucose_bmi_interaction = sample_data[:, 1] * sample_data[:, 5]

    # Concatenate Glucose_BMI_interaction with sample_data
    sample_data_with_interaction = np.concatenate([sample_data, glucose_bmi_interaction.reshape(-1, 1)], axis=1)

    # Make predictions using the XGBoost model
    xgb_predictions = xgb_loaded_model.predict(sample_data_with_interaction)

    # Make predictions using the Neural Network model
    nn_input_data = np.concatenate([sample_data, glucose_bmi_interaction.reshape(-1, 1)], axis=1)
    nn_predictions = nn_loaded_model.predict(nn_input_data)

    # Convert Neural Network probabilities to binary predictions
    nn_binary_prediction = 1 if nn_predictions[0][0] > 0.5 else 0

    # Calculate the final combined outcome (XGBoost takes priority)
    final_outcome = int(xgb_predictions[0]) if xgb_predictions[0] == 0 else nn_binary_prediction

    # Prepare the result as JSON
    result = {
        'Predicted Outcome': final_outcome
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
