from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app) 

# Load the trained model and pipeline
try:
    print("Loading model and pipeline...")
    model = joblib.load("model.pkl")
    pipeline = joblib.load("pipeline.pkl")
    print("Loaded successfully!")
except Exception as e:
    print(f"Error loading files: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Ensure the order of columns matches what the pipeline expects
        # Pipeline expects: ['Brand', 'Display', 'Battery', 'Camera_Score', 'RAM', 'ROM']
        input_df = pd.DataFrame([{
            'Brand': data.get('Brand'),
            'Display': float(data.get('Display')),
            'Battery': float(data.get('Battery')),
            'Camera_Score': float(data.get('Camera_Score')),
            'RAM': float(data.get('RAM')),
            'ROM': float(data.get('ROM'))
        }])
        
        # Transform the data using the pipeline
        prepared_data = pipeline.transform(input_df)
        
        # Make the prediction
        prediction = model.predict(prepared_data)
        
        return jsonify({
            'status': 'success',
            'estimated_value': round(float(prediction[0]), 2)
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)