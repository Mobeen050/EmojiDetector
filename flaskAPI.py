# model_server.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

class ModelServer:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.last_prediction_time = None
        self.prediction_count = 0
    
    def preprocess_input(self, data):
        df = pd.DataFrame(data)
        # Add preprocessing steps here
        return df.values
    
    def predict(self, data):
        processed_data = self.preprocess_input(data)
        predictions = self.model.predict(processed_data)
        probabilities = self.model.predict_proba(processed_data)
        
        self.prediction_count += 1
        self.last_prediction_time = datetime.now()
        
        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'timestamp': self.last_prediction_time.isoformat()
        }

model_server = ModelServer('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        result = model_server.predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'prediction_count': model_server.prediction_count,
        'last_prediction': model_server.last_prediction_time
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)