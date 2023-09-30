from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model  

app = Flask(__name__)

def lo_model():
    try:
        model = load_model('exp_01.h5')
        return model
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

model = lo_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'input_data' not in data:
            return jsonify({'error': 'Input data is missing'}), 400
        input_data = np.array(data['input_data'])
        # if input_data.shape != (14,):
        #     return jsonify({'error': 'Input data must contain 14 elements'}), 400
        prediction = model.predict(input_data.reshape(1,14,1)) 
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
