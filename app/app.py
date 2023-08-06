from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)

# Load the model
model_uri = "./mlruns/1/450ae60519ee43fda2402ae292be69d4/artifacts/model"
model = mlflow.pyfunc.load_model(model_uri)


@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json(force=True)

    # Convert data into pandas DataFrame
    data_df = pd.DataFrame(data=data['data'], columns=data['columns'])

    # Make prediction
    prediction = model.predict(data_df)

    # Return prediction
    return jsonify(prediction)


@app.route('/change_model', methods=['POST'])
def change_model():
    global model
    model_id = request.get_json(force=True)['model_id']
    model_uri = f"./mlruns/1/{model_id}/artifacts/model"
    try:
        new_model = mlflow.pyfunc.load_model(model_uri)
        model = new_model
        return jsonify(status='success')
    except Exception as e:
        return jsonify(status='failure', error=str(e)), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
