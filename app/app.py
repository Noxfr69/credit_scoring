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
    return jsonify(prediction.tolist())



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
