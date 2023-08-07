from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)

model = None
model_dataset_version = None
current_model_id = None

# Load the model
def change_model(model_id : str):
    global model, model_dataset_version, current_model_id

    #set the current model ID and load the model
    current_model_id = model_id
    model_uri = f"./mlruns/1/{model_id}/artifacts/model"
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Fetch the model run
    mlflow_client = mlflow.tracking.MlflowClient(tracking_uri='http://0.0.0.0:5000')
    run = mlflow_client.get_run(run_id=current_model_id)
    
    # Fetch the dataset_version param from the run data
    model_dataset_version = run.data.params['dataset_version']


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

@app.route('/version', methods=['GET'])
def return_version():
    return jsonify(model_dataset_version)

@app.route('/new_model', methods=['POST'])
def new_model():
    # Get data from POST request
    data = request.get_json(force=True)
    new_model_id = data.get('model_id', None)

    # Validate model_id
    if not new_model_id:
        return jsonify({'error': 'No model_id provided.'}), 400

    global current_model_id 
    if new_model_id == current_model_id:
        return jsonify({'message': 'Model already in use.'}), 200
    
    # Change model
    try:
        change_model(new_model_id)
        return jsonify({'message': 'Model updated successfully.'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    change_model('450ae60519ee43fda2402ae292be69d4')
    app.run(host='0.0.0.0', port=5001)

