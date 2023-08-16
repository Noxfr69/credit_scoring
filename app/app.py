from flask import Flask, request, jsonify, send_from_directory
import mlflow.pyfunc
import pandas as pd
import shap

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import numpy as np


app = Flask(__name__)

model = None
model_dataset_version = None
current_model_id = None
current_model_threshold = None

# Load the model
def change_model(model_id : str):
    global model, model_dataset_version, current_model_id, current_model_threshold

    #set the current model ID and load the model
    current_model_id = model_id
    model_uri = f"./mlruns/1/{model_id}/artifacts/model"
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Fetch the model run
    mlflow_client = mlflow.tracking.MlflowClient(tracking_uri='http://0.0.0.0:5000')
    run = mlflow_client.get_run(run_id=current_model_id)
    
    # Fetch the dataset_version param from the run data
    model_dataset_version = run.data.params['dataset_version']
    current_model_threshold = run.data.params['threshold']




@app.route('/predict', methods=['POST'])
def predict():
    class WrappedModel:
        def __init__(self, mlflow_model):
            self.model = mlflow_model

        def predict(self, X):
            return self.model.predict(X)

        def predict_proba(self, X):
            if hasattr(self.model._model_impl, "predict_proba"):
                return self.model._model_impl.predict_proba(X)
            else:
                raise AttributeError("Underlying model does not have a 'predict_proba' method.")
            
    global model, current_model_threshold
    # Get data from POST request
    data = request.get_json(force=True)
    data_df = pd.DataFrame(data)
    data_df.replace({None: np.nan}, inplace=True)

    #Get the predict_proba and predict 
    wrapped_model = WrappedModel(model)
    proba = wrapped_model.predict_proba(data_df)
    prediction = (proba[:,0] < float(current_model_threshold)).astype(int)

    # wrap them
    api_answers = []
    for i in range(0, len(prediction)):
        api_answer = [prediction[i], proba[i][0], proba[i][1]]
        api_answers.append(api_answer)

    # Return prediction
    api_answers = [[float(val) for val in sublist] for sublist in api_answers]

    return jsonify(api_answers)




@app.route('/version', methods=['GET'])
def return_version():
    return jsonify(model_dataset_version)




@app.route('/threshold', methods=['GET'])
def return_threshold():
    return jsonify(current_model_threshold)



@app.route('/global_importance', methods=['GET'])
def return_global_importance():
    global current_model_id 
    global_importance_path = f"./mlruns/1/{current_model_id}/artifacts/shap_summary.png"
    return send_from_directory(directory=f"./mlruns/1/{current_model_id}/artifacts", path= global_importance_path, filename='shap_summary.png')




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




@app.route('/get_local_importance', methods=['POST'])
def local_importance():
    class WrappedModel:
        def __init__(self, mlflow_model):
            self.model = mlflow_model

        def predict(self, X):
            return self.model.predict(X)

        def predict_proba(self, X):
            if hasattr(self.model._model_impl, "predict_proba"):
                return self.model._model_impl.predict_proba(X)
            else:
                raise AttributeError("Underlying model does not have a 'predict_proba' method.")
    global model
    wrapped_model = WrappedModel(model)
    actual_model = wrapped_model.model._model_impl

    data = request.get_json(force=True)
    data_df = pd.DataFrame(data)
    data_df.replace({None: np.nan}, inplace=True)

    # Create SHAP explainer using the underlying model object
    explainer = shap.TreeExplainer(actual_model)
    shap_values = explainer.shap_values(data_df)
    shap_html = shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], data_df.iloc[0,:], show=False)
    # Save the HTML content to a file
    html_filepath = "./Exports/shap_force_plot.html"
    shap.save_html(html_filepath, shap_html)
    
    #buf = io.BytesIO()
    # Now, create the waterfall plot using both positive_class_shap_values and expected_value
    #shap.plots.waterfall(shap_values_explaination)
    #shap.summary_plot(shap_values, data_df, show=False)
    #plt.savefig(buf, format='png')
    #buf.seek(0)
    #return send_file(buf, mimetype='image/png')
    

    return send_from_directory(directory='./Exports', path= "./Exports/shap_force_plot.html", filename='shap_force_plot.html')




if __name__ == '__main__':
    change_model('69c6dfea00f44d549419e7de4cf5262c')
    app.run(host='0.0.0.0', port=5001)



