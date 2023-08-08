import unittest
#from unittest.mock import patch
import pandas as pd
import json
import requests


class TestFlaskAPI(unittest.TestCase):

    def test_change_model_endpoint_200(self):
        model_id = "450ae60519ee43fda2402ae292be69d4"
        # Create data for the POST request
        data = {"model_id": model_id}
        response = requests.post('http://credit_scoring:80/new_model', data=json.dumps(data), headers={'Content-Type': 'application/json'})
        self.assertEqual(response.status_code, 200)

    def test_change_model_endpoint_400(self):
        model_id = None
        # Create data for the POST request
        data = {"model_id": model_id}
        response = requests.post('http://credit_scoring:80/new_model', data=json.dumps(data), headers={'Content-Type': 'application/json'})
        self.assertEqual(response.status_code, 400)

    def test_predict_endpoint_with_valid_data(self):
        X = pd.read_csv("./test/X_head", index_col=0)
        data = X.to_json(orient='split')
        response = requests.post('http://credit_scoring:80/predict', data=data)
        self.assertEqual(response.status_code, 200)
        expected_response = [0.0, 1.0, 0.0, 1.0, 0.0]  # assuming your model predicts these values
        self.assertEqual(response.json(), expected_response)

    def test_get_dataset_version(self):
        response = requests.get('http://credit_scoring:80/version')
        self.assertEqual(response.json(),"4.1" )





if __name__ == '__main__':
    unittest.main()