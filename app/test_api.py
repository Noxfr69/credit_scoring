import unittest
#from unittest.mock import patch
import pandas as pd
import json
import requests


class TestFlaskAPI(unittest.TestCase):

    def test_change_model_endpoint_200(self):
        model_id = "69c6dfea00f44d549419e7de4cf5262c"
        # Create data for the POST request
        data = {"model_id": model_id}
        response = requests.post('http://localhost:80/api/new_model', data=json.dumps(data), headers={'Content-Type': 'application/json'})
        self.assertEqual(response.status_code, 200)

    def test_change_model_endpoint_400(self):
        model_id = None
        # Create data for the POST request
        data = {"model_id": model_id}
        response = requests.post('http://localhost:80/api/new_model', data=json.dumps(data), headers={'Content-Type': 'application/json'})
        self.assertEqual(response.status_code, 400)

    def test_predict_endpoint_with_valid_data(self):
        X = pd.read_csv("./test/X_head", index_col=0)
        data = X.to_json(orient='records')
        response = requests.post('http://localhost:80/api/predict', data=data)
        self.assertEqual(response.status_code, 200)

    def test_get_dataset_version(self):
        response = requests.get('http://localhost:80/api/version')
        self.assertEqual(response.json(),"2.0" )





if __name__ == '__main__':
    unittest.main()