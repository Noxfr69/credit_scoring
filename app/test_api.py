import unittest
from unittest.mock import patch
import pandas as pd
from flask import json
from app import app

class TestFlaskAPI(unittest.TestCase):

    def setUp(self):
        app.testing = True
        self.client = app.test_client()

    def test_predict_endpoint_with_valid_data(self):
        with patch('app.mlflow.pyfunc.load_model', return_value=FakeModel()):
            X = pd.read_csv("./test/X_head", index_col=0)
            data = X.to_json(orient='split')
            response = self.client.post('/predict', data=data, content_type='application/json')
            self.assertEqual(response.status_code, 200)
            expected_response = [0.0, 1.0, 0.0, 1.0, 0.0]  # assuming your model predicts these values
            self.assertEqual(json.loads(response.data), expected_response)




class FakeModel:
    def predict(self, data):
        return [0, 1] 


if __name__ == '__main__':
    unittest.main()