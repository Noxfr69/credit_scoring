import unittest
from unittest.mock import patch
from flask import json
from app import app

class TestFlaskAPI(unittest.TestCase):

    def setUp(self):
        app.testing = True
        self.client = app.test_client()

    def test_predict_endpoint_with_valid_data(self):
        with patch('app.mlflow.pyfunc.load_model', return_value=FakeModel()):
            data = {
                "data": [[1, 2, 3], [4, 5, 6]],
                "columns": ["A", "B", "C"]
            }
            response = self.client.post('/predict', data=json.dumps(data), content_type='application/json')
            self.assertEqual(response.status_code, 200)
            expected_response = [0, 1]  # assuming your model predicts these values
            self.assertEqual(json.loads(response.data), expected_response)

    def test_change_model_endpoint_with_valid_model_id(self):
        with patch('app.mlflow.pyfunc.load_model', return_value=FakeModel()):
            data = {"model_id": "some_valid_model_id"}
            response = self.client.post('/change_model', data=json.dumps(data), content_type='application/json')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(json.loads(response.data), {"status": "success"})

    def test_change_model_endpoint_with_invalid_model_id(self):
        with patch('app.mlflow.pyfunc.load_model', side_effect=Exception("Model not found")):
            data = {"model_id": "some_invalid_model_id"}
            response = self.client.post('/change_model', data=json.dumps(data), content_type='application/json')
            self.assertEqual(response.status_code, 400)
            self.assertEqual(json.loads(response.data), {"status": "failure", "error": "Model not found"})



class FakeModel:
    def predict(self, data):
        return [0, 1] 


if __name__ == '__main__':
    unittest.main()