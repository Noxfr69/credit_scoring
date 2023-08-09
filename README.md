# credit_scoring
**A project made during my data science master degree**

- Full ML pipeline
    - Data cleaning and engineering taken from Kaggle
    - Multiple models built
    - **MLflow** pipeline for model selection
    - **Unittest** of the API
    - **API** built and deployed with selected model
    - **API** able to switch model at runtime
    - **CI/CD** via `Docker` automated with `GitHub Action` on push:
        - Docker container built and pushed to Docker Hub
        - Connect to the API server `(AWS EC2)` via `SSH`
        - API server pulls the new image



### Repo structure:
- At the base level you will find
    - 1 A Data aggregation, cleaning and exploration notebook (from kaggle)
    - 2 Mlflow pipeline and model building notebook
    - 3 A notebook to try the API when it's running localy or on AWS 
    - 4 A notebook that explore data drift
    - The test and train data

- In the App folder you will find everything related to the API
    - app.py the flask app 
    - mlruns folder is where every model was saved by mlflow
        - This is where we go to find and create the model use by the api
    - Dockerfile and .conf files instruction to build the Docker image
    - *requirements.txt a list of package needed 
    - test_api.py the unittest tests for the api

- .github/workflows you will find the action yaml file

- The Exports folder contain the HTML datadrift POC and is used to store temp graphs

-Add new stuff more more 
