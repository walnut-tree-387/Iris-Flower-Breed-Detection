This is the Historic Iris Flower type detection project I started my Machine Learning journey with. This simple project takes in an array of Petal_length, Petal_width, Sepal_length and Sepal_width and detects the breed of Iris flower('Versicolor', 'Setosa', 'Virginica')

Flower_model.pth file has been trained and tested 150 entries collected from Iris Datasets. Model is running on evaluation mode, consuming the input and predict each of flower probability. Flower with highest probabily will bw returned as Output.


#####################
To run the project :
1. Clone the repo
2. python -m venv venv         # Create a virtual environment named 'venv'
3. source venv/bin/activate       # Activate the virtual environment
4. pip install -r requirements.txt  #Install the dependencies
5. uvicorn app.main:app --reload    #Run the Uvicorn server     


Application will be running on localhost:8000
Go to http://localhost:8000/docs to get the FastAPI swagger UI.

There are two endpoints:
1. Get Hello (http://localhost:8000)
2. Predict the flower (http://localhost:8000/predict)
   Example RequestBody : {
    "sepal_length": 7,
    "sepal_width": 2,
    "petal_length": 1,
    "petal_width": 1
    }