import mlflow
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split


X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the tracking Server
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Now you can load the model from path mlflow
logged_model = 'runs:/660633bbc25d4833b6bc88abc8d4025c/iris_model'

loaded_model = mlflow.pyfunc.load_model(logged_model)
predictions = loaded_model.predict(X_test)
iris_feature_names = datasets.load_iris().feature_names
result = pd.DataFrame(X_test, columns=iris_feature_names)
result["Actual_class"] = y_test
result["predicted_class"] = predictions

print(result.head())