import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

model = joblib.load('k-Nearest Neighbours.pkl')

# Load the test data
test_data = pd.read_csv('content/MLOps-Task-1B/mobile_price_range_data.csv')

# Assuming the last column is the target
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Generate test predictions
# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

#save accuracy to a file
with open('accuracy.txt', 'w') as file:
    file.write(str(accuracy))