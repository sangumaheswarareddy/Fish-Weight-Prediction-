import joblib
import numpy as np

model = joblib.load("fish_weight_model.pkl")

# Example input
# Replace values with actual measurements
sample = np.array([[25.0, 26.5, 29.0, 5.9, 3.7]])

prediction = model.predict(sample)
print("Predicted Fish Weight:", prediction[0])
