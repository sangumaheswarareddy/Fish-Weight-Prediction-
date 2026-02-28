import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from data_preprocessing import load_data, preprocess_data

# Load and preprocess
df = load_data("data/fish.csv")
X, y = preprocess_data(df)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

# Save model
joblib.dump(model, "fish_weight_model.pkl")
print("Model saved successfully!")
