import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# We'll use all features from dataset
X = df.drop(columns=[data.target.name])
y = df[data.target.name]

# train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# eval
preds = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
print(f"Test RMSE: {rmse:.3f} (target units: median house value in 100k USD)")

# save artifacts
joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
print("Saved model.joblib and scaler.joblib")
