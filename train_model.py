import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

# Load the dataset
data = pd.read_csv("traffic_volume.csv")

# Combine date and time into a single datetime
data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['Time'], dayfirst=True)

# Extract datetime features
data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month
data['day'] = data['datetime'].dt.day
data['hour'] = data['datetime'].dt.hour
data['minute'] = data['datetime'].dt.minute
data['second'] = data['datetime'].dt.second

# Drop original date/time columns
data.drop(columns=['date', 'Time', 'datetime'], inplace=True)

# Fill missing numerical values
for col in ['temp', 'rain', 'snow']:
    data[col] = data[col].fillna(data[col].mean())

# Fill missing weather values
data['weather'] = data['weather'].fillna('Clouds')

# One-hot encode weather
data = pd.get_dummies(data, columns=['weather'])

# Optionally handle 'holiday' as categorical
data['holiday'] = data['holiday'].fillna('None')
data = pd.get_dummies(data, columns=['holiday'])

# Split features and target
X = data.drop(columns=['traffic_volume'])
y = data['traffic_volume']

# Save feature order
feature_names = X.columns.tolist()
with open("features.json", "w") as f:
    json.dump(feature_names, f)

# Scale inputs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
with open("scale.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Train model
model = RandomForestRegressor()
model.fit(X_scaled, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Evaluation
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
from sklearn.metrics import r2_score, mean_squared_error
print("Train R2:", r2_score(y_train, model.predict(x_train)))
print("Test R2:", r2_score(y_test, model.predict(x_test)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, model.predict(x_test))))
