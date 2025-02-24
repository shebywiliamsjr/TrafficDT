# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split

# # Load data
# data = pd.read_csv("./model_data/preprocessed_data.csv")

# # Select relevant features
# features = ["Avg_X_Center", "Avg_Y_Center", "Avg_Speed", "Region_Encoded"]
# targets = ["Avg_X_Center", "Avg_Y_Center"]

# # Normalize data
# scaler = MinMaxScaler()
# data[features] = scaler.fit_transform(data[features])

# # Create sequences
# def create_sequences(data, features, targets, seq_length=10):
#     X, y = [], []
#     for i in range(len(data) - seq_length):
#         X.append(data[features].iloc[i:i+seq_length].values)
#         y.append(data[targets].iloc[i+seq_length].values)
#     return np.array(X), np.array(y)

# seq_length = 20  # Sequence length
# X, y = create_sequences(data, features, targets, seq_length)

# # Split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Build LSTM model
# model = Sequential([
#     LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, len(features))),
#     LSTM(50, activation='relu'),
#     Dense(32, activation='relu'),
#     Dense(len(targets))
# ])

# model.compile(optimizer='adam', loss='mse')

# # Train model
# epochs = 50
# batch_size = 32
# model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# # Save model
# model.save("lstm_vehicle_prediction.keras")

# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Compute evaluation metrics
# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error (MSE): {mse}")
# print(f"Mean Absolute Error (MAE): {mae}")
# print(f"R-Squared Score (R²): {r2}")

# import matplotlib.pyplot as plt

# plt.scatter(y_test[:, 0], y_test[:, 1], color='blue', label='Actual')
# plt.scatter(y_pred[:, 0], y_pred[:, 1], color='red', label='Predicted')
# plt.xlabel("Avg_X_Center")
# plt.ylabel("Avg_Y_Center")
# plt.legend()
# plt.title("Actual vs Predicted Vehicle Positions")
# plt.show()



import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Flatten, Concatenate
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv("./model_data/preprocessed_data.csv")

# Normalize continuous features
scaler = MinMaxScaler()
data[["Avg_X_Center", "Avg_Y_Center", "Avg_Speed"]] = scaler.fit_transform(data[["Avg_X_Center", "Avg_Y_Center", "Avg_Speed"]])

# One-hot encode categorical features (Vehicle_ID, Last_Region, is_red_light)
encoder = LabelEncoder()
data['Vehicle_ID'] = encoder.fit_transform(data['Vehicle_ID'])
region_encoder = LabelEncoder()
data['Last_Region'] = region_encoder.fit_transform(data['Last_Region'])
data['is_red_light'] = data['is_red_light'].astype(int)

print(data["Vehicle_ID"])

# Define features and targets
features = ["Avg_X_Center", "Avg_Y_Center", "Avg_Speed", "Vehicle_ID", "Last_Region", "is_red_light", "Region_Encoded"]
targets = ["Avg_X_Center", "Avg_Y_Center"]

# Create sequences function (including categorical features)
def create_sequences(data, features, targets, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        # Prepare the feature vector for each sequence (including categorical features)
        sequence = data[features].iloc[i:i+seq_length].values
        X.append(sequence)
        y.append(data[targets].iloc[i+seq_length].values)
    return np.array(X), np.array(y)

# Create sequences
seq_length = 20  # Sequence length
X, y = create_sequences(data, features, targets, seq_length)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model with categorical features
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, len(features))),
    LSTM(50, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(targets))  # Output layer with two predictions (Avg_X_Center, Avg_Y_Center)
])

model.compile(optimizer='adam', loss='mse')

# Train model
epochs = 50
batch_size = 32
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Save model
model.save("lstm_vehicle_prediction_with_routes.keras")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Make predictions on the test set
y_pred = model.predict(X_test)

# Compute evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-Squared Score (R²): {r2}")

import matplotlib.pyplot as plt

plt.scatter(y_test[:, 0], y_test[:, 1], color='blue', label='Actual')
plt.scatter(y_pred[:, 0], y_pred[:, 1], color='red', label='Predicted')
plt.xlabel("Avg_X_Center")
plt.ylabel("Avg_Y_Center")
plt.legend()
plt.title("Actual vs Predicted Vehicle Positions")
plt.show()

# how to check new vehicles? 
# will it be able to understanad the traffic light patterns?
# predict speed