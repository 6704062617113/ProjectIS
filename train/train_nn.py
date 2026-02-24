import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
df = pd.read_csv("data/cardio_train.csv", sep=";")
df.columns = df.columns.str.strip()
df["age"] = df["age"] / 365
df = df[(df["height"] > 0) & (df["weight"] > 0)]
df = df[df["ap_lo"] <= df["ap_hi"]]
df["gender"] = df["gender"].map({1: 0, 2: 1})
df = df.drop("id", axis=1)
X = df.drop("cardio", axis=1)
y = df["cardio"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
model = Sequential([
    Input(shape=(X_scaled.shape[1],)),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

import joblib
joblib.dump(scaler, "models/nn/scaler.pkl")

model.save("models/nn/nn_model.keras")