import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.utils import class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping

column_names = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
]

df = pd.read_csv(
    "data/processed.cleveland.data",
    names=column_names
)

df.replace("?", np.nan, inplace=True)
df = df.astype(float)
df.fillna(df.median(), inplace=True)

df["target"] = (df["target"] > 0).astype(int)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = {0: weights[0], 1: weights[1]}

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", "Recall"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    class_weight=class_weight_dict,
    verbose=1
)

loss, accuracy, recall = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)
print("Test Recall:", recall)

y_pred_prob = model.predict(X_test)

precision, recall_arr, thresholds = precision_recall_curve(y_test, y_pred_prob)
f1 = 2 * (precision * recall_arr) / (precision + recall_arr + 1e-8)

best_threshold = thresholds[np.argmax(f1)]
print("Best threshold:", best_threshold)

y_pred = (y_pred_prob > best_threshold).astype(int)

print(classification_report(y_test, y_pred))

joblib.dump(scaler, "models/nn/scaler.pkl")
model.save("models/nn/nn_model.keras")

print("Model saved successfully!")