import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv("data/diabetes.csv")
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols:
    df[col] = df[col].replace(0, np.nan)
df.fillna(df.median(), inplace=True)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
knn = KNeighborsClassifier()
svm = SVC(probability=True)
rf = RandomForestClassifier()
ensemble = VotingClassifier(
    estimators=[
        ('knn', knn),
        ('svm', svm),
        ('rf', rf)
    ],
    voting='soft'
)
ensemble.fit(X_train, y_train)
pred = ensemble.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ml")
os.makedirs(MODEL_PATH, exist_ok=True)
pickle.dump(ensemble, open(os.path.join(MODEL_PATH, "ml_model.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(MODEL_PATH, "scaler.pkl"), "wb"))