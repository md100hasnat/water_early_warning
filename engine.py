import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

MODEL_FILE = "water_warning_model.pkl"

def generate_synthetic_data(samples=2000):
    """Generates synthetic water quality sensor readings based on WHO standards."""
    np.random.seed(42)
    
    ph = np.random.uniform(5.0, 10.0, samples)
    turbidity = np.random.uniform(0.1, 35.0, samples)  # NTU
    tds = np.random.uniform(50, 1500, samples)         # ppm
    temp = np.random.uniform(15, 38, samples)           # °C

    risk_labels = []
    for p, tur, t_ds, tm in zip(ph, turbidity, tds, temp):
        # Critical Outbreak Risk: Warm water + high turbidity (bacterial bloom risk) or extreme pH/TDS
        if (tur > 15.0 and tm > 28.0) or (p < 6.0 or p > 9.0) or (t_ds > 1000):
            risk_labels.append(2)  # High / Outbreak Alert
        # Moderate Risk: Minor deviations
        elif (tur > 5.0 and tm > 24.0) or (p < 6.5 or p > 8.5) or (t_ds > 500):
            risk_labels.append(1)  # Moderate Warning
        else:
            risk_labels.append(0)  # Safe

    df = pd.DataFrame({
        'pH': ph,
        'Turbidity': turbidity,
        'TDS': tds,
        'Temperature': temp,
        'Risk_Level': risk_labels
    })
    return df

def train_engine():
    """Trains and saves the outbreak prediction model."""
    df = generate_synthetic_data()
    X = df[['pH', 'Turbidity', 'TDS', 'Temperature']]
    y = df['Risk_Level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    acc = clf.score(X_test, y_test)
    print(f"✅ Outbreak Early Warning Engine Trained! Accuracy: {acc * 100:.2f}%")
    
    joblib.dump(clf, MODEL_FILE)
    return clf

def predict_water_safety(model, ph, turbidity, tds, temp):
    """Predicts risk tier based on input sensor parameters."""
    features = np.array([[ph, turbidity, tds, temp]])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    risk_mapping = {
        0: ("SAFE", "🟢", "Water parameters are within normal limits."),
        1: ("MODERATE WARNING", "🟡", "Water parameters show mild deterioration. Monitor closely."),
        2: ("OUTBREAK ALERT", "🔴", "CRITICAL RISK: High bacterial proliferation conditions detected!")
    }
    
    status, icon, message = risk_mapping[prediction]
    return status, icon, message, probabilities

if __name__ == "__main__":
    train_engine()
    