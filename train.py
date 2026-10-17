import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

def datapreparation(filepath):
    # Data load aur clean karein
    df = pd.read_csv(filepath)
    df.drop(["customerID"], inplace=True, axis=1)[cite: 1]
    
    df.TotalCharges = df.TotalCharges.replace(" ", np.nan)[cite: 1]
    df.TotalCharges.fillna(0, inplace=True)[cite: 1]
    df.TotalCharges = df.TotalCharges.astype(float)[cite: 1]
    
    # Binary columns mapping
    cols1 = ['Partner', 'Dependents', 'PaperlessBilling', 'Churn', 'PhoneService'][cite: 1]
    for col in cols1:
        df[col] = df[col].apply(lambda x: 0 if x == "No" else 1)[cite: 1]
   
    df.gender = df.gender.apply(lambda x: 0 if x == "Male" else 1)[cite: 1]
    df.MultipleLines = df.MultipleLines.map({'No phone service': 0, 'No': 0, 'Yes': 1})[cite: 1]
    
    cols2 = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'][cite: 1]
    for col in cols2:
        df[col] = df[col].map({'No internet service': 0, 'No': 0, 'Yes': 1})[cite: 1]
    
    # Categorical columns ke liye dummies banayein
    df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'], drop_first=True)[cite: 1]
    
    return df

if __name__ == "__main__":
    print("Loading and preparing data...")
    # Dataset ka path sahi se set karein
    df = datapreparation(filepath="data/Telco-Customer-Churn.csv")[cite: 1]
    
    # Features aur Target split karein
    x_cols = df.columns[df.columns != "Churn"][cite: 1]
    X = df[x_cols]
    y = df["Churn"][cite: 1]
    
    print("Training Random Forest Model...")
    # Best hyperparameters jo GridSearch se mile the
    model = RandomForestClassifier(
        bootstrap=True, 
        class_weight={0: 1, 1: 2},[cite: 1]
        criterion='entropy',[cite: 1]
        max_depth=10,[cite: 1]
        min_samples_leaf=1,[cite: 1]
        min_samples_split=8,[cite: 1]
        n_estimators=1000,[cite: 1]
        random_state=42
    )
    model.fit(X, y)
    
    print("Saving the model...")
    # Model ko pickle object ke roop mein save karein
    pickle.dump(model, open('model.pkl', 'wb'))[cite: 1]
    print("Model saved successfully as 'model.pkl'!")