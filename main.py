from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API to predict if a telecom customer will churn based on their profile.",
    version="1.0.0"
)

# Model load karein
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None

# Input Validation Schema
class CustomerInput(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    MonthlyCharges: float
    TotalCharges: float
    # Spaces aur brackets wale features ko alias ke zariye map karna[cite: 1]
    InternetService_Fiber_optic: int = Field(alias="InternetService_Fiber optic")
    InternetService_No: int
    Contract_One_year: int = Field(alias="Contract_One year")
    Contract_Two_year: int = Field(alias="Contract_Two year")
    PaymentMethod_Credit_card_automatic: int = Field(alias="PaymentMethod_Credit card (automatic)")
    PaymentMethod_Electronic_check: int = Field(alias="PaymentMethod_Electronic check")
    PaymentMethod_Mailed_check: int = Field(alias="PaymentMethod_Mailed check")

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running."}

@app.post("/predict")
def predict_churn(customer: CustomerInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file not found. Please run train.py first.")
        
    try:
        # Pydantic model ko dictionary mein convert karein aur original names (alias) use karein
        input_data = customer.dict(by_alias=True)
        
        # Single row ki DataFrame banayein
        df_input = pd.DataFrame([input_data])
        
        # Prediction aur probability calculate karein
        prediction = model.predict(df_input)
        probability = model.predict_proba(df_input)[:, 1]
        
        return {
            "prediction": int(prediction[0]),
            "churn_probability": round(float(probability[0]), 4),
            "status": "Churn" if int(prediction[0]) == 1 else "Not Churn"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))