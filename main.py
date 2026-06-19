import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Initialiseer de FastAPI applicatie
app = FastAPI(
    title="Almere Lantaarnpaal Predictor API",
    description="API voor het voorspellen van aanrijdingen op schademasten (Versie 2 - Echte Data).",
    version="2.0",
)

# 2. Laad het getrainde model in
model = joblib.load("lantaarnpaal_model.pkl")


# 3. Het Pydantic input-schema (Aangepast aan de nieuwe werkelijkheid!)
# Jouw Low-Code app hoeft nu veel minder data mee te sturen naar Azure.
class PredictionInput(BaseModel):
    latitude: float
    longitude: float
    mast_hoogte: float
    temperatuur: float
    neerslag_mm: float


# 4. Het GET root-eindpunt
@app.get("/")
def root():
    return {
        "status": "De API is online!",
        "Versie": "2.0 (Getraind op echte data via objectnummers)",
        "Ga naar": "/docs voor de interactieve documentatie.",
    }


# 5. Het POST voorspel-eindpunt
@app.post("/predict")
def predict_crash_chance(data: PredictionInput):
    
    # Automatisch het gladheid_risico berekenen net zoals in de trainingsdata
    gladheid_risico = 1 if (data.temperatuur < 2 and data.neerslag_mm > 0) else 0

    # We maken een dictionary met de EXACTE 6 kolommen in de EXACTE volgorde van de training:
    input_dict = {
        "latitude": [data.latitude],
        "longitude": [data.longitude],
        "mast_hoogte": [data.mast_hoogte],
        "temperatuur": [data.temperatuur],
        "neerslag_mm": [data.neerslag_mm],
        "gladheid_risico": [gladheid_risico] # Deze berekent de API nu dus zelf!
    }

    # Omzetten naar een Pandas DataFrame
    df_features = pd.DataFrame(input_dict)

    # Bereken de kans
    kansen = model.predict_proba(df_features.values)
    kans_op_aanrijding = kansen[0][1]

    return {"kans_op_aanrijding": round(float(kans_op_aanrijding), 4)}


# 6. Automatische opstart-instructie voor Azure
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
