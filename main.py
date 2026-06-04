import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Initialiseer de FastAPI applicatie
app = FastAPI(
    title="Almere Lantaarnpaal Predictor API",
    description="API voor het voorspellen van aanrijdingen op schademasten.",
    version="1.2",
)

# 2. Laad het getrainde model in
model = joblib.load("lantaarnpaal_model.pkl")


# 3. Het Pydantic input-schema
class PredictionInput(BaseModel):
    latitude: float
    longitude: float
    mast_hoogte: float
    wegtype: int  
    dagdeel: int  
    temperatuur: float
    neerslag_mm: float
    gladheid_risico: int  
    lage_zon_risico: int  
    inwoners_gemeente: int = 223000  


# 4. Het GET root-eindpunt
@app.get("/")
def root():
    return {
        "status": "De API is online!",
        "Ga naar": "/docs voor de interactieve documentatie.",
    }


# 5. Het POST voorspel-eindpunt
@app.post("/predict")
def predict_crash_chance(data: PredictionInput):
    # We maken er een dictionary van met de EXACTE kolomnamen uit Google Colab
    input_dict = {
        "latitude": [data.latitude],
        "longitude": [data.longitude],
        "mast_hoogte": [data.mast_hoogte],
        "wegtype": [data.wegtype],
        "dagdeel": [data.dagdeel],
        "temperatuur": [data.temperatuur],
        "neerslag_mm": [data.neerslag_mm],
        "gladheid_risico": [data.gladheid_risico],
        "lage_zon_risico": [data.lage_zon_risico],
        "inwoners_gemeente": [data.inwoners_gemeente]
    }

    # Omzetten naar een Pandas DataFrame zodat Scikit-Learn de kolomnamen herkent
    df_features = pd.DataFrame(input_dict)

    # Bereken de kans
    kansen = model.predict_proba(df_features)
    kans_op_aanrijding = kansen[0][1]

    return {"kans_op_aanrijding": round(float(kans_op_aanrijding), 4)}


# 6. Automatische opstart-instructie voor Azure
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
