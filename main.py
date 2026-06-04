import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Initialiseer de FastAPI applicatie
app = FastAPI(
    title="Almere Lantaarnpaal Predictor API",
    description="API voor het voorspellen van aanrijdingen op schademasten.",
    version="1.1",
)

# 2. Laad het getrainde model in (No-nonsense: als dit faalt, start de API niet eens op)
model = joblib.load("lantaarnpaal_model.pkl")


# 3. Het Pydantic input-schema (exact 10 features, matchend met Colab)
class PredictionInput(BaseModel):
    latitude: float
    longitude: float
    mast_hoogte: float
    wegtype: int  # 0 = Woonwijk, 1 = Doorgaande weg
    dagdeel: int  # 0 = Overdag, 1 = Avond, 2 = Nacht
    temperatuur: float
    neerslag_mm: float
    gladheid_risico: int  # 0 = Nee, 1 = Ja
    lage_zon_risico: int  # 0 = Nee, 1 = Ja
    inwoners_gemeente: int = 223000  # Standaardwaarde voor Almere


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
    # De volgorde is nu 100% geverifieerd met de kolommen uit je Colab script
    features = [[
        data.latitude,
        data.longitude,
        data.mast_hoogte,
        data.wegtype,
        data.dagdeel,
        data.temperatuur,
        data.neerslag_mm,
        data.gladheid_risico,
        data.lage_zon_risico,
        data.inwoners_gemeente,
    ]]

    # Bereken de kans op basis van de twee klassen [kans_op_0, kans_op_1]
    kansen = model.predict_proba(features)
    kans_op_aanrijding = kansen[0][1]

    return {"kans_op_aanrijding": round(float(kans_op_aanrijding), 4)}


# 6. Automatische opstart-instructie voor Azure Linux containers
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
