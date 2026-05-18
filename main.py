import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Initialiseer de FastAPI applicatie
app = FastAPI(
    title="Almere Lantaarnpaal Predictor API",
    description="API voor het voorspellen van aanrijdingen op schademasten.",
    version="1.1",
)

# 2. Laad het getrainde model in
try:
    model = joblib.load("lantaarnpaal_model.pkl")
    print("Succes: Het ML-model is geladen!")
except Exception as e:
    print(
        f"Fout bij het laden van het model: {e}. Staat 'lantaarnpaal_model.pkl' wel in deze map?"
    )


# 3. Het Pydantic input-schema (exact 10 features, passend bij Stap 1)
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
    inwoners_gemeente: int = (
        223000  # Standaardwaarde voor Almere, mocht OutSystems dit vergeten
    )


# 4. Het POST-eindpunt
@app.post("/predict")
def predict_crash_chance(data: PredictionInput):
    # LET OP: De volgorde hieronder moet EXACT gelijk zijn aan de 'features_model' lijst uit Google Colab!
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

    # Bereken de kans
    kansen = model.predict_proba(features)
    kans_op_aanrijding = kansen[0][1]

    return {"kans_op_aanrijding": round(float(kans_op_aanrijding), 4)}


@app.get("/")
def root():
    return {
        "status": "De API is online!",
        "Ga naar": "/docs voor de interactieve documentatie.",
    }