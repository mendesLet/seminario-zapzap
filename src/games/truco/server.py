from fastapi import FastAPI, Request
from pydantic import BaseModel
import threading

app = FastAPI()

# Shared resource
latest_prediction = None
prediction_event = threading.Event()

class Prediction(BaseModel):
    prediction: list

@app.post("/receive_prediction")
async def receive_prediction(prediction: Prediction):
    global latest_prediction
    latest_prediction = prediction.prediction
    prediction_event.set()  # Signal that a new prediction is available
    return {"success": True}

@app.get("/latest_prediction")
async def get_latest_prediction():
    global latest_prediction
    return {"prediction": latest_prediction}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
