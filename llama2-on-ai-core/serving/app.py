from fastapi import FastAPI, Request
from model_pipeline import Model

api = FastAPI()

@api.on_event("startup")
async def on_app_start():
    """this function is called on startup and facilitates the loading and setup of the model for inference"""
    Model.setup()

@api.post("/v2/predict")
async def predict(request: Request):
    """this function exposes the inference endpoint, expecting a json object with the prompt and a dictionary of arguments for the model"""
    request_content = await request.json()
    return Model.predict(request_content["prompt"], args=request_content["args"])

if __name__ == "__main__":
    # local testing
    import uvicorn
    uvicorn.run("app:api", host="0.0.0.0", port=8080, log_level="debug")