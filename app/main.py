from fastapi import FastAPI 
from prometheus_fastapi_instrumentator import Instrumentator

from app.routes import audio

app = FastAPI(
    title="Audio AI",
)
app.include_router(audio.router)
Instrumentator().instrument(app).expose(app)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Audio Processing API"}