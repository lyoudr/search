from fastapi import FastAPI

from app.routes import audio, models, generate, documents
app = FastAPI(
    title="Audio AI",
)
app.include_router(audio.router)
app.include_router(models.router)
app.include_router(generate.router)
app.include_router(documents.router)



@app.get("/")
def read_root():
    return {"message": "Welcome to the Audio Processing API"}