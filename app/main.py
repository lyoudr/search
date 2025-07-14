from fastapi import FastAPI 


from app.routes import audio

app = FastAPI(
    title="Audio AI",
)
app.include_router(audio.router)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Audio Processing API"}