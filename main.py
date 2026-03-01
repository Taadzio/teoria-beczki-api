from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Teoria Beczki działa ??"}

@app.get("/ping")
def ping():
    return {"status": "ok"}