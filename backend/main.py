from fastapi import FastAPI

app = FastAPI(
    title="Agentic DermaGuard API",
    version="0.1.0",
    description="Backend foundation for Agentic DermaGuard.",
)


@app.get("/health", tags=["operations"])
def health():
    return {"status": "ok"}