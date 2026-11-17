import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from app.config import STATIC_DIR, TEMPLATE_DIR, DATA_DIR
from app.api.routes import router as api_router
from app.services.training_service import app_state
import pandas as pd

app = FastAPI(
    title="Sistem Peramalan Listrik Kota Gorontalo - JST CG-BPR",
    description="Aplikasi Web Machine Learning untuk Peramalan Konsumsi Energi Listrik Rumah Tangga Kota Gorontalo menggunakan JST Conjugate Gradient Beale-Powell Restart",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files & Templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# Include API Router
app.include_router(api_router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    # Otomatis muat sample dataset jika file default tersedia
    sample_file = DATA_DIR / "gorontalo_listrik_bulanan_2018_2024.csv"
    if sample_file.exists():
        try:
            df = pd.read_csv(sample_file)
            app_state.load_data_from_dataframe(df)
            print(f"[Startup] Default dataset loaded: {len(df)} records from {sample_file}")
        except Exception as e:
            print(f"[Startup] Error loading default dataset: {e}")

@app.get("/")
def render_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
