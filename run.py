import uvicorn
import sys
from pathlib import Path

# Add current folder to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    print("=" * 75)
    print(" SIKEL-GORONTALO: Sistem Peramalan Konsumsi Energi Listrik Kota Gorontalo")
    print(" Algoritma: Jaringan Saraf Tiruan (ANN) + CG Beale-Powell Restart (CG-BPR)")
    print(" Server berjalan di: http://127.0.0.1:8000")
    print(" Dokumentasi API Swagger: http://127.0.0.1:8000/docs")
    print("=" * 75)
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
