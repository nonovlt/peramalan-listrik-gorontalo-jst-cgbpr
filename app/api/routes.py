import os
import shutil
import json
import asyncio
import pandas as pd
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional

from app.config import UPLOAD_DIR, REPORT_DIR, DATA_DIR
from app.services.training_service import app_state
from app.services.report_service import ReportService

router = APIRouter()

class TrainParams(BaseModel):
    lag: int = Field(default=6, ge=1, le=24)
    hidden_layers: str = Field(default="10")
    activation: str = Field(default="sigmoid")
    optimizer: str = Field(default="cgb")
    max_epochs: int = Field(default=300, ge=10, le=2000)
    target_error: float = Field(default=1e-5)
    restart_interval: int = Field(default=15, ge=2, le=100)
    powell_threshold: float = Field(default=0.2, ge=0.01, le=1.0)
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)

class ForecastParams(BaseModel):
    steps: int = Field(default=12, ge=1, le=60)

class BenchmarkParams(BaseModel):
    max_epochs: int = Field(default=150, ge=20, le=500)

@router.get("/status")
def get_system_status():
    has_data = app_state.df_data is not None and len(app_state.df_data) > 0
    has_model = app_state.model is not None
    has_evaluation = app_state.evaluation_results is not None
    has_forecast = app_state.forecast_results is not None
    has_eda = app_state.eda_results is not None
    has_benchmark = app_state.benchmark_results is not None

    return {
        "has_data": has_data,
        "data_count": len(app_state.df_data) if has_data else 0,
        "has_model": has_model,
        "training_status": app_state.training_status,
        "training_progress": round(app_state.training_progress, 1),
        "has_evaluation": has_evaluation,
        "has_forecast": has_forecast,
        "has_eda": has_eda,
        "has_benchmark": has_benchmark
    }

@router.post("/upload")
async def upload_data(file: Optional[UploadFile] = File(None), use_sample: bool = Form(False)):
    try:
        if use_sample or file is None:
            sample_path = DATA_DIR / "gorontalo_listrik_bulanan_2018_2024.csv"
            if not sample_path.exists():
                raise HTTPException(status_code=404, detail="Dataset sample belum tersedia!")
            df = pd.read_csv(sample_path)
            rows = app_state.load_data_from_dataframe(df)
            return {"status": "success", "message": f"Dataset sampel Kota Gorontalo berhasil dimuat ({rows} baris data).", "rows": rows}

        filename = file.filename
        ext = filename.split(".")[-1].lower()
        if ext not in ["csv", "xlsx", "xls"]:
            raise HTTPException(status_code=400, detail="Format file harus berupa CSV atau Excel (.xlsx/.xls)")

        filepath = UPLOAD_DIR / filename
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if ext == "csv":
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        rows = app_state.load_data_from_dataframe(df)
        return {"status": "success", "message": f"File '{filename}' berhasil diunggah ({rows} data).", "rows": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses file: {str(e)}")

@router.get("/data")
def get_data():
    if app_state.df_data is None:
        raise HTTPException(status_code=400, detail="Data belum dimuat.")

    df = app_state.df_data
    values = df["Konsumsi_kWh"].values
    stats = {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "total": float(np.sum(values))
    }
    return {
        "periods": df["Periode"].tolist(),
        "values": values.tolist(),
        "stats": stats,
        "table": df.to_dict(orient="records")
    }

@router.get("/eda/decomposition")
def get_decomposition():
    """Mengambil hasil dekomposisi musiman: Trend, Seasonal, Residual"""
    if app_state.eda_results is None or "decomposition" not in app_state.eda_results:
        if app_state.df_data is not None:
            # Hitung ulang
            from app.ml.diagnostics import TimeSeriesDiagnostics
            series = app_state.df_data["Konsumsi_kWh"].values
            app_state.eda_results = {
                "adf": TimeSeriesDiagnostics.adf_test(series),
                "decomposition": TimeSeriesDiagnostics.seasonal_decompose(series, period=12)
            }
        else:
            raise HTTPException(status_code=400, detail="Data belum dimuat.")

    return {
        "periods": app_state.df_data["Periode"].tolist(),
        "actual": app_state.df_data["Konsumsi_kWh"].tolist(),
        "decomposition": app_state.eda_results["decomposition"]
    }

@router.get("/eda/stationarity")
def get_stationarity():
    """Uji stasioneritas Augmented Dickey-Fuller (ADF)"""
    if app_state.eda_results is None or "adf" not in app_state.eda_results:
        raise HTTPException(status_code=400, detail="Data belum dimuat.")
    return app_state.eda_results["adf"]

def run_training_background(params: dict):
    try:
        app_state.train_model(params)
    except Exception as e:
        app_state.training_status = "error"
        print(f"Error training: {e}")

@router.post("/train")
async def start_training(params: TrainParams, background_tasks: BackgroundTasks):
    if app_state.df_data is None or len(app_state.df_data) == 0:
        raise HTTPException(status_code=400, detail="Data historis belum tersedia.")

    app_state.training_status = "training"
    app_state.training_progress = 0.0
    app_state.training_logs = []

    background_tasks.add_task(run_training_background, params.dict())
    return {"status": "started", "message": "Proses pelatihan model JST dimulai di background."}

@router.get("/train/status")
def get_train_status():
    return {
        "status": app_state.training_status,
        "progress": round(app_state.training_progress, 1),
        "latest": app_state.latest_epoch_data,
        "log_count": len(app_state.training_logs),
        "logs": app_state.training_logs[-20:] if app_state.training_logs else []
    }

@router.get("/train/stream")
async def stream_training_progress():
    """
    SSE Stream yang dioptimasi untuk Cloudflare (Proxy / Pages / Tunnel)
    Dilengkapi anti-buffering header dan heartbeat pings untuk mencegah timeout 524.
    """
    async def event_generator():
        sent_idx = 0
        heartbeat_counter = 0

        while True:
            curr_len = len(app_state.training_logs)
            if sent_idx < curr_len:
                batch = app_state.training_logs[sent_idx:curr_len]
                sent_idx = curr_len
                for item in batch:
                    yield f"data: {json.dumps(item)}\n\n"

            if app_state.training_status in ["completed", "error"]:
                yield f"data: {json.dumps({'status': app_state.training_status, 'done': True})}\n\n"
                break

            # Heartbeat ping setiap ~5 detik agar Cloudflare tidak memutus koneksi
            heartbeat_counter += 1
            if heartbeat_counter >= 50:
                yield ": keep-alive-ping\n\n"
                heartbeat_counter = 0

            await asyncio.sleep(0.08)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )

@router.get("/evaluation")
def get_evaluation():
    if app_state.evaluation_results is None:
        raise HTTPException(status_code=400, detail="Model belum selesai dilatih.")
    return app_state.evaluation_results

@router.get("/diagnostics")
def get_diagnostics():
    if app_state.residual_diagnostics is None:
        raise HTTPException(status_code=400, detail="Diagnostik residual belum tersedia. Latih model terlebih dahulu.")
    return app_state.residual_diagnostics

@router.post("/benchmark")
def run_benchmark(params: BenchmarkParams):
    """Menjalankan komparasi ilmiah: CG-BPR vs Standard CG (PR) vs SGD Momentum"""
    try:
        results = app_state.run_benchmark(max_epochs=params.max_epochs)
        return {"status": "success", "benchmark": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/benchmark")
def get_benchmark():
    if app_state.benchmark_results is None:
        raise HTTPException(status_code=400, detail="Benchmark belum dijalankan.")
    return app_state.benchmark_results

@router.post("/forecast")
def run_forecast(params: ForecastParams):
    try:
        results = app_state.forecast_future(steps=params.steps)
        return {
            "status": "success",
            "steps": params.steps,
            "forecast": results,
            "confidence_level": 0.95
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/export/csv")
def export_csv():
    if not app_state.forecast_results:
        raise HTTPException(status_code=400, detail="Belum ada data peramalan untuk diekspor.")
    out_path = REPORT_DIR / "peramalan_listrik_gorontalo_95ci.csv"
    ReportService.export_csv(app_state.forecast_results, str(out_path))
    return FileResponse(out_path, filename="peramalan_listrik_gorontalo_95ci.csv", media_type="text/csv")

@router.get("/export/excel")
def export_excel():
    if not app_state.forecast_results or not app_state.evaluation_results:
        raise HTTPException(status_code=400, detail="Data evaluasi atau peramalan belum lengkap.")
    out_path = REPORT_DIR / "peramalan_listrik_gorontalo_lengkap.xlsx"
    ReportService.export_excel(
        model_params=app_state.model_params,
        metrics=app_state.evaluation_results.get("test_metrics", {}),
        predictions=app_state.forecast_results,
        benchmark_results=app_state.benchmark_results,
        diagnostics=app_state.residual_diagnostics,
        filepath=str(out_path)
    )
    return FileResponse(
        out_path,
        filename="peramalan_listrik_gorontalo_lengkap.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@router.get("/export/pdf")
def export_pdf():
    if not app_state.forecast_results or not app_state.evaluation_results:
        raise HTTPException(status_code=400, detail="Data evaluasi atau peramalan belum lengkap.")
    out_path = REPORT_DIR / "laporan_ilmiah_peramalan_listrik_gorontalo.pdf"
    ReportService.export_pdf(
        model_params=app_state.model_params,
        metrics=app_state.evaluation_results.get("test_metrics", {}),
        predictions=app_state.forecast_results,
        benchmark_results=app_state.benchmark_results,
        diagnostics=app_state.residual_diagnostics,
        filepath=str(out_path)
    )
    return FileResponse(
        out_path,
        filename="laporan_ilmiah_peramalan_listrik_gorontalo.pdf",
        media_type="application/pdf"
    )
