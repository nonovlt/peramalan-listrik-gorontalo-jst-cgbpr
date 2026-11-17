import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from app.ml.ann_cgb import BealePowellANN
from app.ml.preprocessor import TimeSeriesPreprocessor
from app.ml.metrics import evaluate_all
from app.config import DEFAULT_LAG, DEFAULT_HIDDEN_NEURONS, DEFAULT_MAX_EPOCHS, DEFAULT_RESTART_INTERVAL, DEFAULT_POWELL_THRESHOLD

MONTH_NAMES_ID = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]

class AppState:
    def __init__(self):
        self.df_data: Optional[pd.DataFrame] = None
        self.preprocessor: Optional[TimeSeriesPreprocessor] = None
        self.model: Optional[BealePowellANN] = None
        self.model_params: Dict[str, Any] = {
            "lag": DEFAULT_LAG,
            "hidden_layers": DEFAULT_HIDDEN_NEURONS,
            "max_epochs": DEFAULT_MAX_EPOCHS,
            "target_error": 1e-5,
            "restart_interval": DEFAULT_RESTART_INTERVAL,
            "powell_threshold": DEFAULT_POWELL_THRESHOLD,
            "test_size": 0.2
        }
        self.training_status: str = "idle"  # idle | training | completed | error
        self.training_progress: float = 0.0
        self.training_logs: List[Dict[str, Any]] = []
        self.latest_epoch_data: Optional[Dict[str, Any]] = None
        self.evaluation_results: Optional[Dict[str, Any]] = None
        self.forecast_results: Optional[List[Dict[str, Any]]] = None
        self.event_queue: asyncio.Queue = asyncio.Queue()

    def load_data_from_dataframe(self, df: pd.DataFrame):
        # Normalisasi nama kolom jika diperlukan
        cols = {c.lower(): c for c in df.columns}
        
        # Deteksi kolom nilai konsumsi
        val_col = None
        for candidate in ["konsumsi_kwh", "konsumsi", "kwh", "nilai", "value"]:
            if candidate in cols:
                val_col = cols[candidate]
                break
        if val_col is None:
            # Gunakan kolom numerik terakhir
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                val_col = num_cols[-1]
            else:
                raise ValueError("Tidak ditemukan kolom numerik konsumsi listrik dalam file!")

        # Pastikan kolom standar tersedia
        df_clean = pd.DataFrame()
        df_clean["Konsumsi_kWh"] = pd.to_numeric(df[val_col], errors='coerce').ffill()

        # Deteksi Periode atau Tahun/Bulan
        if "periode" in cols:
            df_clean["Periode"] = df[cols["periode"]].astype(str)
        elif "tahun" in cols and "bulan" in cols:
            t = df[cols["tahun"]].astype(str)
            b = df[cols["bulan"]].astype(str)
            df_clean["Tahun"] = t
            df_clean["Bulan"] = b
            df_clean["Periode"] = t + "-" + b
        else:
            # Generate default periode
            df_clean["Periode"] = [f"M-{i+1}" for i in range(len(df_clean))]

        if "Tahun" not in df_clean.columns:
            # Coba ekstrak tahun dari Periode jika format YYYY-MM
            try:
                df_clean["Tahun"] = df_clean["Periode"].apply(lambda x: int(str(x).split("-")[0]))
            except Exception:
                df_clean["Tahun"] = 2024

        if "Bulan" not in df_clean.columns:
            try:
                df_clean["Bulan"] = df_clean["Periode"].apply(
                    lambda x: MONTH_NAMES_ID[int(str(x).split("-")[1]) - 1] if len(str(x).split("-")) > 1 else str(x)
                )
            except Exception:
                df_clean["Bulan"] = df_clean["Periode"]

        self.df_data = df_clean
        return len(df_clean)

    def train_model(self, params: Dict[str, Any]):
        if self.df_data is None or len(self.df_data) == 0:
            raise ValueError("Data historis belum dimuat!")

        self.model_params.update(params)
        lag = int(self.model_params["lag"])
        hidden_layers = self.model_params["hidden_layers"]
        if isinstance(hidden_layers, str):
            hidden_layers = [int(x.strip()) for x in hidden_layers.split(",") if x.strip()]
        max_epochs = int(self.model_params["max_epochs"])
        target_error = float(self.model_params.get("target_error", 1e-5))
        restart_interval = int(self.model_params.get("restart_interval", DEFAULT_RESTART_INTERVAL))
        powell_threshold = float(self.model_params.get("powell_threshold", DEFAULT_POWELL_THRESHOLD))
        test_size = float(self.model_params.get("test_size", 0.2))

        # 1. Preprocessing
        self.preprocessor = TimeSeriesPreprocessor(lag=lag)
        raw_values = self.df_data["Konsumsi_kWh"].values
        self.preprocessor.fit(raw_values)
        scaled_values = self.preprocessor.transform(raw_values)

        X, Y = self.preprocessor.create_sliding_window(scaled_values)
        X_train, X_test, Y_train, Y_test = self.preprocessor.train_test_split(X, Y, test_size=test_size)

        # 2. Inisialisasi Model
        self.model = BealePowellANN(
            input_dim=lag,
            hidden_layers=hidden_layers,
            output_dim=1,
            max_epochs=max_epochs,
            target_error=target_error,
            restart_interval=restart_interval,
            powell_threshold=powell_threshold,
            random_state=42
        )

        self.training_status = "training"
        self.training_logs = []

        # Callback untuk update progress
        def training_callback(event: Dict[str, Any]):
            epoch = event["epoch"]
            progress = min(100.0, (epoch / max_epochs) * 100.0)
            event["progress"] = round(progress, 1)
            self.training_progress = progress
            self.latest_epoch_data = event
            self.training_logs.append(event)
            # Push ke async queue jika ada subscriber
            try:
                self.event_queue.put_nowait(event)
            except Exception:
                pass

        # 3. Training Loop
        self.model.train(X_train, Y_train, callback=training_callback)

        # 4. Evaluasi Hasil Training & Testing
        y_pred_train_scaled = self.model.predict(X_train)
        y_pred_test_scaled = self.model.predict(X_test)

        y_pred_train = self.preprocessor.inverse_transform(y_pred_train_scaled).flatten()
        y_true_train = self.preprocessor.inverse_transform(Y_train).flatten()

        y_pred_test = self.preprocessor.inverse_transform(y_pred_test_scaled).flatten()
        y_true_test = self.preprocessor.inverse_transform(Y_test).flatten()

        test_metrics = evaluate_all(y_true_test, y_pred_test)
        train_metrics = evaluate_all(y_true_train, y_pred_train)

        # Siapkan plotting points kronologis
        # Indeks data X dimulai dari index ke-lag
        period_labels = self.df_data["Periode"].tolist()[lag:]
        n_train = len(y_true_train)
        train_periods = period_labels[:n_train]
        test_periods = period_labels[n_train:]

        self.evaluation_results = {
            "test_metrics": test_metrics,
            "train_metrics": train_metrics,
            "train_periods": train_periods,
            "train_actual": y_true_train.tolist(),
            "train_predicted": y_pred_train.tolist(),
            "test_periods": test_periods,
            "test_actual": y_true_test.tolist(),
            "test_predicted": y_pred_test.tolist(),
            "loss_history": self.model.history["loss"],
            "epoch_history": self.model.history["epoch"],
            "restart_events": self.model.history["restart_events"]
        }

        self.training_status = "completed"
        return self.evaluation_results

    def forecast_future(self, steps: int = 12) -> List[Dict[str, Any]]:
        """
        Melakukan peramalan rekursif (multi-step ahead) sebanyak `steps` bulan ke depan.
        """
        if self.model is None or self.preprocessor is None or self.df_data is None:
            raise ValueError("Model belum dilatih!")

        lag = self.model_params["lag"]
        raw_series = self.df_data["Konsumsi_kWh"].values
        scaled_series = self.preprocessor.transform(raw_series)

        # Ambil jendela lag terakhir sebagai input awal
        current_window = list(scaled_series[-lag:])

        # Ambil periode terakhir untuk menghitung nama periode ke depan
        last_period_str = str(self.df_data["Periode"].iloc[-1])
        try:
            parts = last_period_str.split("-")
            last_year = int(parts[0])
            last_month = int(parts[1])
        except Exception:
            last_year = 2024
            last_month = 12

        future_predictions = []
        for step in range(1, steps + 1):
            # Input shape: (1, lag)
            inp = np.array(current_window[-lag:], dtype=float).reshape(1, -1)
            pred_scaled = self.model.predict(inp)[0, 0]

            # Invers ke nilai kWh
            pred_kwh = float(self.preprocessor.inverse_transform(np.array([[pred_scaled]]))[0, 0])

            # Hitung tahun dan bulan berikutnya
            next_month = (last_month + step - 1) % 12 + 1
            next_year = last_year + (last_month + step - 1) // 12
            month_name = MONTH_NAMES_ID[next_month - 1]
            period_str = f"{next_year}-{next_month:02d}"

            future_predictions.append({
                "Langkah": step,
                "Periode": period_str,
                "Tahun": next_year,
                "Bulan": month_name,
                "Konsumsi_Prediksi_kWh": round(pred_kwh, 2)
            })

            # Geser sliding window rekursif
            current_window.append(pred_scaled)

        self.forecast_results = future_predictions
        return future_predictions

app_state = AppState()
