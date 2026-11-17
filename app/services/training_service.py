import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from app.ml.ann_cgb import BealePowellANN
from app.ml.preprocessor import TimeSeriesPreprocessor
from app.ml.metrics import evaluate_all
from app.ml.diagnostics import TimeSeriesDiagnostics
from app.ml.benchmark import BenchmarkEngine
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
            "activation": "sigmoid",
            "optimizer": "cgb",
            "max_epochs": DEFAULT_MAX_EPOCHS,
            "target_error": 1e-5,
            "restart_interval": DEFAULT_RESTART_INTERVAL,
            "powell_threshold": DEFAULT_POWELL_THRESHOLD,
            "test_size": 0.2
        }
        self.training_status: str = "idle"
        self.training_progress: float = 0.0
        self.training_logs: List[Dict[str, Any]] = []
        self.latest_epoch_data: Optional[Dict[str, Any]] = None
        self.evaluation_results: Optional[Dict[str, Any]] = None
        self.forecast_results: Optional[List[Dict[str, Any]]] = None
        self.eda_results: Optional[Dict[str, Any]] = None
        self.benchmark_results: Optional[Dict[str, Any]] = None
        self.residual_diagnostics: Optional[Dict[str, Any]] = None
        self.test_residual_std: float = 0.0

    def load_data_from_dataframe(self, df: pd.DataFrame):
        cols = {c.lower(): c for c in df.columns}
        val_col = None
        for candidate in ["konsumsi_kwh", "konsumsi", "kwh", "nilai", "value"]:
            if candidate in cols:
                val_col = cols[candidate]
                break
        if val_col is None:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                val_col = num_cols[-1]
            else:
                raise ValueError("Tidak ditemukan kolom numerik konsumsi listrik dalam file!")

        df_clean = pd.DataFrame()
        df_clean["Konsumsi_kWh"] = pd.to_numeric(df[val_col], errors='coerce').ffill()

        if "periode" in cols:
            df_clean["Periode"] = df[cols["periode"]].astype(str)
        elif "tahun" in cols and "bulan" in cols:
            t = df[cols["tahun"]].astype(str)
            b = df[cols["bulan"]].astype(str)
            df_clean["Tahun"] = t
            df_clean["Bulan"] = b
            df_clean["Periode"] = t + "-" + b
        else:
            df_clean["Periode"] = [f"M-{i+1}" for i in range(len(df_clean))]

        if "Tahun" not in df_clean.columns:
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

        # Hitung diagnostik EDA otomatis saat data berhasil dimuat
        try:
            series = df_clean["Konsumsi_kWh"].values
            adf_res = TimeSeriesDiagnostics.adf_test(series)
            decomp_res = TimeSeriesDiagnostics.seasonal_decompose(series, period=12)
            self.eda_results = {
                "adf": adf_res,
                "decomposition": decomp_res
            }
        except Exception as e:
            print(f"[EDA Warning] Gagal menghitung EDA otomatis: {e}")

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
        activation = self.model_params.get("activation", "sigmoid")
        optimizer = self.model_params.get("optimizer", "cgb")

        self.preprocessor = TimeSeriesPreprocessor(lag=lag)
        raw_values = self.df_data["Konsumsi_kWh"].values
        self.preprocessor.fit(raw_values)
        scaled_values = self.preprocessor.transform(raw_values)

        X, Y = self.preprocessor.create_sliding_window(scaled_values)
        X_train, X_test, Y_train, Y_test = self.preprocessor.train_test_split(X, Y, test_size=test_size)

        self.model = BealePowellANN(
            input_dim=lag,
            hidden_layers=hidden_layers,
            output_dim=1,
            optimizer=optimizer,
            activation=activation,
            max_epochs=max_epochs,
            target_error=target_error,
            restart_interval=restart_interval,
            powell_threshold=powell_threshold,
            random_state=42
        )

        self.training_status = "training"
        self.training_logs = []

        def training_callback(event: Dict[str, Any]):
            epoch = event["epoch"]
            progress = min(100.0, (epoch / max_epochs) * 100.0)
            event["progress"] = round(progress, 1)
            self.training_progress = progress
            self.latest_epoch_data = event
            self.training_logs.append(event)

        self.model.train(X_train, Y_train, callback=training_callback)

        # Evaluasi
        y_pred_train_scaled = self.model.predict(X_train)
        y_pred_test_scaled = self.model.predict(X_test)

        y_pred_train = self.preprocessor.inverse_transform(y_pred_train_scaled).flatten()
        y_true_train = self.preprocessor.inverse_transform(Y_train).flatten()

        y_pred_test = self.preprocessor.inverse_transform(y_pred_test_scaled).flatten()
        y_true_test = self.preprocessor.inverse_transform(Y_test).flatten()

        test_metrics = evaluate_all(y_true_test, y_pred_test)
        train_metrics = evaluate_all(y_true_train, y_pred_train)

        # Analisis Residual & Durbin-Watson
        residuals_test = y_true_test - y_pred_test
        self.test_residual_std = float(np.std(residuals_test))
        dw_stat = TimeSeriesDiagnostics.calculate_durbin_watson(residuals_test)
        resid_summary = TimeSeriesDiagnostics.residual_summary(residuals_test)

        self.residual_diagnostics = {
            "durbin_watson": dw_stat,
            "summary": resid_summary,
            "residuals": [round(float(r), 2) for r in residuals_test],
            "fitted": [round(float(p), 2) for p in y_pred_test],
            "actual": [round(float(a), 2) for a in y_true_test]
        }

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
            "restart_events": self.model.history["restart_events"],
            "diagnostics": self.residual_diagnostics
        }

        self.training_status = "completed"
        return self.evaluation_results

    def forecast_future(self, steps: int = 12) -> List[Dict[str, Any]]:
        """
        Melakukan peramalan n-langkah ke depan dengan estimasi
        Interval Kepercayaan 95% (Lower Bound, Point Forecast, Upper Bound).
        """
        if self.model is None or self.preprocessor is None or self.df_data is None:
            raise ValueError("Model belum dilatih!")

        lag = self.model_params["lag"]
        raw_series = self.df_data["Konsumsi_kWh"].values
        scaled_series = self.preprocessor.transform(raw_series)

        current_window = list(scaled_series[-lag:])
        last_period_str = str(self.df_data["Periode"].iloc[-1])
        try:
            parts = last_period_str.split("-")
            last_year = int(parts[0])
            last_month = int(parts[1])
        except Exception:
            last_year = 2024
            last_month = 12

        future_point_preds = []
        future_meta = []

        for step in range(1, steps + 1):
            inp = np.array(current_window[-lag:], dtype=float).reshape(1, -1)
            pred_scaled = self.model.predict(inp)[0, 0]
            pred_kwh = float(self.preprocessor.inverse_transform(np.array([[pred_scaled]]))[0, 0])

            next_month = (last_month + step - 1) % 12 + 1
            next_year = last_year + (last_month + step - 1) // 12
            month_name = MONTH_NAMES_ID[next_month - 1]
            period_str = f"{next_year}-{next_month:02d}"

            future_point_preds.append(pred_kwh)
            future_meta.append({
                "Langkah": step,
                "Periode": period_str,
                "Tahun": next_year,
                "Bulan": month_name,
            })
            current_window.append(pred_scaled)

        # Hitung Interval Kepercayaan 95%
        res_std = self.test_residual_std if self.test_residual_std > 0 else 0.05 * np.mean(future_point_preds)
        lower_bounds, upper_bounds = TimeSeriesPreprocessor.calculate_confidence_intervals(
            future_point_preds, res_std, confidence_level=0.95
        )

        future_predictions = []
        for i in range(len(future_meta)):
            item = future_meta[i]
            item["Konsumsi_Prediksi_kWh"] = round(future_point_preds[i], 2)
            item["Batas_Bawah_95_kWh"] = lower_bounds[i]
            item["Batas_Atas_95_kWh"] = upper_bounds[i]
            future_predictions.append(item)

        self.forecast_results = future_predictions
        return future_predictions

    def run_benchmark(self, max_epochs: int = 150) -> Dict[str, Any]:
        """Menjalankan pengujian komparasi CG-BPR vs CG-PR vs SGD"""
        if self.df_data is None or len(self.df_data) == 0:
            raise ValueError("Data historis belum tersedia.")

        series = self.df_data["Konsumsi_kWh"].values
        lag = self.model_params.get("lag", DEFAULT_LAG)
        hidden_layers = self.model_params.get("hidden_layers", DEFAULT_HIDDEN_NEURONS)
        if isinstance(hidden_layers, str):
            hidden_layers = [int(x.strip()) for x in hidden_layers.split(",") if x.strip()]

        res = BenchmarkEngine.run_comparison(
            series=series,
            lag=lag,
            hidden_layers=hidden_layers,
            max_epochs=max_epochs,
            activation=self.model_params.get("activation", "sigmoid"),
            restart_interval=self.model_params.get("restart_interval", DEFAULT_RESTART_INTERVAL),
            powell_threshold=self.model_params.get("powell_threshold", DEFAULT_POWELL_THRESHOLD),
            test_size=self.model_params.get("test_size", 0.2)
        )
        self.benchmark_results = res
        return res

app_state = AppState()
