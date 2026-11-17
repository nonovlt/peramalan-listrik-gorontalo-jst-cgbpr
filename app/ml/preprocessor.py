import numpy as np
from typing import Tuple, Optional, List, Dict, Any

class TimeSeriesPreprocessor:
    """
    Preprocessor untuk data time-series konsumsi energi listrik.
    - Normalisasi Min-Max ke rentang [feature_min, feature_max] (default: [0.1, 0.9])
    - Pembentukan matriks sliding window (lag features)
    - Split kronologis tanpa shuffle
    - Estimasi Interval Kepercayaan (Confidence Intervals 95%)
    """
    def __init__(self, lag: int = 6, feature_range: Tuple[float, float] = (0.1, 0.9)):
        self.lag = lag
        self.feature_min, self.feature_max = feature_range
        self.data_min: Optional[float] = None
        self.data_max: Optional[float] = None

    def fit(self, data: np.ndarray):
        arr = np.asarray(data, dtype=float).flatten()
        self.data_min = float(np.min(arr))
        self.data_max = float(np.max(arr))
        if self.data_max == self.data_min:
            self.data_max += 1e-6
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.data_min is None or self.data_max is None:
            raise ValueError("Preprocessor belum di-fit dengan data!")
        arr = np.asarray(data, dtype=float)
        scaled = (arr - self.data_min) / (self.data_max - self.data_min)
        scaled = scaled * (self.feature_max - self.feature_min) + self.feature_min
        return scaled

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.data_min is None or self.data_max is None:
            raise ValueError("Preprocessor belum di-fit dengan data!")
        arr = np.asarray(data, dtype=float)
        unscaled = (arr - self.feature_min) / (self.feature_max - self.feature_min)
        orig = unscaled * (self.data_max - self.data_min) + self.data_min
        return orig

    def create_sliding_window(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(series, dtype=float).flatten()
        n = len(arr)
        if n <= self.lag:
            raise ValueError(f"Jumlah data ({n}) harus lebih besar dari lag ({self.lag})!")

        X, Y = [], []
        for i in range(n - self.lag):
            X.append(arr[i : i + self.lag])
            Y.append(arr[i + self.lag])

        return np.array(X, dtype=float), np.array(Y, dtype=float).reshape(-1, 1)

    def train_test_split(self, X: np.ndarray, Y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(X)
        split_idx = int(n * (1.0 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        Y_train, Y_test = Y[:split_idx], Y[split_idx:]
        return X_train, X_test, Y_train, Y_test

    @staticmethod
    def calculate_confidence_intervals(
        predictions: List[float],
        residual_std: float,
        confidence_level: float = 0.95
    ) -> Tuple[List[float], List[float]]:
        """
        Menghitung batas bawah dan batas atas untuk interval kepercayaan 95%.
        Deviasi standar melebar seiring bertambahnya horizon langkah (h).
        """
        # z-score untuk 95% = 1.96
        z = 1.96 if confidence_level == 0.95 else 2.576
        lower_bounds = []
        upper_bounds = []

        for h, y_hat in enumerate(predictions, start=1):
            # Penambahan ketidakpastian seiring langkah ke depan
            sigma_h = residual_std * np.sqrt(1.0 + 0.12 * (h - 1))
            margin = z * sigma_h
            lower = max(0.0, y_hat - margin) # Konsumsi listrik tidak negatif
            upper = y_hat + margin
            lower_bounds.append(round(float(lower), 2))
            upper_bounds.append(round(float(upper), 2))

        return lower_bounds, upper_bounds
