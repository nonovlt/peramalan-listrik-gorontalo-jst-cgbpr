import numpy as np
from typing import Tuple, Optional

class TimeSeriesPreprocessor:
    """
    Preprocessor untuk data time-series konsumsi energi listrik.
    Melakukan normalisasi Min-Max ke rentang [feature_min, feature_max] (default: [0.1, 0.9])
    dan pembentukan sliding window (lag features).
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
        # Normalisasi ke [feature_min, feature_max]
        scaled = (arr - self.data_min) / (self.data_max - self.data_min)
        scaled = scaled * (self.feature_max - self.feature_min) + self.feature_min
        return scaled

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.data_min is None or self.data_max is None:
            raise ValueError("Preprocessor belum di-fit dengan data!")
        arr = np.asarray(data, dtype=float)
        # Kembalikan ke skala original
        unscaled = (arr - self.feature_min) / (self.feature_max - self.feature_min)
        orig = unscaled * (self.data_max - self.data_min) + self.data_min
        return orig

    def create_sliding_window(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Membentuk matriks fitur X dan target Y menggunakan sliding window.
        X[i] = [y_{i}, y_{i+1}, ..., y_{i+lag-1}]
        Y[i] = y_{i+lag}
        """
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
        """
        Split time-series secara kronologis (tanpa shuffle untuk menghindari look-ahead bias).
        """
        n = len(X)
        split_idx = int(n * (1.0 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        Y_train, Y_test = Y[:split_idx], Y[split_idx:]
        return X_train, X_test, Y_train, Y_test
