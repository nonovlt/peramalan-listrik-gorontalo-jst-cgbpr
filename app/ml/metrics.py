import numpy as np

def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error (MSE)"""
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    return float(np.mean((y_true - y_pred) ** 2))

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE)"""
    return float(np.sqrt(calculate_mse(y_true, y_pred)))

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (MAPE) dalam persentase (%)"""
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    
    # Menghindari pembagian dengan 0
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Menghitung seluruh metrik evaluasi model"""
    return {
        "mse": calculate_mse(y_true, y_pred),
        "rmse": calculate_rmse(y_true, y_pred),
        "mape": calculate_mape(y_true, y_pred)
    }
