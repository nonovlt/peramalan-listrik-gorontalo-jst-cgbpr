import numpy as np
from typing import Dict, Any, List, Tuple
import scipy.stats as stats

class TimeSeriesDiagnostics:
    """
    Modul diagnostik lanjutan untuk data deret waktu kelistrikan:
    - Uji Stasioneritas Augmented Dickey-Fuller (ADF)
    - Dekomposisi Musiman Klasik (Trend, Seasonal, Residual)
    - Analisis Residual & Statistik Durbin-Watson
    """

    @staticmethod
    def adf_test(series: np.ndarray, max_lags: int = 4) -> Dict[str, Any]:
        """
        Melakukan Uji Augmented Dickey-Fuller (ADF) untuk menguji stasioneritas.
        H0: Data memiliki unit root (tidak stasioner).
        H1: Data tidak memiliki unit root (stasioner).
        """
        y = np.asarray(series, dtype=float).flatten()
        n = len(y)
        if n < 15:
            return {"error": "Data terlalu sedikit untuk uji ADF"}

        # First difference: dy_t = y_t - y_{t-1}
        dy = np.diff(y)
        # Lag level: y_{t-1}
        y_lag1 = y[:-1]

        # Buat regresi: dy_t = c + gamma * y_{t-1} + sum(beta_i * dy_{t-i})
        # Optimal lag selection via AIC sederhana
        best_aic = np.inf
        best_t_stat = 0.0
        best_lag = 1

        for p in range(1, min(max_lags + 1, n // 5)):
            Y_reg = dy[p:]
            X_cols = [np.ones(len(Y_reg)), y_lag1[p:]]
            for lag in range(1, p + 1):
                X_cols.append(dy[p - lag : -lag])

            X_mat = np.column_stack(X_cols)
            # OLS estimate: beta = (X^T X)^{-1} X^T Y
            try:
                beta, residuals, rank, s = np.linalg.lstsq(X_mat, Y_reg, rcond=None)
                e = Y_reg - np.dot(X_mat, beta)
                rss = np.sum(e ** 2)
                k = X_mat.shape[1]
                aic = len(Y_reg) * np.log(rss / len(Y_reg)) + 2 * k

                # Standard error of gamma (kolom ke-1)
                sigma2 = rss / (len(Y_reg) - k)
                cov_mat = sigma2 * np.linalg.pinv(np.dot(X_mat.T, X_mat))
                se_gamma = np.sqrt(max(cov_mat[1, 1], 1e-12))
                t_stat = beta[1] / se_gamma

                if aic < best_aic:
                    best_aic = aic
                    best_t_stat = t_stat
                    best_lag = p
            except Exception:
                continue

        # Nilai Kritis MacKinnon standar (untuk model dengan konstanta):
        critical_values = {
            "1%": -3.51,
            "5%": -2.89,
            "10%": -2.58
        }

        is_stationary = best_t_stat < critical_values["5%"]
        # Aproksimasi p-value
        if best_t_stat < -3.51:
            p_val = 0.005
        elif best_t_stat < -2.89:
            p_val = 0.035
        elif best_t_stat < -2.58:
            p_val = 0.075
        else:
            p_val = min(0.99, 0.10 + 0.3 * (best_t_stat - (-2.58)) / 2.0)

        return {
            "adf_statistic": float(round(best_t_stat, 4)),
            "p_value": float(round(p_val, 4)),
            "used_lag": best_lag,
            "critical_values": critical_values,
            "is_stationary": bool(is_stationary),
            "conclusion": "Data Stasioner (Tolak H0)" if is_stationary else "Data Memiliki Tren / Tidak Stasioner (Gagal Tolak H0)"
        }

    @staticmethod
    def seasonal_decompose(series: np.ndarray, period: int = 12) -> Dict[str, Any]:
        """
        Dekomposisi Deret Waktu Klasik (Model Aditif):
        Y(t) = Trend(t) + Seasonal(t) + Residual(t)
        """
        y = np.asarray(series, dtype=float).flatten()
        n = len(y)
        if n < period * 2:
            raise ValueError(f"Dekomposisi memerlukan minimal {period * 2} observasi bulanan.")

        # 1. Menghitung Trend menggunakan 2x12 Centered Moving Average
        trend = np.full(n, np.nan)
        half_p = period // 2
        for t in range(half_p, n - half_p):
            # Centered moving average untuk periode genap
            window = y[t - half_p : t + half_p + 1]
            # Bobot: ujung bernilai 0.5, tengah bernilai 1.0
            weights = np.ones(period + 1)
            weights[0] = 0.5
            weights[-1] = 0.5
            trend[t] = np.sum(window * weights) / period

        # 2. Detrending
        detrended = y - trend

        # 3. Menghitung Komponen Musiman (Seasonal Index) per Bulan (1-12)
        seasonal_index = np.zeros(period)
        for m in range(period):
            indices = [i for i in range(n) if i % period == m and not np.isnan(detrended[i])]
            seasonal_index[m] = np.mean([detrended[i] for i in indices]) if indices else 0.0

        # Normalisasi agar rata-rata efek musiman = 0
        seasonal_index -= np.mean(seasonal_index)

        # Komponen musiman sepanjang waktu
        seasonal_full = np.array([seasonal_index[i % period] for i in range(n)])

        # 4. Menghitung Komponen Residual (Irregular)
        residual = y - trend - seasonal_full

        # Ubah NaN menjadi None/null untuk serialisasi JSON
        trend_clean = [None if np.isnan(v) else round(float(v), 2) for v in trend]
        resid_clean = [None if np.isnan(v) else round(float(v), 2) for v in residual]

        month_names = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Ags", "Sep", "Okt", "Nov", "Des"]
        seasonal_monthly = [
            {"bulan": month_names[i], "indeks_musiman_kwh": round(float(seasonal_index[i]), 2)}
            for i in range(period)
        ]

        return {
            "trend": trend_clean,
            "seasonal": [round(float(v), 2) for v in seasonal_full],
            "residual": resid_clean,
            "seasonal_monthly_profile": seasonal_monthly
        }

    @staticmethod
    def calculate_durbin_watson(residuals: np.ndarray) -> Dict[str, Any]:
        """
        Statistik Durbin-Watson untuk mendeteksi autokorelasi pada residual:
        DW = sum_{t=2}^T (e_t - e_{t-1})^2 / sum_{t=1}^T e_t^2
        """
        e = np.asarray(residuals, dtype=float).flatten()
        diff_e = np.diff(e)
        sum_sq_diff = np.sum(diff_e ** 2)
        sum_sq_e = np.sum(e ** 2)

        if sum_sq_e == 0:
            dw = 2.0
        else:
            dw = float(sum_sq_diff / sum_sq_e)

        if 1.7 <= dw <= 2.3:
            interp = "Bebas Autokorelasi (Residual acak dan independen)"
            badge = "Ideal"
        elif dw < 1.7:
            interp = "Indikasi Autokorelasi Positif"
            badge = "Perlu Perhatian"
        else:
            interp = "Indikasi Autokorelasi Negatif"
            badge = "Perlu Perhatian"

        return {
            "durbin_watson": round(dw, 4),
            "interpretation": interp,
            "status": badge
        }

    @staticmethod
    def residual_summary(residuals: np.ndarray) -> Dict[str, Any]:
        """Ringkasan statistik distribusi residual"""
        e = np.asarray(residuals, dtype=float).flatten()
        mean_err = float(np.mean(e))
        std_err = float(np.std(e))
        skew_err = float(stats.skew(e)) if len(e) > 3 else 0.0
        kurt_err = float(stats.kurtosis(e)) if len(e) > 3 else 0.0

        # Histogram data
        counts, bin_edges = np.histogram(e, bins=10)
        hist_data = []
        for i in range(len(counts)):
            hist_data.append({
                "bin_start": round(float(bin_edges[i]), 2),
                "bin_end": round(float(bin_edges[i+1]), 2),
                "count": int(counts[i])
            })

        return {
            "mean": round(mean_err, 4),
            "std": round(std_err, 4),
            "skewness": round(skew_err, 4),
            "kurtosis": round(kurt_err, 4),
            "histogram": hist_data
        }
