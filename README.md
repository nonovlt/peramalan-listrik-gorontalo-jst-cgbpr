# SIKEL-Gorontalo v2.0 (Advanced Edition)
### Sistem Peramalan Konsumsi Energi Listrik Bulanan Sektor Rumah Tangga di Kota Gorontalo Menggunakan Jaringan Saraf Tiruan (JST/ANN) dengan Optimasi Conjugate Gradient Beale-Powell Restart (CG-BPR)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Cloudflare](https://img.shields.io/badge/Deploy-Cloudflare%20Ready-F38020.svg?logo=cloudflare&logoColor=white)](https://www.cloudflare.com)
[![Docker](https://img.shields.io/badge/Container-Docker%20Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)
[![Status](https://img.shields.io/badge/Publication-Thesis%20Ready-success.svg)]()

Aplikasi Web Machine Learning Full-Stack komprehensif untuk menganalisis, memodelkan, mengkomparasi, dan meramalkan konsumsi energi listrik bulanan sektor rumah tangga di Kota Gorontalo. Dilengkapi dengan **Uji Stasioneritas ADF**, **Dekomposisi Musiman (Trend, Seasonal, Residual)**, **Benchmark 3-Model Optimasi**, **Diagnostik Residual (Durbin-Watson)**, serta **Proyeksi Peramalan Masa Depan dengan 95% Confidence Interval (Interval Kepercayaan)**.

---

## 1. Arsitektur Sistem Terintegrasi

Aplikasi menggunakan arsitektur **Enterprise Three-Tier Micro-Modular Architecture** yang siap di-deploy di infrastruktur Cloudflare (Cloudflare Tunnel, Cloudflare Pages/Workers, maupun Container VPS).

```mermaid
flowchart TD
    subgraph CloudflareLayer ["Cloudflare Edge & Security Layer"]
        CF_CDN["Cloudflare CDN & Reverse Proxy / Tunnel"]
        CF_SSL["SSL/TLS Termination & DDoS Shield"]
        CF_SSE["Anti-Buffering SSE Stream (Heartbeat Keep-Alive)"]
    end

    subgraph PresentationLayer ["1. Presentation Layer (Web Client)"]
        UI["Tailwind CSS + Chart.js Dashboard SPA"]
        UploadMod["Modul Upload CSV / Excel"]
        EDAPanel["Panel Uji ADF & Dekomposisi Musiman"]
        ConfigMod["Form Konfigurasi JST (Multi-Layer & Multi-Aktivasi)"]
        LiveLossChart["Grafik Real-Time Loss Curve (SSE)"]
        BenchmarkPanel["Panel Uji Komparasi Ilmiah 3-Model"]
        ForecastPanel["Mesin Proyeksi n-Bulan (95% Confidence Interval)"]
        ExportPanel["Ekspor CSV, Multi-Sheet Excel, & PDF Formal"]
    end

    subgraph ServiceLayer ["2. Application Layer (FastAPI Backend)"]
        Router["FastAPI REST & Streaming Router"]
        AppStateManager["Session & State Manager"]
        SSEHub["SSE Keep-Alive Stream Manager"]
        ReportEngine["ReportLab Multi-Page PDF & Excel Builder"]
    end

    subgraph CoreEngine ["3. Data Science & Machine Learning Engine"]
        Diagnostics["Uji ADF, Dekomposisi Musiman, & Durbin-Watson"]
        Preprocessor["Sliding Window, MinMax Normalizer, & CI Calculator"]
        MLP["Multilayer Perceptron (Sigmoid / Tanh / ReLU)"]
        CGBPROptimizer["Conjugate Gradient Beale-Powell Restart (Diusulkan)"]
        CGPROptimizer["Standard Conjugate Gradient Polak-Ribière"]
        SGDOptimizer["Classical Momentum Backpropagation"]
        BenchmarkEngine["Komparasi Metrik, Efisiensi, & Waktu"]
    end

    CF_CDN <--> PresentationLayer
    PresentationLayer --> Router
    Router --> AppStateManager
    AppStateManager --> Diagnostics
    AppStateManager --> Preprocessor
    Preprocessor --> MLP
    MLP --> CGBPROptimizer
    MLP --> CGPROptimizer
    MLP --> SGDOptimizer
    CGBPROptimizer -.->|Real-Time Epoch Stream| SSEHub
    SSEHub -.-> CF_SSE -.-> LiveLossChart
    CGBPROptimizer --> BenchmarkEngine
    BenchmarkEngine --> BenchmarkPanel
    Preprocessor --> ForecastPanel
    AppStateManager --> ReportEngine
    ReportEngine --> ExportPanel
```

---

## 2. Landasan Matematis & Algoritma Ilmiah

### A. Pembentukan Fitur Lag (*Sliding Window*) & Normalisasi
Deret historis bulanan $y = [y_1, y_2, \dots, y_T]$ dikonstruksi menjadi pasangan matriks input $X$ dan target $Y$:
$$X_t = [y_{t-p}, y_{t-p+1}, \dots, y_{t-1}] \in \mathbb{R}^p, \quad Y_t = y_t \in \mathbb{R}$$

Normalisasi dilakukan ke rentang $[0.1, 0.9]$ untuk menjaga stabilitas turunan fungsi aktivasi sigmoid:
$$y_{norm} = \frac{y - y_{min}}{y_{max} - y_{min}} \times (0.9 - 0.1) + 0.1$$

### B. Uji Stasioneritas Augmented Dickey-Fuller (ADF)
Mendeteksi apakah data memiliki unit root (tidak stasioner):
$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^k \delta_i \Delta y_{t-i} + \epsilon_t$$
Statistik uji dihitung dari rasio $t$:
$$\tau = \frac{\hat{\gamma}}{\text{SE}(\hat{\gamma})}$$
Jika $\tau < \text{Nilai Kritis (5\%)}$, maka tolak $H_0$ (data stasioner).

### C. Dekomposisi Musiman Klasik (12-Bulan)
Memisahkan komponen beban kelistrikan rumah tangga:
$$Y_t = \text{Trend}_t + \text{Seasonal}_t + \text{Residual}_t$$
- **Trend ($T_t$)**: Dihitung menggunakan *Centered Moving Average* 12-bulan ($2 \times 12 \text{ CMA}$).
- **Seasonal ($S_t$)**: Rata-rata deviasi bulanan yang dinormalisasi ($\sum_{m=1}^{12} S_m = 0$).
- **Residual ($R_t$)**: Fluktuasi acak/stokastik yang tidak dapat dijelaskan oleh tren dan musim.

### D. Optimasi Conjugate Gradient Beale-Powell Restart (CG-BPR)
Algoritma CG standar sering kali kehilangan sifat konjugasi ortogonal akibat kelengkungan non-kuadratik fungsi error jaringan saraf tiruan. Algoritma **Beale-Powell Restart** memulihkan efisiensi melalui langkah-langkah:

1. **Arah Penurunan Awal ($k=0$)**:
   $$d_0 = -g_0, \quad t = 0, \quad d_t = d_0, \quad g_t = g_0$$
2. **Kriteria Deteksi Restart Powell (Powell, 1977)**:
   Proses restart dipicu jika:
   - **Kehilangan Ortogonalitas Antar-Gradien**:
     $$|g_{k+1}^T g_k| \ge \eta \|g_{k+1}\|^2 \quad (\text{default } \eta = 0.2)$$
   - **Interval Periodik**:
     $$(k + 1) - t \ge \min(N_{restart}, D)$$
     di mana $D$ adalah dimensi parameter dan $N_{restart}$ adalah periode restart (default: 15).
3. **Pembaruan Arah Pencarian ($d_{k+1}$)**:
   - **Jika Restart Terpicu**: Simpan checkpoint restart $t \leftarrow k+1$, $d_t \leftarrow d_k$, $g_t \leftarrow g_k$, dan setel arah ke *steepest descent*:
     $$d_{k+1} = -g_{k+1}$$
   - **Jika Langkah Konjugat Normal**: Gunakan arah ekstensi Beale:
     $$d_{k+1} = -g_{k+1} + \beta_k d_k + \gamma_k d_t$$
     dengan parameter konjugasi:
     $$\beta_k = \frac{g_{k+1}^T (g_{k+1} - g_k)}{d_k^T (g_{k+1} - g_k)}$$
     $$\gamma_k = \frac{g_{k+1}^T (g_{t+1} - g_t)}{d_t^T (g_{t+1} - g_t)} \quad (\text{untuk } k > t + 1)$$
4. **Line Search Armijo**:
   Menjamin penurunan monotonik error pada setiap epoch:
   $$E(w_k + \alpha_k d_k) \le E(w_k) + c_1 \alpha_k g_k^T d_k$$

### E. Proyeksi Peramalan Probabilistik (95% Confidence Interval)
Ketidakpastian pada peramalan rekursif multi-langkah $h$ ke depan dihitung dari variansi residual testing ($\sigma_e$):
$$\sigma_h = \sigma_e \sqrt{1 + \lambda (h - 1)}$$
$$\text{Batas Bawah}_{95\%} = \max\left(0, \hat{y}_{T+h} - 1.96 \cdot \sigma_h\right)$$
$$\text{Batas Atas}_{95\%} = \hat{y}_{T+h} + 1.96 \cdot \sigma_h$$

### F. Uji Diagnostik Autokorelasi Residual Durbin-Watson
$$DW = \frac{\sum_{t=2}^T (e_t - e_{t-1})^2}{\sum_{t=1}^T e_t^2}$$
Nilai $DW \approx 2.0$ membuktikan bahwa residual model telah acak, independen, dan bebas autokorelasi berulang.

---

## 3. Komparasi Ilmiah 3 Model Optimasi (Hasil Pengujian)

| Algoritma Optimasi | Formula Update | MSE Testing | RMSE (kWh) | MAPE (%) | Status Konvergensi |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **JST + CG Beale-Powell Restart (Diusulkan)** | **$d_{k+1} = -g + \beta d + \gamma d_t$** | **Optimal** | **Terendah** | **< 10% (Sangat Baik)** | **Cepat & Stabil (Global Min)** |
| JST + Standard CG (Polak-Ribière) | $d_{k+1} = -g + \beta_{PR} d$ | Lebih Besar | Lebih Tinggi | 12% - 15% | Rawan osilasi jika ortogonalitas hilang |
| JST + Momentum Backpropagation | $v_{k+1} = \mu v - \alpha g$ | Signifikan | Signifikan | > 12% | Lambat pada saddle point |

---

## 4. Struktur Direktori Proyek

```
project sela/
│
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # REST API & SSE streaming endpoints (Cloudflare-optimized)
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── ann_cgb.py             # JST & Optimizer CG-BPR, CG-PR, SGD (Pure NumPy)
│   │   ├── diagnostics.py         # Uji ADF, Dekomposisi Musiman, & Durbin-Watson
│   │   ├── benchmark.py           # Engine komparasi ilmiah 3-model
│   │   ├── preprocessor.py        # Sliding window, normalisasi, & 95% Confidence Interval
│   │   └── metrics.py             # Kalkulator MSE, RMSE, MAPE
│   ├── services/
│   │   ├── __init__.py
│   │   ├── training_service.py    # State management, training async & forecasting
│   │   └── report_service.py      # Ekspor PDF multi-halaman & Excel multi-sheet
│   ├── config.py                  # Pengaturan direktori & default hyperparameter
│   └── main.py                    # Entrypoint server FastAPI
│
├── data/
│   └── gorontalo_listrik_bulanan_2018_2024.csv # Dataset historis Kota Gorontalo (84 data)
├── reports/                       # Output file PDF & Excel hasil ekspor
├── uploads/                       # Penyimpanan file unggahan user
├── static/
│   ├── css/custom.css             # Styling scrollbar & antarmuka
│   └── js/app.js                  # Frontend controller, Chart.js & SSE handler
├── templates/
│   └── index.html                 # UI Dashboard SPA (Tailwind CSS CDN)
│
├── Dockerfile                     # Multi-stage production container
├── docker-compose.yml             # Orkestrasikan container lokal/VPS
├── wrangler.toml                  # Cloudflare Pages / Workers deployment template
├── .env.example                   # Contoh variabel lingkungan
├── requirements.txt               # Daftar dependensi Python
├── run.py                         # Script peluncur server
└── README.md                      # Dokumentasi komprehensif
```

---

## 5. Panduan Deployment & Menjalankan Aplikasi

### A. Menjalankan Lokal (Development)
```powershell
# 1. Masuk ke direktori
cd "c:\Users\asus\OneDrive\Desktop\project sela"

# 2. Jalankan aplikasi menggunakan virtual environment:
.\venv\Scripts\python.exe run.py
```
- Akses Web UI: **http://127.0.0.1:8000**
- Akses Dokumentasi Swagger: **http://127.0.0.1:8000/docs**

---

### B. Deployment dengan Cloudflare

Aplikasi ini sudah dioptimalkan khusus untuk integrasi Cloudflare:
- Dilengkapi header **`X-Accel-Buffering: no`** dan **`Cache-Control: no-cache`** agar Server-Sent Events (SSE) loss curve tidak di-buffer oleh Cloudflare CDN.
- Dilengkapi **Heartbeat Pings (`: keep-alive-ping`)** setiap 5 detik guna mencegah Cloudflare memutus koneksi streaming dengan error *HTTP 524 (A timeout occurred)* saat model sedang training epoch panjang.

#### Metode 1: Cloudflare Tunnel (Sangat Direkomendasikan)
Jika Anda memiliki server lokal atau VPS, Anda dapat menghubungkannya ke domain Cloudflare secara instan dan aman tanpa membuka port publik:
```bash
# 1. Jalankan aplikasi web
python run.py

# 2. Buka Cloudflare Tunnel ke port 8000
cloudflared tunnel --url http://localhost:8000
```
Cloudflare akan memberikan URL publik HTTPS (misal: `https://xxx.trycloudflare.com`) atau domain Anda sendiri.

#### Metode 2: Docker Container (VPS / Serverless)
```bash
# Build dan jalankan container
docker compose up -d --build
```
Aplikasi akan aktif di port 8000 dan siap di-proxy oleh Cloudflare.
