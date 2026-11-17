import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Application directories
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
REPORT_DIR = BASE_DIR / "reports"
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

for d in [DATA_DIR, UPLOAD_DIR, REPORT_DIR, STATIC_DIR]:
    os.makedirs(d, exist_ok=True)

# Default configuration parameters for JST & CG-BPR
DEFAULT_LAG = 6                  # 6 bulan sebelumnya sebagai input (sliding window)
DEFAULT_HIDDEN_NEURONS = [10]    # Hidden layer: 1 layer dengan 10 neuron
DEFAULT_MAX_EPOCHS = 300
DEFAULT_TARGET_ERROR = 1e-5
DEFAULT_RESTART_INTERVAL = 15    # Periodik restart interval N
DEFAULT_POWELL_THRESHOLD = 0.2   # Orthogonality threshold Powell (|g_k^T g_{k-1}| >= 0.2 ||g_k||^2)
DEFAULT_TEST_SIZE = 0.2          # 80% train, 20% test
