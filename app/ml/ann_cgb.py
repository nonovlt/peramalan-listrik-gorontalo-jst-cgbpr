import numpy as np
from typing import List, Callable, Optional, Dict, Any

class BealePowellANN:
    """
    Artificial Neural Network (Multilayer Perceptron) untuk Regresi Time-Series
    dengan Algoritma Optimasi Conjugate Gradient Beale-Powell Restart (CG-BPR).
    
    Referensi Algoritma:
    - Beale, E. M. L. (1972). A derivation of conjugate gradients.
    - Powell, M. J. D. (1977). Restart procedures for the conjugate gradient method.
    - MATLAB Neural Network Toolbox: 'traincgb' (Conjugate gradient backpropagation with Powell-Beale restarts).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [10],
        output_dim: int = 1,
        max_epochs: int = 300,
        target_error: float = 1e-5,
        restart_interval: int = 15,
        powell_threshold: float = 0.2,
        random_state: int = 42
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.max_epochs = max_epochs
        self.target_error = target_error
        self.restart_interval = restart_interval
        self.powell_threshold = powell_threshold
        self.random_state = random_state

        # Total arsitektur: [input_dim, hidden_1, ..., hidden_m, output_dim]
        self.layer_sizes = [input_dim] + hidden_layers + [output_dim]
        self.num_layers = len(self.layer_sizes)

        # Inisialisasi bobot dan bias
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._init_parameters()

        # Riwayat training
        self.history: Dict[str, List] = {
            "epoch": [],
            "loss": [],
            "gradient_norm": [],
            "restart_events": []
        }

    def _init_parameters(self):
        np.random.seed(self.random_state)
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            # Inisialisasi Xavier/Glorot
            limit = np.sqrt(6.0 / (n_in + n_out))
            W = np.random.uniform(-limit, limit, (n_in, n_out))
            b = np.zeros((1, n_out))
            self.weights.append(W)
            self.biases.append(b)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

    @staticmethod
    def _sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        return a * (1.0 - a)

    def _pack_params(self, weights: List[np.ndarray], biases: List[np.ndarray]) -> np.ndarray:
        """Mengemas seluruh bobot dan bias ke dalam satu vektor 1D theta"""
        parts = []
        for W, b in zip(weights, biases):
            parts.append(W.flatten())
            parts.append(b.flatten())
        return np.concatenate(parts)

    def _unpack_params(self, theta: np.ndarray):
        """Mengekstrak vektor theta 1D kembali ke matriks bobot dan bias"""
        weights = []
        biases = []
        idx = 0
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            
            w_size = n_in * n_out
            W = theta[idx : idx + w_size].reshape((n_in, n_out))
            idx += w_size

            b_size = n_out
            b = theta[idx : idx + b_size].reshape((1, n_out))
            idx += b_size

            weights.append(W)
            biases.append(b)
        return weights, biases

    def _forward(self, X: np.ndarray, weights: List[np.ndarray], biases: List[np.ndarray]):
        """Forward pass melalui jaringan"""
        activations = [X]
        zs = []
        curr = X
        # Hidden layers: Sigmoid activation
        for i in range(len(weights) - 1):
            z = np.dot(curr, weights[i]) + biases[i]
            a = self._sigmoid(z)
            zs.append(z)
            activations.append(a)
            curr = a

        # Output layer: Linear activation untuk regresi
        z_out = np.dot(curr, weights[-1]) + biases[-1]
        a_out = z_out
        zs.append(z_out)
        activations.append(a_out)
        return activations, zs

    def _compute_loss_and_grad(self, theta: np.ndarray, X: np.ndarray, Y: np.ndarray):
        """Menghitung MSE loss dan gradien analitik menggunakan Backpropagation"""
        weights, biases = self._unpack_params(theta)
        N = X.shape[0]

        activations, _ = self._forward(X, weights, biases)
        y_pred = activations[-1]

        # MSE Loss: E = 1/(2N) * sum((y_pred - Y)^2)
        error = y_pred - Y
        loss = 0.5 * np.mean(error ** 2)

        # Backpropagation
        d_weights = [np.zeros_like(W) for W in weights]
        d_biases = [np.zeros_like(b) for b in biases]

        # Gradien layer output (linear activation: f'(z) = 1)
        delta = error / N
        d_weights[-1] = np.dot(activations[-2].T, delta)
        d_biases[-1] = np.sum(delta, axis=0, keepdims=True)

        # Gradien hidden layers
        for layer_idx in range(len(weights) - 2, -1, -1):
            delta = np.dot(delta, weights[layer_idx + 1].T) * self._sigmoid_derivative(activations[layer_idx + 1])
            d_weights[layer_idx] = np.dot(activations[layer_idx].T, delta)
            d_biases[layer_idx] = np.sum(delta, axis=0, keepdims=True)

        grad = self._pack_params(d_weights, d_biases)
        return loss, grad

    def _line_search(
        self,
        theta: np.ndarray,
        d: np.ndarray,
        grad: np.ndarray,
        loss: float,
        X: np.ndarray,
        Y: np.ndarray,
        c1: float = 1e-4,
        rho: float = 0.5,
        max_ls_iters: int = 25
    ) -> float:
        """
        Backtracking Line Search yang memenuhi kondisi Armijo:
        E(theta + alpha * d) <= E(theta) + c1 * alpha * (grad^T d)
        """
        alpha = 1.0
        directional_deriv = np.dot(grad, d)
        
        # Jika bukan descent direction, kembalikan step size kecil
        if directional_deriv >= 0:
            return 1e-4

        for _ in range(max_ls_iters):
            theta_new = theta + alpha * d
            new_loss, _ = self._compute_loss_and_grad(theta_new, X, Y)
            if new_loss <= loss + c1 * alpha * directional_deriv:
                return alpha
            alpha *= rho

        return max(alpha, 1e-6)

    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Melatih model JST menggunakan optimasi Conjugate Gradient Beale-Powell Restart.
        
        Parameter:
            X_train: Matriks fitur input [N x input_dim]
            Y_train: Target aktual [N x output_dim]
            callback: Fungsi callback opsional per epoch (untuk SSE/WebSocket streaming)
        """
        theta = self._pack_params(self.weights, self.biases)
        dim = len(theta)
        
        # Hitung loss dan gradien awal
        loss, grad = self._compute_loss_and_grad(theta, X_train, Y_train)
        grad_norm = np.linalg.norm(grad)

        # Inisialisasi arah pencarian awal: steepest descent d_0 = -g_0
        d = -grad
        d_prev = d.copy()
        grad_prev = grad.copy()

        # Beale-Powell restart tracking
        t = 0                  # Iterasi restart terakhir
        d_t = d.copy()         # Arah pencarian pada iterasi restart
        grad_t = grad.copy()   # Gradien pada iterasi restart
        grad_t_next = grad.copy()

        self.history = {
            "epoch": [0],
            "loss": [float(loss)],
            "gradient_norm": [float(grad_norm)],
            "restart_events": []
        }

        if callback:
            callback({
                "epoch": 0,
                "loss": float(loss),
                "gradient_norm": float(grad_norm),
                "is_restart": True,
                "restart_reason": "Inisialisasi Awal"
            })

        for k in range(self.max_epochs):
            # Kriteria berhenti
            if loss <= self.target_error or grad_norm < 1e-7:
                break

            # Line search untuk menentukan panjang langkah alpha_k
            alpha = self._line_search(theta, d, grad, loss, X_train, Y_train)

            # Update parameter bobot dan bias: theta_{k+1} = theta_k + alpha * d_k
            theta_next = theta + alpha * d
            loss_next, grad_next = self._compute_loss_and_grad(theta_next, X_train, Y_train)
            grad_next_norm = np.linalg.norm(grad_next)

            # Cek Powell Restart Criteria
            # 1. Powell Orthogonality Condition: |g_{k+1}^T g_k| >= 0.2 * ||g_{k+1}||^2
            orthogonality_test = abs(np.dot(grad_next, grad))
            powell_threshold_val = self.powell_threshold * (grad_next_norm ** 2)
            is_orthogonal_lost = orthogonality_test >= powell_threshold_val

            # 2. Interval Periodik: k + 1 - t >= restart_interval atau dimensi bobot
            is_interval_exceeded = (k + 1 - t) >= min(self.restart_interval, dim)

            # Evaluasi kondisi restart
            is_restart = False
            restart_reason = ""

            if is_orthogonal_lost:
                is_restart = True
                restart_reason = "Powell Orthogonality Violation"
            elif is_interval_exceeded:
                is_restart = True
                restart_reason = "Periodic Interval Restart"

            # Tentukan arah pencarian baru d_{k+1}
            if is_restart:
                # RESTART EVENT: Simpan state restart dan reset arah ke steepest descent
                t = k + 1
                d_t = d.copy()
                grad_t = grad.copy()
                grad_t_next = grad_next.copy()
                d_next = -grad_next
                self.history["restart_events"].append({"epoch": k + 1, "reason": restart_reason})
            else:
                # NORMAL STEP: Conjugate update dengan Beale-Powell formula
                delta_g = grad_next - grad
                denom_beta = np.dot(d, delta_g)

                if abs(denom_beta) > 1e-12:
                    beta = np.dot(grad_next, delta_g) / denom_beta
                else:
                    beta = 0.0

                if k + 1 > t + 1 and abs(np.dot(d_t, grad_t_next - grad_t)) > 1e-12:
                    # Beale extension formula: d_{k+1} = -g_{k+1} + beta * d_k + gamma * d_t
                    gamma = np.dot(grad_next, grad_t_next - grad_t) / np.dot(d_t, grad_t_next - grad_t)
                    d_next = -grad_next + beta * d + gamma * d_t
                else:
                    d_next = -grad_next + beta * d

                # Verifikasi arah penurunan (Descent condition: d^T g < 0)
                if np.dot(d_next, grad_next) >= 0:
                    d_next = -grad_next

            # Update variabel untuk iterasi berikutnya
            theta = theta_next
            loss = loss_next
            grad_prev = grad.copy()
            grad = grad_next
            grad_norm = grad_next_norm
            d_prev = d.copy()
            d = d_next

            # Simpan histori
            self.history["epoch"].append(k + 1)
            self.history["loss"].append(float(loss))
            self.history["gradient_norm"].append(float(grad_norm))

            # Trigger callback
            if callback:
                callback({
                    "epoch": k + 1,
                    "loss": float(loss),
                    "gradient_norm": float(grad_norm),
                    "is_restart": is_restart,
                    "restart_reason": restart_reason
                })

        # Simpan bobot final
        self.weights, self.biases = self._unpack_params(theta)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Melakukan inferensi/prediksi data baru"""
        activations, _ = self._forward(X, self.weights, self.biases)
        return activations[-1]
