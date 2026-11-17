import numpy as np
import time
from typing import List, Callable, Optional, Dict, Any

class BealePowellANN:
    """
    Artificial Neural Network (Multilayer Perceptron) untuk Regresi Time-Series
    Mendukung berbagai algoritma optimasi:
    1. 'cgb'          : Conjugate Gradient Beale-Powell Restart (Metode Utama)
    2. 'cg_pr'        : Standard Conjugate Gradient Polak-Ribière
    3. 'sgd_momentum' : Classical Stochastic/Batch Gradient Descent with Momentum
    
    Aktivasi yang didukung: 'sigmoid', 'tanh', 'relu' (Hidden Layer) dan 'linear' (Output).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [10],
        output_dim: int = 1,
        optimizer: str = "cgb",
        activation: str = "sigmoid",
        max_epochs: int = 300,
        target_error: float = 1e-5,
        restart_interval: int = 15,
        powell_threshold: float = 0.2,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        random_state: int = 42
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.optimizer = optimizer.lower()
        self.activation_name = activation.lower()
        self.max_epochs = max_epochs
        self.target_error = target_error
        self.restart_interval = restart_interval
        self.powell_threshold = powell_threshold
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.random_state = random_state

        # Arsitektur: [input_dim, hidden_1, ..., hidden_m, output_dim]
        self.layer_sizes = [input_dim] + hidden_layers + [output_dim]
        self.num_layers = len(self.layer_sizes)

        # Inisialisasi bobot dan bias
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._init_parameters()

        # Riwayat training & metrik performa
        self.history: Dict[str, Any] = {
            "epoch": [],
            "loss": [],
            "gradient_norm": [],
            "restart_events": [],
            "elapsed_time_sec": 0.0,
            "total_restarts": 0,
            "optimizer": self.optimizer
        }

    def _init_parameters(self):
        np.random.seed(self.random_state)
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]

            if self.activation_name == "relu":
                # He initialization for ReLU
                limit = np.sqrt(2.0 / n_in)
                W = np.random.randn(n_in, n_out) * limit
            else:
                # Xavier / Glorot initialization
                limit = np.sqrt(6.0 / (n_in + n_out))
                W = np.random.uniform(-limit, limit, (n_in, n_out))

            b = np.zeros((1, n_out))
            self.weights.append(W)
            self.biases.append(b)

    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self.activation_name == "tanh":
            return np.tanh(z)
        elif self.activation_name == "relu":
            return np.maximum(0, z)
        else: # default: sigmoid
            return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

    def _activate_derivative(self, a: np.ndarray, z: Optional[np.ndarray] = None) -> np.ndarray:
        if self.activation_name == "tanh":
            return 1.0 - a ** 2
        elif self.activation_name == "relu":
            return (a > 0).astype(float)
        else: # sigmoid
            return a * (1.0 - a)

    def _pack_params(self, weights: List[np.ndarray], biases: List[np.ndarray]) -> np.ndarray:
        parts = []
        for W, b in zip(weights, biases):
            parts.append(W.flatten())
            parts.append(b.flatten())
        return np.concatenate(parts)

    def _unpack_params(self, theta: np.ndarray):
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
        activations = [X]
        zs = []
        curr = X
        # Hidden layers
        for i in range(len(weights) - 1):
            z = np.dot(curr, weights[i]) + biases[i]
            a = self._activate(z)
            zs.append(z)
            activations.append(a)
            curr = a

        # Output layer (Linear activation)
        z_out = np.dot(curr, weights[-1]) + biases[-1]
        a_out = z_out
        zs.append(z_out)
        activations.append(a_out)
        return activations, zs

    def _compute_loss_and_grad(self, theta: np.ndarray, X: np.ndarray, Y: np.ndarray):
        weights, biases = self._unpack_params(theta)
        N = X.shape[0]

        activations, zs = self._forward(X, weights, biases)
        y_pred = activations[-1]

        error = y_pred - Y
        loss = 0.5 * np.mean(error ** 2)

        d_weights = [np.zeros_like(W) for W in weights]
        d_biases = [np.zeros_like(b) for b in biases]

        # Gradien output layer
        delta = error / N
        d_weights[-1] = np.dot(activations[-2].T, delta)
        d_biases[-1] = np.sum(delta, axis=0, keepdims=True)

        # Gradien hidden layers
        for layer_idx in range(len(weights) - 2, -1, -1):
            delta = np.dot(delta, weights[layer_idx + 1].T) * self._activate_derivative(activations[layer_idx + 1], zs[layer_idx])
            d_weights[layer_idx] = np.dot(activations[layer_idx].T, delta)
            d_biases[layer_idx] = np.sum(delta, axis=0, keepdims=True)

        grad = self._pack_params(d_weights, d_biases)
        return loss, grad

    def _line_search_armijo(
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
        """Backtracking Line Search Armijo"""
        alpha = 1.0
        directional_deriv = np.dot(grad, d)
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
        """Melatih jaringan dengan algoritma optimasi yang dipilih"""
        start_time = time.time()
        theta = self._pack_params(self.weights, self.biases)
        dim = len(theta)

        loss, grad = self._compute_loss_and_grad(theta, X_train, Y_train)
        grad_norm = np.linalg.norm(grad)

        d = -grad
        grad_prev = grad.copy()

        # State Beale-Powell
        t = 0
        d_t = d.copy()
        grad_t = grad.copy()
        grad_t_next = grad.copy()

        # State Momentum (untuk SGD)
        velocity = np.zeros_like(theta)

        self.history = {
            "epoch": [0],
            "loss": [float(loss)],
            "gradient_norm": [float(grad_norm)],
            "restart_events": [],
            "elapsed_time_sec": 0.0,
            "total_restarts": 0,
            "optimizer": self.optimizer
        }

        if callback:
            callback({
                "epoch": 0,
                "loss": float(loss),
                "gradient_norm": float(grad_norm),
                "is_restart": True,
                "restart_reason": "Inisialisasi",
                "optimizer": self.optimizer
            })

        for k in range(self.max_epochs):
            if loss <= self.target_error or grad_norm < 1e-7:
                break

            if self.optimizer == "sgd_momentum":
                # Classical Gradient Descent with Momentum
                velocity = self.momentum * velocity - self.learning_rate * grad
                theta = theta + velocity
                loss_next, grad_next = self._compute_loss_and_grad(theta, X_train, Y_train)
                loss = loss_next
                grad = grad_next
                grad_norm = np.linalg.norm(grad)
                is_restart = False
                restart_reason = ""
            else:
                # Conjugate Gradient Methods (cgb atau cg_pr)
                alpha = self._line_search_armijo(theta, d, grad, loss, X_train, Y_train)
                theta_next = theta + alpha * d
                loss_next, grad_next = self._compute_loss_and_grad(theta_next, X_train, Y_train)
                grad_next_norm = np.linalg.norm(grad_next)

                is_restart = False
                restart_reason = ""

                if self.optimizer == "cgb":
                    # Kriteria Powell:
                    # 1. Ortogonalitas: |g_{k+1}^T g_k| >= eta * ||g_{k+1}||^2
                    ortho_test = abs(np.dot(grad_next, grad))
                    if ortho_test >= self.powell_threshold * (grad_next_norm ** 2):
                        is_restart = True
                        restart_reason = "Powell Orthogonality Violation"
                    elif (k + 1 - t) >= min(self.restart_interval, dim):
                        is_restart = True
                        restart_reason = "Periodic Interval Restart"

                    if is_restart:
                        t = k + 1
                        d_t = d.copy()
                        grad_t = grad.copy()
                        grad_t_next = grad_next.copy()
                        d_next = -grad_next
                        self.history["restart_events"].append({"epoch": k + 1, "reason": restart_reason})
                    else:
                        delta_g = grad_next - grad
                        denom_beta = np.dot(d, delta_g)
                        beta = np.dot(grad_next, delta_g) / denom_beta if abs(denom_beta) > 1e-12 else 0.0

                        if k + 1 > t + 1 and abs(np.dot(d_t, grad_t_next - grad_t)) > 1e-12:
                            # Beale extension direction
                            gamma = np.dot(grad_next, grad_t_next - grad_t) / np.dot(d_t, grad_t_next - grad_t)
                            d_next = -grad_next + beta * d + gamma * d_t
                        else:
                            d_next = -grad_next + beta * d

                        if np.dot(d_next, grad_next) >= 0:
                            d_next = -grad_next

                elif self.optimizer == "cg_pr":
                    # Standar Polak-Ribière tanpa Beale Restart
                    delta_g = grad_next - grad
                    denom_beta = np.dot(d, delta_g)
                    beta = max(0.0, np.dot(grad_next, delta_g) / denom_beta) if abs(denom_beta) > 1e-12 else 0.0
                    d_next = -grad_next + beta * d
                    if np.dot(d_next, grad_next) >= 0:
                        d_next = -grad_next

                theta = theta_next
                loss = loss_next
                grad = grad_next
                grad_norm = grad_next_norm
                d = d_next

            self.history["epoch"].append(k + 1)
            self.history["loss"].append(float(loss))
            self.history["gradient_norm"].append(float(grad_norm))

            if callback:
                callback({
                    "epoch": k + 1,
                    "loss": float(loss),
                    "gradient_norm": float(grad_norm),
                    "is_restart": is_restart,
                    "restart_reason": restart_reason,
                    "optimizer": self.optimizer
                })

        self.weights, self.biases = self._unpack_params(theta)
        elapsed = time.time() - start_time
        self.history["elapsed_time_sec"] = round(elapsed, 3)
        self.history["total_restarts"] = len(self.history["restart_events"])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self._forward(X, self.weights, self.biases)
        return activations[-1]
