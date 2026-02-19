import torch
import numpy as np
from scipy.linalg import svd
from typing import Union, Optional
import scipy.stats
import ot  # POT: Python Optimal Transport
from scipy.spatial.distance import cdist

class CCA:
    """
    Canonical Correlation Analysis (CCA) for comparing neural network representations.
    Implementation based on the paper "Similarity of Neural Network Representations Revisited"
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize CCA calculator.
        
        Args:
            epsilon: Small constant for numerical stability
        """
        self.epsilon = epsilon

    def _center_data(self, X: np.ndarray) -> np.ndarray:
        """Center the data by removing the mean."""
        return X - X.mean(axis=0, keepdims=True)

    def compute_similarity(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor],
        return_correlations: bool = False
    ) -> Union[float, tuple]:
        """
        Compute CCA similarity between two representation matrices.
        
        Args:
            X: First representation matrix (n_samples, n_features_1)
            Y: Second representation matrix (n_samples, n_features_2)
            return_correlations: If True, return individual correlations as well
            
        Returns:
            float or tuple: CCA similarity score, and optionally the canonical correlations
        """
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().cpu().numpy()
            
        # Flatten if needed
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        if Y.ndim > 2:
            Y = Y.reshape(Y.shape[0], -1)

        # Center the representations (vectorized)
        X = self._center_data(X)
        Y = self._center_data(Y)
        
        # Compute QR decomposition of X and Y
        # Note: QR decomposition is already optimized in numpy/scipy
        Qx, Rx = np.linalg.qr(X)
        Qy, Ry = np.linalg.qr(Y)
        
        # Compute SVD of Qx.T @ Qy (vectorized matrix multiplication)
        try:
            cross_cov = Qx.T @ Qy  # Efficient matrix multiplication
            U, correlations, Vh = svd(cross_cov)
            # Clip to handle numerical errors
            correlations = np.clip(correlations, 0, 1)
            
            # Take only the minimum number of correlations
            K = min(X.shape[1], Y.shape[1])
            correlations = correlations[:K]
            
            mean_correlation = np.mean(correlations)
        except np.linalg.LinAlgError:
            print("Warning: SVD computation failed, returning zeros")
            K = min(X.shape[1], Y.shape[1])
            correlations = np.zeros(K)
            mean_correlation = 0.0
        
        if return_correlations:
            return mean_correlation, correlations
        return mean_correlation

    def compute_similarity_batch(
        self,
        model1: torch.nn.Module,
        model2: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        layer_name: str,
        max_batches: Optional[int] = None
    ) -> float:
        """
        Compute CCA similarity between two models' representations on a dataset.
        
        Args:
            model1: First model
            model2: Second model
            dataloader: DataLoader for input data
            device: Device to run computation on
            layer_name: Name of layer to extract features from
            max_batches: Maximum number of batches to process
            
        Returns:
            float: CCA similarity score
        """
        features1 = []
        features2 = []
        
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                if max_batches and i >= max_batches:
                    break
                    
                inputs = inputs.to(device)
                
                # Get features from both models
                feat1 = model1(inputs)
                feat2 = model2(inputs)
                
                features1.append(feat1.cpu())
                features2.append(feat2.cpu())
        
        # Concatenate all features
        features1 = torch.cat(features1, dim=0)
        features2 = torch.cat(features2, dim=0)
        
        return self.compute_similarity(features1, features2)

class CKA:
    """
    Centered Kernel Alignment (CKA) for comparing neural network representations.
    Implementation based on "Similarity of Neural Network Representations Revisited"
    """
    
    def __init__(self, kernel: str = 'linear'):
        """
        Initialize CKA calculator.
        
        Args:
            kernel: Type of kernel to use ('linear' or 'rbf')
        """
        self.kernel = kernel

    def _center_gram(self, K: np.ndarray) -> np.ndarray:
        """Center the Gram matrix."""
        means = K.mean(axis=0, keepdims=True)
        means_T = means.T
        global_mean = means.mean()
        return K - means - means_T + global_mean

    def _linear_kernel(self, X: np.ndarray) -> np.ndarray:
        """Compute the linear kernel."""
        return X @ X.T

    def _rbf_kernel(self, X: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """Compute the RBF (Gaussian) kernel using vectorized operations."""
        n_samples = X.shape[0]
        
        # Compute sigma if not provided
        if sigma is None:
            # Use a deterministic subset for sigma estimation
            max_subset = min(1000, n_samples)
            # Use a fixed random state for reproducibility
            rng = np.random.RandomState(42)
            subset_idx = rng.choice(n_samples, size=max_subset, replace=False)
            X_subset = X[subset_idx]
            
            # Vectorized computation of squared distances
            X_subset_squared = np.sum(X_subset**2, axis=1, keepdims=True)
            subset_sq_dists = X_subset_squared + X_subset_squared.T - 2 * X_subset @ X_subset.T
            
            # Ensure numerical stability
            subset_sq_dists = np.maximum(subset_sq_dists, 0)
            sigma = np.median(subset_sq_dists[subset_sq_dists > 0])
            if sigma == 0:
                sigma = 1.0
        
        # Vectorized computation of the full kernel matrix
        # Use the ||x-y||^2 = ||x||^2 + ||y||^2 - 2x·y trick for efficiency
        X_squared_norms = np.sum(X**2, axis=1, keepdims=True)
        sq_dists = X_squared_norms + X_squared_norms.T - 2 * X @ X.T
        
        # Ensure numerical stability
        sq_dists = np.maximum(sq_dists, 0)
        
        # Apply RBF kernel
        K = np.exp(-sq_dists / (2 * sigma))
        
        return K

    def compute_similarity(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Compute CKA similarity between two representation matrices.
        
        Args:
            X: First representation matrix (n_samples, n_features_1)
            Y: Second representation matrix (n_samples, n_features_2)
            
        Returns:
            float: CKA similarity score
        """
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().cpu().numpy()
            
        # Flatten if needed
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        if Y.ndim > 2:
            Y = Y.reshape(Y.shape[0], -1)

        # Compute kernel matrices (vectorized)
        if self.kernel == 'linear':
            K = self._linear_kernel(X)
            L = self._linear_kernel(Y)
        else:  # rbf kernel
            K = self._rbf_kernel(X)
            L = self._rbf_kernel(Y)

        # Center kernel matrices (vectorized)
        K = self._center_gram(K)
        L = self._center_gram(L)

        # Compute HSIC (Hilbert-Schmidt Independence Criterion) - vectorized
        HSIC = np.sum(K * L)
        
        # Normalize (vectorized)
        normalization = np.sqrt(np.sum(K * K) * np.sum(L * L))
        
        if normalization > 0:
            CKA = HSIC / normalization
        else:
            CKA = 0.0

        return CKA

class EuclideanDistance:
    """
    Euclidean Distance calculator for comparing neural network representations.
    Lower values indicate more similar representations.
    """
    
    def compute_similarity(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Compute average Euclidean distance between two representation matrices.
        
        Args:
            X: First representation matrix (n_samples, n_features_1) or vector (n_features_1,)
            Y: Second representation matrix (n_samples, n_features_2) or vector (n_features_2,)
            
        Returns:
            float: Average Euclidean distance (normalized)
        """
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().cpu().numpy()
            
        # Ensure inputs are 2D arrays
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
            
        # Flatten if needed (for higher dimensions)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        if Y.ndim > 2:
            Y = Y.reshape(Y.shape[0], -1)

        # Ensure X and Y have the same number of features
        if X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y must have the same number of features")

        # Normalize the representations
        X_norm = np.linalg.norm(X, axis=1, keepdims=True)
        Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
        
        # Avoid division by zero
        X_normalized = X / (X_norm + 1e-8)
        Y_normalized = Y / (Y_norm + 1e-8)

        # Compute Euclidean distance
        distances = np.linalg.norm(X_normalized - Y_normalized, axis=1)
        
        # Return average distance
        return float(np.mean(distances))

class CosineSimilarity:
    """
    Cosine Similarity calculator for comparing neural network representations.
    Values closer to 1 indicate more similar representations.
    """
    
    def compute_similarity(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Compute average cosine similarity between two representation matrices.
        
        Args:
            X: First representation matrix (n_samples, n_features_1) or vector (n_features_1,)
            Y: Second representation matrix (n_samples, n_features_2) or vector (n_features_2,)
            
        Returns:
            float: Average cosine similarity
        """
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().cpu().numpy()
            
        # Ensure inputs are 2D arrays
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
            
        # Flatten if needed (for higher dimensions)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        if Y.ndim > 2:
            Y = Y.reshape(Y.shape[0], -1)

        # Ensure X and Y have the same number of features
        if X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y must have the same number of features")

        # Compute cosine similarity for each sample
        X_norm = np.linalg.norm(X, axis=1, keepdims=True)
        Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
        
        # Avoid division by zero
        X_normalized = X / (X_norm + 1e-8)
        Y_normalized = Y / (Y_norm + 1e-8)
        
        # Compute cosine similarities
        similarities = np.sum(X_normalized * Y_normalized, axis=1)
        
        # Return average similarity
        return float(np.mean(similarities))

class EarthMoversDistance:
    """
    Earth Mover's Distance (Wasserstein distance) for comparing neural network representations.
    Lower values indicate more similar representations.
    """
    def compute_similarity(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Compute average EMD (Wasserstein distance) between two representation matrices.
        Args:
            X: First representation matrix (n_samples, n_features)
            Y: Second representation matrix (n_samples, n_features)
        Returns:
            float: Average EMD across features
        """
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().cpu().numpy()
        # Flatten if needed
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        if Y.ndim > 2:
            Y = Y.reshape(Y.shape[0], -1)
        # Ensure same number of features
        if X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y must have the same number of features")
        # Compute EMD for each feature dimension
        emd_per_feature = [
            scipy.stats.wasserstein_distance(X[:, i], Y[:, i])
            for i in range(X.shape[1])
        ]
        return float(np.mean(emd_per_feature))

class GromovWassersteinDistance:
    """
    Computes the (entropic-regularised) squared Gromov-Wasserstein distance
    between two point clouds, then converts it to a similarity score.
    """

    def __init__(self,
                 intra_metric: str = "cosine",
                 loss_fun: str = "square_loss",
                 reg: float = 5e-4,
                 max_iter: int = 30,
                 tol: float = 1e-9,
                 to_similarity=lambda d: 1.0 / (1.0 + d)):
        self.intra_metric = intra_metric
        self.loss_fun = loss_fun
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol
        self._to_sim = to_similarity     # maps distance → similarity

    def _cost_matrices(self, X, Y):
        Cx = cdist(X, X, metric=self.intra_metric).astype(np.float64)
        Cy = cdist(Y, Y, metric=self.intra_metric).astype(np.float64)
        Cx /= np.median(Cx)
        Cy /= np.median(Cy)
        return Cx, Cy

    def compute_similarity(self, X: np.ndarray, Y: np.ndarray) -> float:
        Cx, Cy = self._cost_matrices(X, Y)
        p = np.ones(Cx.shape[0]) / Cx.shape[0]
        q = np.ones(Cy.shape[0]) / Cy.shape[0]

        # squared GW distance
        gw_dist = ot.gromov.gromov_wasserstein2(
            Cx, Cy, p, q,
            loss_fun=self.loss_fun,
            epsilon=self.reg,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=False,
            log=False
        )
        return self._to_sim(gw_dist)  # Use the single return value directly

class KLDivergence:
    """
    Kullback-Leibler Divergence for comparing neural network representations.
    Treats representations as probability distributions and measures their divergence.
    Lower values indicate more similar representations.
    """
    
    def __init__(self, epsilon: float = 1e-8, symmetric: bool = True):
        """
        Initialize KL divergence calculator.
        
        Args:
            epsilon: Small constant added for numerical stability
            symmetric: If True, compute symmetric KL divergence (Jensen-Shannon style)
        """
        self.epsilon = epsilon
        self.symmetric = symmetric
    
    def _to_probability_distribution(self, X: np.ndarray) -> np.ndarray:
        """
        Convert representation matrix to probability distributions.
        Each row becomes a probability distribution over features.
        """
        # Ensure non-negative values by taking absolute value
        X_abs = np.abs(X)
        
        # Add small epsilon to avoid zeros
        X_stable = X_abs + self.epsilon
        
        # Normalize each row to sum to 1 (making them probability distributions)
        X_prob = X_stable / np.sum(X_stable, axis=1, keepdims=True)
        
        return X_prob
    
    def _kl_divergence(self, P: np.ndarray, Q: np.ndarray) -> float:
        """
        Compute KL divergence between two probability distributions.
        KL(P||Q) = sum(P * log(P/Q))
        """
        # Ensure Q doesn't have zeros by adding epsilon
        Q_stable = Q + self.epsilon
        
        # Compute KL divergence
        kl = np.sum(P * np.log(P / Q_stable), axis=1)
        
        return np.mean(kl)
    
    def compute_similarity(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Compute KL divergence between two representation matrices.
        
        Args:
            X: First representation matrix (n_samples, n_features)
            Y: Second representation matrix (n_samples, n_features)
            
        Returns:
            float: KL divergence (lower values indicate more similarity)
        """
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().cpu().numpy()
            
        # Ensure inputs are 2D arrays
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
            
        # Flatten if needed (for higher dimensions)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        if Y.ndim > 2:
            Y = Y.reshape(Y.shape[0], -1)

        # Ensure X and Y have the same number of features
        if X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y must have the same number of features")
        
        # Ensure same number of samples for pairwise comparison
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples")
        
        # Convert to probability distributions
        P = self._to_probability_distribution(X)
        Q = self._to_probability_distribution(Y)
        
        if self.symmetric:
            # Compute symmetric KL divergence: 0.5 * (KL(P||Q) + KL(Q||P))
            kl_pq = self._kl_divergence(P, Q)
            kl_qp = self._kl_divergence(Q, P)
            return 0.5 * (kl_pq + kl_qp)
        else:
            # Compute standard KL divergence: KL(P||Q)
            return self._kl_divergence(P, Q)
