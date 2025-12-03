import math  # we import math utilities for scale computations
import warnings  # we optionally warn for edge cases
from typing import List, Optional, Sequence, Tuple, Union  # we import typing for annotations

import numpy as np  # we use numpy for vectorized numeric computation

try:
    import pywt  # we use pywt for continuous complex wavelet transforms
except Exception as e:  # we handle optional dependency gracefully
    pywt = None  # we mark pywt as missing
    _PYWT_IMPORT_ERROR = e  # we keep original exception for reporting

try:
    from sklearn.decomposition import KernelPCA  # we use sklearn for kernel pca
    from sklearn.preprocessing import StandardScaler  # we use scaler to standardize features
except Exception as e:  # we handle optional dependency gracefully
    KernelPCA = None  # we mark sklearn as missing
    StandardScaler = None  # we mark sklearn as missing
    _SKLEARN_IMPORT_ERROR = e  # we keep original exception for reporting


class WaveletModel:
    """
    Wavelet-based feature extractor + Kernel PCA embedding for jump time-series.

    This implements Φ(x) as in the paper:
      - Primary complex wavelet coefficients at scales 2^j, j=1..J, evaluated at t=0 (center)
      - Second-order coefficients W_{j2} | W_{j1} x | (0) for all j1 < j2
      - Normalization by scale energy to obtain amplitude invariance
      - Real/imag parts concatenated to a fixed-length feature vector (size 42 for J=6)

    Kernel PCA is then applied to the standardized features to obtain a low-dimensional embedding.

    Parameters
    ----------
    n_layers : int
        Unused (kept only for backward compatibility with existing initialization)  # we keep legacy signature compatibility
    n_neurons : int
        Unused (kept only for backward compatibility with existing initialization)  # we keep legacy signature compatibility
    n_outputs : int
        Unused (kept only for backward compatibility with existing initialization)  # we keep legacy signature compatibility
    J : int, default=6
        Number of dyadic scales (1..J) used to build features  # we define number of scales used
    wavelet : str, default="cmor1.5-1.0"
        PyWavelets complex wavelet name (e.g., "cmorB-C", "cmor1.5-1.0")  # we select complex morlet by default
    kernel : str, default="rbf"
        Kernel name for KernelPCA ("rbf", "poly", "cosine", "sigmoid", "linear")  # we expose kernel name
    n_components : int, default=3
        Number of embedding dimensions to return  # we define embedding dimensionality
    gamma : Union[str, float], default="scale"
        Kernel width param for RBF/poly/sigmoid; "scale"->1/(n_features*var), "auto"->1/n_features  # we implement sklearn-like gamma policy
    degree : int, default=3
        Degree for polynomial kernel  # we expose polynomial degree
    coef0 : float, default=1.0
        Independent term in poly/sigmoid kernels  # we expose coef0
    standardize_features : bool, default=True
        Whether to standardize Φ(x) before KernelPCA  # we recommend standardization
    center_index : Optional[int], default=None
        Index of jump time (t=0) within each time-series; if None uses len(x)//2  # we center at middle by default
    random_state : Optional[int], default=None
        Random seed for reproducibility in KernelPCA  # we expose seed
    """

    def __init__(
        self,
        n_layers: int,
        n_neurons: int,
        n_outputs: int,
        J: int = 6,
        wavelet: str = "cmor1.5-1.0",
        kernel: str = "rbf",
        n_components: int = 3,
        gamma: Union[str, float] = "scale",
        degree: int = 3,
        coef0: float = 1.0,
        standardize_features: bool = True,
        center_index: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.J: int = int(J)  # we store number of scales
        self.wavelet: str = wavelet  # we store wavelet identifier
        self.kernel: str = kernel  # we store kernel name
        self.n_components: int = int(n_components)  # we store embedding dimensionality
        self.gamma: Union[str, float] = gamma  # we store gamma policy or value
        self.degree: int = int(degree)  # we store polynomial degree
        self.coef0: float = float(coef0)  # we store coef0 for kernels
        self.standardize_features: bool = bool(standardize_features)  # we store scaling flag
        self.center_index: Optional[int] = center_index  # we store optional center index
        self.random_state: Optional[int] = random_state  # we store random seed

        self._scaler: Optional[StandardScaler] = None  # we will create the feature scaler on build/fit
        self._kpca: Optional[KernelPCA] = None  # we will create kernel pca on build/fit

        self._train_features_: Optional[np.ndarray] = None  # we keep last computed feature matrix for training set
        self.embedding_: Optional[np.ndarray] = None  # we keep latest training embedding
        self.eigenvalues_: Optional[np.ndarray] = None  # we keep KPCA eigenvalues
        self._feature_names: Optional[List[str]] = None  # we keep cached feature names

    # ---------- public api ----------

    def build_model(self) -> None:
        """Initialize internal scaler and KernelPCA estimator based on current settings."""
        self._ensure_deps()  # we validate optional dependencies availability
        if self.standardize_features and self._scaler is None:  # we lazily instantiate scaler
            self._scaler = StandardScaler(copy=True, with_mean=True, with_std=True)  # we build feature scaler
        self._kpca = KernelPCA(
            n_components=self.n_components,
            kernel=self.kernel,
            gamma=None if isinstance(self.gamma, str) else float(self.gamma),
            degree=self.degree,
            coef0=self.coef0,
            fit_inverse_transform=False,
            eigen_solver="auto",
            random_state=self.random_state,
        )  # we build kernel pca with deferred gamma resolution
        return None  # we return nothing explicitly

    def fit(self, X: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
        """
        Fit the wavelet feature extractor and Kernel PCA on a batch of time-series.

        Parameters
        ----------
        X : array-like of shape (n_samples, T)
            Batch of jump-centered time-series (odd length recommended); center chosen via center_index or mid-point.

        Returns
        -------
        embedding : np.ndarray of shape (n_samples, n_components)
            Low-dimensional kernel PCA embedding of the batch.
        """
        self.build_model()  # we initialize estimators if needed
        X_arr = self._to_2d_array(X)  # we coerce to (n_samples, T)
        feats = self._batch_features(X_arr)  # we compute Φ(X) for all samples
        self._train_features_ = feats  # we cache training features

        feats_scaled = self._scale_fit_transform(feats)  # we optionally standardize features
        self._resolve_kpca_gamma(feats_scaled)  # we resolve gamma if "scale" or "auto"
        embedding = self._kpca.fit_transform(feats_scaled)  # we fit and transform into embedding
        self.embedding_ = embedding  # we cache embedding
        self.eigenvalues_ = getattr(self._kpca, "lambdas_", None)  # we expose eigenvalues
        return embedding  # we return embedding

    def transform(self, X: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
        """
        Transform a batch of time-series into the learned KPCA embedding.

        Parameters
        ----------
        X : array-like of shape (n_samples, T)
            Batch of jump-centered time-series.

        Returns
        -------
        embedding : np.ndarray of shape (n_samples, n_components)
            Low-dimensional kernel PCA embedding.
        """
        self._check_fitted()  # we ensure model was previously fit
        X_arr = self._to_2d_array(X)  # we coerce input to 2D
        feats = self._batch_features(X_arr)  # we compute Φ(X)
        feats_scaled = self._scale_transform(feats)  # we standardize with training stats
        embedding = self._kpca.transform(feats_scaled)  # we project through kernel pca
        return embedding  # we return embedding

    def fit_transform(self, X: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
        """Fit on X and return embedding, equivalent to sequential fit+transform on training data."""
        return self.fit(X)  # we reuse fit to return embedding

    def predict(self, X: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
        """Alias of transform(X) for compatibility with typical API expectations."""
        return self.transform(X)  # we map to transform

    def get_feature_names(self) -> List[str]:
        """Return list of feature names in Φ(x) consistent with compute_features ordering."""
        if self._feature_names is None:  # we lazily build names
            names: List[str] = []  # we initialize list
            # primary wavelet coefficients W_j \bar{x}(0)  # we define names for primary coefficients
            for j in range(1, self.J + 1):
                names.append(f"Re_W_{j}")  # we add real part name
                names.append(f"Im_W_{j}")  # we add imaginary part name
            # second-order scattering coefficients W_{j2} | W_{j1} x | (0) with j1 < j2  # we define names for scattering
            for j2 in range(2, self.J + 1):
                for j1 in range(1, j2):
                    names.append(f"Re_S_{j1}_{j2}")  # we add real part name
                    names.append(f"Im_S_{j1}_{j2}")  # we add imaginary part name
            self._feature_names = names  # we cache names
        return list(self._feature_names)  # we return a copy

    def get_wavelet_function(self, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recover the time-domain wavelet function psi(t) for the configured wavelet.
        
        Returns
        -------
        t : np.ndarray
            Time grid.
        psi : np.ndarray
            Complex wavelet values.
        """
        self._ensure_deps()  # we validate dependencies
        w = pywt.ContinuousWavelet(self.wavelet)  # we create wavelet object
        psi, t = w.wavefun(level=12)  # we evaluate on dense grid
        return t * scale, psi  # we rescale and return

    # ---------- core feature extraction ----------

    def compute_features(self, x: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """
        Compute Φ(x) for a single time-series:
          - W_j \bar{x}(0) normalized, j=1..J (complex -> Re, Im)
          - W_{j2} | W_{j1} x | (0) normalized for all 1 <= j1 < j2 <= J (complex -> Re, Im)

        Parameters
        ----------
        x : array-like of shape (T,)
            Jump-centered time-series, odd length recommended.

        Returns
        -------
        features : np.ndarray of shape (2*(J + J*(J-1)/2),)
            Concatenated real/imag parts of all complex coefficients (size 42 when J=6).
        """
        self._ensure_deps()  # we validate dependencies
        x_arr = np.asarray(x, dtype=float).reshape(-1)  # we coerce to 1D float
        if x_arr.ndim != 1 or x_arr.size < 5:  # we validate minimal length
            raise ValueError("`x` must be 1D with length >= 5")  # we enforce constraints
        T: int = x_arr.size  # we get length
        c0 = self.center_index if self.center_index is not None else (T // 2)  # we choose center index
        if c0 < 0 or c0 >= T:  # we validate center index
            raise ValueError(f"center_index {c0} is out of bounds for length {T}")  # we report invalid index

        sgn = np.sign(x_arr[c0])  # we compute sign at jump time
        if sgn == 0.0:  # we avoid degeneracy when exact zero
            sgn = 1.0  # we default to +1 to keep invariance
        x_aligned = sgn * x_arr  # we build jump-aligned series

        scales = self._dyadic_scales(self.J)  # we build dyadic scales 2^j
        coeffs_x, _ = pywt.cwt(x_aligned, scales, self.wavelet, method="fft")  # we compute complex wavelet transform
        coeffs_x = np.asarray(coeffs_x, dtype=np.complex128)  # we coerce to complex matrix (J, T)

        # primary coefficients W_j \bar{x}(0) normalized by scale energy  # we compute primary normalized coefficients
        feats: List[float] = []  # we initialize features list
        energy_primary = np.sqrt(np.mean(np.abs(coeffs_x) ** 2, axis=1))  # we compute per-scale energy
        for j_idx in range(self.J):
            denom = energy_primary[j_idx] if energy_primary[j_idx] > 0 else 0.0  # we guard divide by zero
            val = coeffs_x[j_idx, c0] / denom if denom > 0 else 0.0  # we normalize at t=0
            feats.append(float(np.real(val)))  # we append real part
            feats.append(float(np.imag(val)))  # we append imaginary part

        # scattering coefficients W_{j2} | W_{j1} x | (0) for all j1 < j2, normalized per (j1,j2)  # we compute second-order coefficients
        for j2_idx in range(1, self.J):
            j2_scale = scales[j2_idx]  # we pick target scale for second order
            for j1_idx in range(0, j2_idx):
                m1 = np.abs(coeffs_x[j1_idx, :])  # we compute modulus |W_{j1} x|(t)
                coeffs_m1_j2, _ = pywt.cwt(m1, [j2_scale], self.wavelet, method="fft")  # we compute wavelet on modulus at scale j2
                coeffs_m1_j2 = np.asarray(coeffs_m1_j2, dtype=np.complex128).reshape(1, T)  # we coerce to shape (1, T)
                energy_sc = float(np.sqrt(np.mean(np.abs(coeffs_m1_j2) ** 2)))  # we compute second-order energy
                denom_sc = energy_sc if energy_sc > 0 else 0.0  # we guard zero
                val_sc = coeffs_m1_j2[0, c0] / denom_sc if denom_sc > 0 else 0.0  # we normalize at t=0
                feats.append(float(np.real(val_sc)))  # we append real part
                feats.append(float(np.imag(val_sc)))  # we append imaginary part

        features = np.asarray(feats, dtype=np.float64)  # we convert features to float array
        return features  # we return Φ(x)

    # ---------- internal utilities ----------

    def _batch_features(self, X: np.ndarray) -> np.ndarray:
        """Compute Φ(x_i) for all rows of X."""
        n, _ = X.shape  # we get shape
        out = np.empty((n, self._feature_dim()), dtype=np.float64)  # we allocate output
        for i in range(n):
            out[i] = self.compute_features(X[i])  # we compute features row-wise
        return out  # we return batch features

    def _feature_dim(self) -> int:
        """Return feature dimension for current J."""
        primary = self.J  # we count primary complex coefficients
        second_order = self.J * (self.J - 1) // 2  # we count j1<j2 complex coefficients
        return 2 * (primary + second_order)  # we return real+imag dimension

    @staticmethod
    def _to_2d_array(X: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
        """Coerce input to 2D float array (n_samples, T)."""
        X_arr = np.asarray(X, dtype=float)  # we coerce to numpy
        if X_arr.ndim == 1:  # we lift single series to batch
            X_arr = X_arr.reshape(1, -1)  # we ensure (1, T)
        if X_arr.ndim != 2:  # we validate dimensionality
            raise ValueError("`X` must be 1D or 2D array-like")  # we enforce shape
        return X_arr  # we return 2D array

    @staticmethod
    def _dyadic_scales(J: int) -> np.ndarray:
        """Return array of dyadic scales [2^1, 2^2, ..., 2^J]."""
        return np.array([2.0**j for j in range(1, J + 1)], dtype=float)  # we build dyadic scales

    def _scale_fit_transform(self, feats: np.ndarray) -> np.ndarray:
        """Fit scaler on feats and return transformed feats or feats if disabled."""
        if not self.standardize_features:  # we skip if not requested
            return feats  # we return original feats
        assert self._scaler is not None  # we ensure scaler exists
        return self._scaler.fit_transform(feats)  # we learn and apply scaling

    def _scale_transform(self, feats: np.ndarray) -> np.ndarray:
        """Transform feats with fitted scaler or pass through if disabled."""
        if not self.standardize_features:  # we skip if not requested
            return feats  # we passthrough
        if self._scaler is None:  # we validate that fit was done
            raise RuntimeError("Scaler is not fitted; call fit() first")  # we guard misuse
        return self._scaler.transform(feats)  # we apply scaling

    def _resolve_kpca_gamma(self, feats_scaled: np.ndarray) -> None:
        """Resolve gamma policy if set to 'scale' or 'auto' and update KPCA estimator."""
        if self._kpca is None:  # we ensure estimator is built
            raise RuntimeError("KernelPCA not initialized; call build_model()")  # we guard misuse
        if isinstance(self.gamma, str):  # we compute gamma if string policy
            n_features = feats_scaled.shape[1]  # we get feature dimension
            if self.gamma == "scale":  # we compute scale policy
                var = float(np.var(feats_scaled)) if np.var(feats_scaled) > 0 else 1.0  # we avoid zero
                gamma_val = 1.0 / (n_features * var)  # we match sklearn-like scale
            elif self.gamma == "auto":  # we compute auto policy
                gamma_val = 1.0 / n_features  # we match sklearn-like auto
            else:
                raise ValueError("gamma must be float, 'scale' or 'auto'")  # we enforce valid gamma
            # rebuild KPCA with resolved gamma while preserving learned state only pre-fit  # we re-instantiate with gamma
            self._kpca = KernelPCA(
                n_components=self.n_components,
                kernel=self.kernel,
                gamma=gamma_val,
                degree=self.degree,
                coef0=self.coef0,
                fit_inverse_transform=False,
                eigen_solver="auto",
                random_state=self.random_state,
            )  # we finalize kpca with numeric gamma

    def _check_fitted(self) -> None:
        """Raise if model was not fitted."""
        if self._kpca is None or (self.standardize_features and self._scaler is None):  # we validate setup
            raise RuntimeError("Model is not fitted; call fit() first")  # we signal improper usage

    @staticmethod
    def _ensure_deps() -> None:
        """Ensure optional dependencies are installed."""
        if pywt is None:  # we validate pywt
            raise ImportError(f"pywt (PyWavelets) is required but not available: {_PYWT_IMPORT_ERROR}")  # we raise informative error
        if KernelPCA is None or StandardScaler is None:  # we validate sklearn
            raise ImportError(f"scikit-learn is required but not available: {_SKLEARN_IMPORT_ERROR}")  # we raise informative error
