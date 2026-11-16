"""Continuous frequency nadir regression models."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


class BaseNadirRegressor(BaseEstimator, RegressorMixin):
    """Base estimator for frequency nadir regressors."""

    def __init__(
        self,
        clip_bounds: Optional[Tuple[float, float]] = (45.0, 50.5),
        scale_features: bool = False,
    ) -> None:
        self.clip_bounds = clip_bounds
        self.scale_features = scale_features
        self.model_name = "base"
        self.model_: Any = None
        self.scaler_: Optional[StandardScaler] = None
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[List[str]] = None

    def _prepare_X(self, X: Any, is_fit: bool = False) -> np.ndarray:
        if X is None:
            if is_fit:
                raise ValueError("Cannot fit on None dataset")
            return np.empty((0, self.n_features_in_ or 0), dtype=float)

        if hasattr(X, "shape") and len(X.shape) > 0 and X.shape[0] == 0:
            if is_fit:
                raise ValueError("Cannot fit on empty dataset")
            return np.empty((0, self.n_features_in_ or 0), dtype=float)

        if isinstance(X, (list, tuple)) and len(X) == 0:
            if is_fit:
                raise ValueError("Cannot fit on empty dataset")
            return np.empty((0, self.n_features_in_ or 0), dtype=float)

        if hasattr(X, "columns"):
            cols = list(X.columns)
            if is_fit:
                self.feature_names_in_ = cols
            X_arr = np.asarray(X.values, dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)

        if X_arr.ndim == 1:
            if is_fit:
                X_arr = X_arr.reshape(-1, 1)
            else:
                if self.n_features_in_ is not None and len(X_arr) == self.n_features_in_:
                    X_arr = X_arr.reshape(1, -1)
                else:
                    X_arr = X_arr.reshape(-1, 1)

        if is_fit:
            if X_arr.shape[0] == 0:
                raise ValueError("Cannot fit on empty dataset")
            self.n_features_in_ = X_arr.shape[1]
        else:
            if self.n_features_in_ is not None and X_arr.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Feature dimension mismatch: expected {self.n_features_in_}, got {X_arr.shape[1]}"
                )

        return X_arr

    def _prepare_y(self, y: Any) -> np.ndarray:
        if y is None:
            raise ValueError("Target vector y cannot be None")
        if hasattr(y, "values"):
            y_arr = np.asarray(y.values, dtype=float)
        else:
            y_arr = np.asarray(y, dtype=float)

        y_flat = y_arr.ravel()
        if len(y_flat) == 0:
            raise ValueError("Cannot fit on empty target vector")
        return y_flat

    def fit(self, X: Any, y: Any) -> BaseNadirRegressor:
        """Fit regressor on training data."""
        X_proc = self._prepare_X(X, is_fit=True)
        y_proc = self._prepare_y(y)

        if len(X_proc) != len(y_proc):
            raise ValueError(
                f"Sample count mismatch: X has {len(X_proc)} samples, y has {len(y_proc)}"
            )

        # Shift default 50 Hz clipping bounds when fitting on 60 Hz fixtures
        if self.clip_bounds == (45.0, 50.5) and len(y_proc) > 0 and float(np.median(y_proc)) > 54.0:
            self._effective_clip_bounds_: Optional[Tuple[float, float]] = (55.0, 60.5)
        else:
            self._effective_clip_bounds_ = self.clip_bounds

        if self.scale_features:
            self.scaler_ = StandardScaler()
            X_proc = self.scaler_.fit_transform(X_proc)

        self._fit_underlying_model(X_proc, y_proc)
        return self

    def _fit_underlying_model(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, X: Any) -> np.ndarray:
        """Predict frequency nadir in Hz, clipped to physical bounds."""
        if self.model_ is None:
            raise RuntimeError("Model is not fitted")

        X_proc = self._prepare_X(X, is_fit=False)
        if len(X_proc) == 0:
            return np.array([], dtype=float)

        if self.scale_features and self.scaler_ is not None:
            X_proc = self.scaler_.transform(X_proc)

        preds = np.asarray(self.model_.predict(X_proc), dtype=float).ravel()

        eff_bounds = getattr(self, "_effective_clip_bounds_", self.clip_bounds)
        if eff_bounds is not None:
            low, high = eff_bounds
            preds = np.clip(preds, low, high)

        return preds

    def evaluate(self, X: Any, y: Any) -> Dict[str, float]:
        """Compute MAE, RMSE, and R2 on (X, y)."""
        y_true = self._prepare_y(y)
        y_pred = self.predict(X)

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Sample count mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}"
            )

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(math.sqrt(float(np.mean((y_true - y_pred) ** 2))))

        y_var = float(np.var(y_true))
        if y_var < 1e-12:
            r2 = 1.0 if np.allclose(y_true, y_pred, atol=1e-4) else 0.0
        else:
            r2 = float(r2_score(y_true, y_pred))

        return {"mae": mae, "rmse": rmse, "r2": r2}


class LinearNadirRegressor(BaseNadirRegressor):
    """Ridge regression baseline with feature scaling for nadir prediction."""

    def __init__(
        self,
        alpha: float = 1e-3,
        fit_intercept: bool = True,
        clip_bounds: Optional[Tuple[float, float]] = (45.0, 50.5),
        scale_features: bool = True,
    ) -> None:
        super().__init__(clip_bounds=clip_bounds, scale_features=scale_features)
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.model_name = "linear_ridge"

    def _fit_underlying_model(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model_ = Ridge(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            solver="auto",
        )
        self.model_.fit(X, y)

    @property
    def weights(self) -> Optional[np.ndarray]:
        return self.model_.coef_ if self.model_ is not None else None

    @property
    def intercept(self) -> Optional[float]:
        return float(self.model_.intercept_) if self.model_ is not None else None


class DecisionTreeNadirRegressor(BaseNadirRegressor):
    """CART decision tree regressor for nadir prediction."""

    def __init__(
        self,
        max_depth: Optional[int] = 6,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = 42,
        clip_bounds: Optional[Tuple[float, float]] = (45.0, 50.5),
        scale_features: bool = False,
    ) -> None:
        super().__init__(clip_bounds=clip_bounds, scale_features=scale_features)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model_name = "decision_tree"

    def _fit_underlying_model(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model_ = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.model_.fit(X, y)


class RandomForestNadirRegressor(BaseNadirRegressor):
    """Random forest regressor for nadir prediction."""

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: Optional[int] = 8,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = 42,
        clip_bounds: Optional[Tuple[float, float]] = (45.0, 50.5),
        scale_features: bool = False,
        n_jobs: Optional[int] = None,
    ) -> None:
        super().__init__(clip_bounds=clip_bounds, scale_features=scale_features)
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_name = "random_forest"

    def _fit_underlying_model(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X, y)


class GradientBoostingNadirRegressor(BaseNadirRegressor):
    """Gradient boosted decision tree regressor for nadir prediction."""

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: Optional[int] = 42,
        clip_bounds: Optional[Tuple[float, float]] = (45.0, 50.5),
        scale_features: bool = False,
    ) -> None:
        super().__init__(clip_bounds=clip_bounds, scale_features=scale_features)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = float(subsample)
        self.random_state = random_state
        self.model_name = "gradient_boosting"

    def _fit_underlying_model(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model_ = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        self.model_.fit(X, y)


REGRESSOR_REGISTRY: Dict[str, Type[BaseNadirRegressor]] = {
    "linear": LinearNadirRegressor,
    "ridge": LinearNadirRegressor,
    "linear_nadir": LinearNadirRegressor,
    "decision_tree": DecisionTreeNadirRegressor,
    "cart": DecisionTreeNadirRegressor,
    "dt": DecisionTreeNadirRegressor,
    "random_forest": RandomForestNadirRegressor,
    "rf": RandomForestNadirRegressor,
    "gradient_boosting": GradientBoostingNadirRegressor,
    "gbdt": GradientBoostingNadirRegressor,
    "gbr": GradientBoostingNadirRegressor,
}


def get_nadir_regressor(name: str = "gradient_boosting", **kwargs: Any) -> BaseNadirRegressor:
    """Instantiate a nadir regressor by name."""
    key = name.strip().lower()
    if key not in REGRESSOR_REGISTRY:
        raise ValueError(
            f"Unknown regressor name: '{name}'. Available models: {sorted(list(REGRESSOR_REGISTRY.keys()))}"
        )
    return REGRESSOR_REGISTRY[key](**kwargs)


class NadirRegressor:
    """Wrapper around registered nadir regression models."""

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        clip_bounds: Optional[Tuple[float, float]] = (45.0, 50.5),
        scale_features: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        init_kwargs = dict(kwargs)
        if clip_bounds is not None:
            init_kwargs["clip_bounds"] = clip_bounds
        if scale_features is not None:
            init_kwargs["scale_features"] = scale_features
        self.model_ = get_nadir_regressor(model_type, **init_kwargs)

    def fit(self, X: Any, y: Any) -> NadirRegressor:
        self.model_.fit(X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return self.model_.predict(X)

    def evaluate(self, X: Any, y: Any) -> Dict[str, float]:
        return self.model_.evaluate(X, y)
