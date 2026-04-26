from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from typing import Optional, Iterable
from tqdm.auto import tqdm

from sklearn.base import ClassifierMixin


class Boosting(ClassifierMixin):

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 20,
        learning_rate: float = 0.05,
        random_state: int | None = None,
        verbose: bool = False,
        cat_features: Iterable | None = None,
        early_stopping_rounds: int | None = 0,
        eval_metric: str | None = None,
        subsample: float = 1.0,
        bagging_temperature: float = 1.0,
        bootstrap_type: str | None = 'Bernoulli',
        rsm: float = 1.0,
        goss: bool=False,
        goss_k: float=0.2,
        quantization_type: str | None=None, 
        nbins: int=255,
        dart: bool = False,
        dropout_rate: float = 0.05
    ):
        super().__init__()

        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.models = []
        self.gammas = []

        self.random_state = random_state  # не забудьте вставить его везде, где у вас возникает рандом
        self.verbose = verbose
        self.classes_ = np.array([-1, 1])  # в нашей задаче классы захардкожены
        
        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y / (1 + np.exp(y * z))  # Исправлено
        
        # ранняя остановка
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        
        # категории
        self.cat_features = list(cat_features) if cat_features is not None else []
        self.cat_encoder = None
        
        # бутстрап
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature 
        self.bootstrap_type = bootstrap_type
        
        if not (0 < self.subsample <= 1):
            raise ValueError("subsample must be in (0, 1]")

        if self.bagging_temperature < 0:
            raise ValueError("bagging_temperature must be >= 0")
        
        self._rng = np.random.default_rng(self.random_state)
        
        # rsm
        self.rsm = rsm 
        self._features_idx = []
        
        # goss
        self.goss = goss
        self.goss_k = goss_k
        
        # бинаризация
        self.quantization_type = quantization_type
        self.nbins = nbins
        self.quantizer = None
        
        # dart
        self.dart = dart
        self.dropout_rate = dropout_rate

    def partial_fit(self, X: np.ndarray, y: np.ndarray, train_predictions):
        residuals = -self.loss_derivative(y, train_predictions) # считаем сдвиги
        
        d = X.shape[1] # все признаки
        # rsm
        f = max(1, int(np.ceil(self.rsm * d))) # сколько признаков выбираем
        feature_idx = self._rng.choice(d, size=f, replace=False) # рандомно выбираем индексы для признаков
        
        if self.goss:
            abs_residuals = np.abs(residuals) # считаем абсолютные сдвиги
            n = X.shape[0] # число объектов
            k = max(1, int(np.ceil(self.goss_k * n))) # число важных объектов, на которых большие сдвиги
            
            top_idx = np.argpartition(abs_residuals, -k)[-k:]   # индексы k самых больших по модулю
            #top_idx = np.argsort(abs_residuals)[-k:]
            
            mask_top = np.zeros(n, dtype=bool)
            mask_top[top_idx] = True # взяли самые большие по модулю отклонения
            
            small_idx = np.flatnonzero(~mask_top)
            
            keep_small = self._rng.random(small_idx.shape[0]) < self.subsample
            small_keep_idx = small_idx[keep_small]
            
            idx = np.concatenate([top_idx, small_keep_idx])
            
            X_fit = X[idx]
            residuals_fit = residuals[idx]
            sample_weight = np.ones(idx.shape[0], dtype=float)
            if small_keep_idx.size > 0:
                sample_weight[top_idx.size:] = (1.0 - self.goss_k) / self.subsample
                
            self.history["goss_n_used"].append(len(idx))
            self.history["goss_top_k"].append(k)
        
        else:
            if self.bootstrap_type is None:
                X_fit = X
                residuals_fit = residuals
                sample_weight = None
            elif self.bootstrap_type == 'Bernoulli':
                mask = self._rng.random(X.shape[0]) < self.subsample
                if not mask.any():
                    idx = self._rng.integers(0, X.shape[0])
                    mask[idx] = True
                X_fit = X[mask]
                residuals_fit = residuals[mask]
                sample_weight = None
            elif self.bootstrap_type == 'Bayesian':
                U = self._rng.random(X.shape[0])
                w = (-np.log(U)) ** self.bagging_temperature
                X_fit = X
                residuals_fit = residuals
                sample_weight = w
            
            else:
                raise ValueError("Unknown bootstrap_type")
        
        params = dict(self.base_model_params)
        if self.random_state is not None:
            params["random_state"] = self.random_state
        model = self.base_model_class(**params) # берем базовую модель
        model.fit(X_fit[:, feature_idx], residuals_fit, sample_weight=sample_weight) # обучаемся на эти сдвиги
        new_pred = model.predict(X[:, feature_idx]) # получаем предсказания
        gamma = self.find_optimal_gamma(y, train_predictions, new_pred)
        train_predictions += new_pred * gamma * self.learning_rate # добавляем к предсказаниям
        
        # добавляем модель к ансамблю и gamma
        self.models.append(model)
        self.gammas.append(gamma)
        self._features_idx.append(feature_idx)
        
        #self.history['train_loss'].append(self.loss_fn(y, train_predictions))
        #self.history['train_roc_auc'].append(roc_auc_score(y==1, self.sigmoid(train_predictions)))
        
        return train_predictions, model, gamma, feature_idx

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, eval_set: tuple[np.ndarray, np.ndarray] | None = None,
            use_best_model: bool = False):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        """
        X_tr = X_train
        
        self.models = []
        self.gammas = []
        self._features_idx = []
        self.history = defaultdict(list)
        
        if self.cat_features:
            self.cat_encoder = CatFeaturesEncoder(self.cat_features)
            X_tr = self.cat_encoder.fit_transform(X_train, y_train)

        if eval_set is not None:
           X_val, y_val = eval_set
           if self.cat_encoder is not None:
               X_val = self.cat_encoder.transform(X_val) 
               
        if self.quantization_type is not None:
            self.quantizer = Quantizator(self.quantization_type, self.nbins)
            X_tr = self.quantizer.fit_transform(X_tr)
            if eval_set is not None:
                X_val = self.quantizer.transform(X_val)
                
        self.n_features_in_ = X_tr.shape[1]
        
        train_predictions = np.zeros(X_tr.shape[0])
        
        if eval_set is not None:
            val_predictions = np.zeros(X_val.shape[0])
        
        if self.eval_metric is not None:
            metric_key = self.eval_metric
        else:
            metric_key = 'val_roc_auc' if eval_set is not None else 'train_roc_auc' # дефолтная метрика
        
        if eval_set is None and metric_key.startswith("val_"):
                raise ValueError("eval_metric starts with 'val_' but eval_set is None")
        
        minimize = "loss" in metric_key
        
        best_iter = -1
        best_score = np.inf if minimize else -np.inf
        no_improve_rounds = 0          
            
        estimator_range = range(self.n_estimators)
        if self.verbose:
            estimator_range = tqdm(estimator_range)

        for _ in estimator_range: # прогон по числу моделей в ансамбле
            if self.dart and self.dropout_rate > 0 and len(self.models) > 0:
                m = len(self.models)
                
                drop_mask = self._rng.random(m) < self.dropout_rate
                drop_idx = np.flatnonzero(drop_mask)
                
                if drop_idx.size == 0:
                    drop_idx = np.array([self._rng.integers(0, m)])
                    
                k = int(drop_idx.size)
                
                scale_drop = k / (k + 1)
                scale_new = 1 / k
                
                S_drop_tr = np.zeros_like(train_predictions)
                for idx in drop_idx:
                    fi = self._features_idx[idx]
                    S_drop_tr += self.learning_rate * self.gammas[idx] * self.models[idx].predict(X_tr[:, fi])
                
                base_tr = train_predictions - S_drop_tr
                
                if eval_set is not None:
                    S_drop_val = np.zeros_like(val_predictions)
                    for idx in drop_idx:
                        fi = self._features_idx[idx]
                        S_drop_val += self.learning_rate * self.gammas[idx] * self.models[idx].predict(X_val[:, fi])
                    base_val = val_predictions - S_drop_val
                    
                tmp_tr, model, gamma, feature_idx = self.partial_fit(X_tr, y_train, base_tr.copy())
                C_new_tr = tmp_tr - base_tr  # = lr * gamma * pred_new
                
                if eval_set is not None:
                    C_new_val = self.learning_rate * gamma * model.predict(X_val[:, feature_idx])
                    tmp_val = base_val + C_new_val
                    
                
                for idx in drop_idx:
                    self.gammas[idx] *= scale_drop
                self.gammas[-1] *= scale_new  # новое дерево в конце списка

                train_predictions = tmp_tr + scale_drop * S_drop_tr + (scale_new - 1.0) * C_new_tr

                if eval_set is not None:
                    val_predictions = tmp_val + scale_drop * S_drop_val + (scale_new - 1.0) * C_new_val
                       
            else:
                train_predictions, model, gamma, feature_idx = self.partial_fit(X_tr, y_train, train_predictions)
                if eval_set is not None:
                    val_predictions += model.predict(X_val[:, feature_idx]) * gamma * self.learning_rate
            
            self.history['train_loss'].append(self.loss_fn(y_train, train_predictions))
            self.history['train_roc_auc'].append(roc_auc_score(y_train == 1, self.sigmoid(train_predictions)))
            
            if eval_set is not None:
                self.history['val_loss'].append(self.loss_fn(y_val, val_predictions))
                self.history['val_roc_auc'].append(roc_auc_score(y_val == 1, self.sigmoid(val_predictions)))
            
            current = self.history[metric_key][-1]
            improved = (current < best_score) if minimize else (current > best_score)

            if improved:
                best_score = current
                best_iter = len(self.models) - 1  
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

            if self.early_stopping_rounds and no_improve_rounds >= self.early_stopping_rounds:
                break  
        
        if best_iter == -1:
            best_iter = len(self.models) - 1
        
        if use_best_model:
            self.models = self.models[:best_iter+1]
            self.gammas = self.gammas[:best_iter+1]
            self._features_idx = self._features_idx[:best_iter+1]

        # чтобы было удобнее смотреть
        for key in self.history:
            if use_best_model:
                self.history[key] = np.array(self.history[key][:best_iter+1])
            else:
                self.history[key] = np.array(self.history[key])

    def predict_proba(self, X: np.ndarray): 
        predicted_proba = np.zeros(X.shape[0])
        
        if self.cat_encoder is not None:
            X = self.cat_encoder.transform(X)
        
        if self.quantizer is not None:
            X = self.quantizer.transform(X)
        
        for model, gamma, feature_idx in zip(self.models, self.gammas, self._features_idx): 
            predicted_proba += self.learning_rate * gamma * model.predict(X[:, feature_idx])
        proba_1 = self.sigmoid(predicted_proba)
        proba_0 = 1 - proba_1
        return np.vstack([proba_0, proba_1]).T

    def find_optimal_gamma(self, y: np.ndarray, old_predictions: np.ndarray, new_predictions: np.ndarray) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [
            self.loss_fn(y, old_predictions + gamma * new_predictions * self.learning_rate)
            for gamma in gammas
        ]
        return gammas[np.argmin(losses)]

    def score(self, X: np.ndarray, y: np.ndarray):
        return roc_auc_score(y == 1, self.predict_proba(X)[:, 1])
    
    def plot_history(self, keys: str | Iterable[str]): 
        if isinstance(keys, str):
            keys = [keys]
        else:
            keys = list(keys)
        plt.figure()
        for key in keys:
            if key not in self.history:
                raise KeyError(f"Unknown metric key: {key}. Available: {list(self.history.keys())}")
            
            plt.plot(self.history[key], label=key)
            plt.xlabel("iteration")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def feature_importances_(self): 
        if len(self.models) == 0:
            raise RuntimeError("Model is not fitted yet.")
        d = self.n_features_in_
        imp = np.zeros(d, dtype=float)
        
        for model, gamma, feature_idx in zip(self.models, self.gammas, self._features_idx):
            imp[feature_idx] += abs(gamma) * model.feature_importances_
            
        s = imp.sum()
        if s > 0:
            imp /= s

   
        if (imp < -1e-12).any():
            raise RuntimeError("Feature importances became negative — check weights.")

        return imp
        
        
class CatFeaturesEncoder():
    def __init__(self, cat_features: Iterable | None):
        self.cat_features = list(cat_features) if cat_features is not None else []

        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        y01 = (y_train == 1).astype(float)
        self.global_mean = y01.mean() # просто доля положительных объектов на обучении
        self.tables = {}
        
        for j in self.cat_features:
            col = X_train[:, j] # берем категориальный признак
            cats, inv = np.unique(col, return_inverse=True)
            cnt = np.bincount(inv)
            pos = np.bincount(inv, weights=y01)
            mean = pos / cnt
            self.tables[j] = (cats, mean)
    
    def fit_transform(self, X_train, y_train):
        y01 = (y_train == 1).astype(float)
        self.global_mean = y01.mean() # просто доля положительных объектов на обучении
        self.tables = {}
        X_train_encoded = X_train.copy()
        
        for j in self.cat_features:
            col = X_train[:, j] # берем категориальный признак
            cats, inv = np.unique(col, return_inverse=True)
            order = np.argsort(inv)
            inv_s = inv[order]
            y_s = y01[order]
            start = np.r_[True, inv_s[1:] != inv_s[:-1]]
            idx_all = np.arange(len(inv_s))
            group_start_idx = np.maximum.accumulate(np.where(start, idx_all, 0))
            cnt_in_group = idx_all - group_start_idx + 1
            pos_cum = np.cumsum(y_s)
            pos_cum_prev = np.r_[0.0, pos_cum[:-1]]         
            pos_base = np.where(start, pos_cum_prev, 0.0)   
            pos_base = np.maximum.accumulate(pos_base)      
            pos_in_group = pos_cum - pos_base  
            prev_cnt = cnt_in_group - 1
            prev_pos = pos_in_group - y_s
            encoded_s = np.where(prev_cnt > 0, prev_pos / prev_cnt, self.global_mean)
            encoded = np.empty_like(encoded_s, dtype=float)
            encoded[order] = encoded_s
            X_train_encoded[:, j] = encoded
            cnt = np.bincount(inv)
            pos = np.bincount(inv, weights=y01)
            mean = pos / cnt
            self.tables[j] = (cats, mean)
        return X_train_encoded.astype(float)    
        
    def transform(self, X: np.ndarray):
        X_out = X.copy()
        for j in self.cat_features:
            cats, mean = self.tables[j]
            col = X_out[:, j]
            
            idx = np.searchsorted(cats, col)
            ok = idx < len(cats)
            ok[ok] = cats[idx[ok]] == col[ok]

            encoded = np.full(col.shape[0], self.global_mean, dtype=float)
            encoded[ok] = mean[idx[ok]]

            X_out[:, j] = encoded
        return X_out.astype(float)

class Quantizator:
    def __init__(self, quantization_type: str | None = None, nbins: int = 255):
        self.quantization_type = quantization_type
        self.nbins = nbins

        self.thresholds_ = None   
        self.mins_ = None        
        self.maxs_ = None         

    def fit(self, X: np.ndarray):
        if self.quantization_type is None:
            self.thresholds_ = None
            self.mins_ = None
            self.maxs_ = None
            return self

        if self.quantization_type not in ("uniform", "quantile", "min_entropy"):
            raise ValueError("quantization_type must be one of: None, 'uniform', 'quantile', 'min_entropy'")

        if self.nbins < 2:
            raise ValueError("nbins must be >= 2")

        X = np.asarray(X)
        d = X.shape[1]

        thresholds = []
        mins = np.empty(d, dtype=float)
        maxs = np.empty(d, dtype=float)

        q_grid = np.linspace(0.0, 1.0, self.nbins + 1)[1:-1]

        for j in range(d):
            col = X[:, j].astype(float, copy=False)

            if self.quantization_type == "uniform":
                mn = np.nanmin(col)
                mx = np.nanmax(col)
                mins[j] = mn
                maxs[j] = mx

                if not np.isfinite(mn) or not np.isfinite(mx) or mn == mx:
                    thr = np.array([], dtype=float)
                else:
                    thr = np.linspace(mn, mx, self.nbins + 1)[1:-1].astype(float)

            elif self.quantization_type == "quantile":
                if np.isnan(col).any():
                    thr = np.nanquantile(col, q_grid, method="linear")
                else:
                    try:
                        thr = np.quantile(col, q_grid, method="linear")
                    except TypeError:
                        thr = np.quantile(col, q_grid, interpolation="linear")
                thr = np.asarray(thr, dtype=float)

            else:  
                col2 = col[~np.isnan(col)]
                if col2.size == 0:
                    thr = np.array([], dtype=float)
                else:
                    u, w = np.unique(col2, return_counts=True)  
                    m = u.size

                    if m <= 1 or self.nbins >= m:
                        thr = np.array([], dtype=float)
                    else:
                        groups = [(i, i) for i in range(m)]

                        while len(groups) > self.nbins:
                            gw = np.array([w[l:r + 1].sum() for (l, r) in groups])
                            i_min = int(np.argmin(gw))

                            if i_min == 0:
                                j_nb = 1
                            elif i_min == len(groups) - 1:
                                j_nb = i_min - 1
                            else:
                                j_nb = i_min - 1 if gw[i_min - 1] <= gw[i_min + 1] else i_min + 1

                            l1, r1 = groups[i_min]
                            l2, r2 = groups[j_nb]
                            new_group = (min(l1, l2), max(r1, r2))

                            a, b = sorted([i_min, j_nb], reverse=True)
                            groups.pop(a)
                            groups.pop(b)
                            groups.append(new_group)
                            groups.sort()

                        groups.sort()
                        right_edges = np.array([r for (l, r) in groups[:-1]], dtype=int)
                        thr = ((u[right_edges] + u[right_edges + 1]) / 2.0).astype(float)

            thresholds.append(thr)

        self.thresholds_ = thresholds
        self.mins_ = mins if self.quantization_type == "uniform" else None
        self.maxs_ = maxs if self.quantization_type == "uniform" else None
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.quantization_type is None:
            return np.asarray(X)

        if self.thresholds_ is None:
            raise RuntimeError("Quantizator is not fitted yet. Call fit() first.")

        X = np.asarray(X)
        n, d = X.shape
        if d != len(self.thresholds_):
            raise ValueError("X has different number of features than during fit().")

        if self.nbins <= 256:
            out_dtype = np.uint8
        elif self.nbins <= 65536:
            out_dtype = np.uint16
        else:
            out_dtype = np.uint32

        Xq = np.empty((n, d), dtype=out_dtype)

        for j in range(d):
            col = X[:, j].astype(float, copy=False)
            thr = self.thresholds_[j]

            if np.isnan(col).any():
                col = col.copy()
                col[np.isnan(col)] = -np.inf

            if self.quantization_type == "uniform" and thr.size > 0:
                col = np.clip(col, self.mins_[j], self.maxs_[j])

            Xq[:, j] = np.searchsorted(thr, col, side="right").astype(out_dtype, copy=False)

        return Xq

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

