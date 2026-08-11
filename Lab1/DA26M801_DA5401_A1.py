"""
DA5401 DA Lab - Assignment 1
Student Name: Akash Kumar Jaishwal
Roll Number: DA26M801
"""

import numpy as np
import pandas as pd


def _DA26M801_clean_y(y):
    # y ko flatten karke float array bana rahe hain, DataFrame/Series dono handle honge
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.to_numpy()
    else:
        y = np.array(y, copy=True)

    if y.ndim == 2:
        if y.shape[1] != 1:
            raise ValueError("y single column array hona chahiye.")
        y = y.reshape(-1)
    elif y.ndim != 1:
        raise ValueError("y strictly 1-dimensional hona chahiye.")

    try:
        y = y.astype(np.float64)
    except Exception:
        raise ValueError("y mein non-numeric values nahi honi chahiye.")

    if not np.all(np.isfinite(y)):
        raise ValueError("y mein NaN/Inf values hain.")
    return y


def _check_hyperparams(lr, iterations, tol):
    if type(iterations) is bool or not isinstance(iterations, (int, np.integer)) or iterations <= 0:
        raise ValueError("max_iter positive integer hona chahiye.")
    if lr <= 0:
        raise ValueError("learning_rate > 0 hona chahiye.")
    if tol < 0:
        raise ValueError("tol >= 0 hona chahiye.")


def linear_predict(X, weights, bias):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X 2D array hona chahiye.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X mein NaN/Inf values hain.")
    w = np.asarray(weights, dtype=np.float64)
    return (X @ w + bias).astype(np.float64)


def linear_loss(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_pred - y_true) ** 2))


def linear_gradients(X, y, weights, bias):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X empty ya non-2D nahi ho sakta.")

    y = _DA26M801_clean_y(y)
    n = X.shape[0]

    error = linear_predict(X, weights, bias) - y
    dw = (2.0 / n) * (X.T @ error)
    db = float((2.0 / n) * np.sum(error))
    return dw, db


def fit_linear_regression(X, y, learning_rate=0.01, max_iter=1000, tol=1e-8):
    _check_hyperparams(learning_rate, max_iter, tol)

    X = X.to_numpy(dtype=np.float64) if isinstance(
        X, (pd.DataFrame, pd.Series)) else np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X non-empty 2D array hona chahiye.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X mein invalid values hain.")

    y = _DA26M801_clean_y(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X aur y ke sample size match nahi kar rahe.")

    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history = []

    for _ in range(max_iter):
        dw, db = linear_gradients(X, y, weights, bias)
        weights -= learning_rate * dw
        bias -= learning_rate * db

        loss_history.append(linear_loss(y, linear_predict(X, weights, bias)))
        if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) <= tol:
            break

    return {"weights": weights, "bias": float(bias), "loss_history": loss_history}


def predict_linear_regression(model, X):
    return linear_predict(X, model["weights"], model["bias"])


def fit_preprocessor(X):
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X Pandas DataFrame hona chahiye.")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("DataFrame X empty nahi hona chahiye.")

    cols = list(X.columns)
    num_cols, cat_cols = [], []
    categories = {}
    encoded_names = []

    for col in cols:
        series = X[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            num_cols.append(col)
            encoded_names.append(col)
        else:
            cat_cols.append(col)
            levels = sorted(str(v) for v in series.unique())
            categories[col] = levels
            encoded_names.extend(f"{col}={v}" for v in levels)

    return {
        "original_columns": cols,
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "categories": categories,
        "encoded_feature_names": encoded_names,
    }


def transform_features(X, preprocessor):
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X Pandas DataFrame hona chahiye.")

    train_cols = preprocessor["original_columns"]
    if list(X.columns) != train_cols:
        if set(X.columns) != set(train_cols):
            raise ValueError(
                "Test columns training schema se match nahi karte.")
        X = X[train_cols]

    blocks = []
    for col in train_cols:
        data = X[col]
        if col in preprocessor["numerical_columns"]:
            arr = data.to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Numeric column '{col}' mein NaN/Inf hain.")
            blocks.append(arr.reshape(-1, 1))
        else:
            str_data = data.astype(str)
            levels = preprocessor["categories"][col]
            one_hot = np.column_stack(
                [(str_data == lvl).to_numpy(dtype=np.float64) for lvl in levels])
            blocks.append(one_hot)

    return np.hstack(blocks)


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)

    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_neg = np.exp(z[neg])
    out[neg] = exp_neg / (1.0 + exp_neg)
    return out


def logistic_loss(y_true, probabilities, epsilon=1e-12):
    if not (0.0 < epsilon < 0.5):
        raise ValueError("epsilon 0 aur 0.5 ke beech hona chahiye.")

    y = _DA26M801_clean_y(y_true)
    p = np.asarray(probabilities, dtype=np.float64)
    if not np.all(np.isfinite(p)):
        raise ValueError("probabilities mein NaN/Inf nahi hone chahiye.")

    p = np.clip(p, epsilon, 1.0 - epsilon)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def logistic_gradients(X_encoded, y, weights, bias):
    X = np.asarray(X_encoded, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X_encoded non-empty 2D array hona chahiye.")

    y = _DA26M801_clean_y(y)
    n = X.shape[0]

    error = sigmoid(X @ weights + bias) - y
    dw = (1.0 / n) * (X.T @ error)
    db = float((1.0 / n) * np.sum(error))
    return dw, db


def fit_logistic_regression(X, y, learning_rate=0.01, max_iter=1000, tol=1e-8, epsilon=1e-12):
    _check_hyperparams(learning_rate, max_iter, tol)
    if not (0.0 < epsilon < 0.5):
        raise ValueError("epsilon 0 aur 0.5 ke beech hona chahiye.")

    y = _DA26M801_clean_y(y)
    if not np.all(np.isin(np.unique(y), [0, 1])):
        raise ValueError(
            "Logistic regression targets sirf 0 aur 1 hone chahiye.")

    preprocessor = fit_preprocessor(X)
    X_enc = transform_features(X, preprocessor)
    if X_enc.shape[0] != y.shape[0]:
        raise ValueError("X aur y ke sample counts mismatch ho rahe hain.")

    n_samples, n_features = X_enc.shape
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history = []

    for _ in range(max_iter):
        dw, db = logistic_gradients(X_enc, y, weights, bias)
        weights -= learning_rate * dw
        bias -= learning_rate * db

        probs = sigmoid(X_enc @ weights + bias)
        loss_history.append(logistic_loss(y, probs, epsilon=epsilon))
        if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) <= tol:
            break

    return {"weights": weights, "bias": float(bias), "preprocessor": preprocessor, "loss_history": loss_history}


def predict_proba_logistic(model, X):
    X_enc = transform_features(X, model["preprocessor"])
    return sigmoid(X_enc @ model["weights"] + model["bias"])


def predict_logistic(model, X, threshold=0.5):
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold 0.0 aur 1.0 ke beech hona chahiye.")
    return (predict_proba_logistic(model, X) >= threshold).astype(int)
