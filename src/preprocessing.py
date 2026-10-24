"""Preprocessing for the Kaggle Loan Prediction dataset.

Every statistic (imputation neighbours, modes, IQR bounds) is learned on the
training fold only, so the transformer can sit inside a scikit-learn Pipeline
and be refit inside cross-validation.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer

NUMERIC = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
           "Loan_Amount_Term", "Credit_History"]
CONTINUOUS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
CATEGORICAL = ["Gender", "Married", "Dependents", "Education",
               "Self_Employed", "Property_Area"]


class LoanPreprocessor(BaseEstimator, TransformerMixin):
    """KNN-impute numeric columns, mode-impute categoricals, IQR-clip `clip_cols`, encode."""

    def __init__(self, clip_cols=None, n_neighbors=10):
        self.clip_cols = clip_cols
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        X = X.copy()
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors).fit(X[NUMERIC])
        self.modes_ = {c: X[c].mode()[0] for c in CATEGORICAL}
        X[NUMERIC] = self.imputer_.transform(X[NUMERIC])
        self.bounds_ = {}
        for c in (self.clip_cols or []):
            q1, q3 = X[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.bounds_[c] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        return self

    def transform(self, X):
        X = X.copy()
        X[NUMERIC] = self.imputer_.transform(X[NUMERIC])
        for c, (lo, hi) in self.bounds_.items():
            X[c] = X[c].clip(lo, hi)
        for c, m in self.modes_.items():
            X[c] = X[c].fillna(m)
        out = pd.DataFrame(index=X.index)
        out[NUMERIC] = X[NUMERIC]
        out["Gender"] = (X["Gender"] == "Male").astype(float)
        out["Married"] = (X["Married"] == "Yes").astype(float)
        out["Dependents"] = X["Dependents"].replace("3+", "3").astype(float)
        out["Education"] = (X["Education"] == "Graduate").astype(float)
        out["Self_Employed"] = (X["Self_Employed"] == "Yes").astype(float)
        out["Property_Semiurban"] = (X["Property_Area"] == "Semiurban").astype(float)
        out["Property_Urban"] = (X["Property_Area"] == "Urban").astype(float)
        return out
