# src/features.py
"""
Feature engineering utilities for Credit Card Fraud Detection.

These transformations go beyond simple scaling:
- Log transformation for skewed features (e.g., Amount)
- Time-based feature extraction (hour of day, day of week)
- Custom ratio / interaction features
"""

import numpy as np
import pandas as pd


def add_log_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log-transformed 'Amount' to reduce skewness.
    """
    if "Amount" in df.columns:
        df = df.copy()
        df["LogAmount"] = np.log1p(df["Amount"])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from 'Time' (in seconds).
    Dataset encodes 'Time' as seconds elapsed since first transaction.
    """
    if "Time" in df.columns:
        df = df.copy()
        df["Hour"] = (df["Time"] // 3600) % 24  # Hour of day
        df["Day"] = df["Time"] // (3600 * 24)  # Day index
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example interaction features (can be tuned later).
    """
    df = df.copy()
    if "Amount" in df.columns and "V1" in df.columns:
        df["Amt_V1_ratio"] = df["Amount"] / (df["V1"].abs() + 1e-6)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: applies all feature engineering steps.
    """
    df = add_log_amount(df)
    df = add_time_features(df)
    df = add_interaction_features(df)
    return df
