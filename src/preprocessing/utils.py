"""
Various functions that may be helpful in the preprocessing steps of pipeline such as technical indicator calculations
Source/links for technical indicators:
    - 
"""

import numpy as np


def EMA(df, window):
    """
    Calculates exponential moving average given price data and sliding window size
    """
    scaler = np.exp(-1/window)
    num = 0
    denom = 0
    exp_ma = []
    for p in df['VWAP']:
        num = scaler*num + p
        denom = scaler*denom + 1
        exp_ma.append(num/denom)
    return exp_ma


def EMA_5(df):
    return EMA(df, 5)


def EMA_20(df):
    return EMA(df, 20)


def EMA_50(df):
    return EMA(df, 50)