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


def RSI(df):
    """
    Calculates relative strength index on a 14-day lookback window (standard)
    https://www.investopedia.com/terms/r/rsi.asp
    """
    rsi = []
    for i in range(df.shape[0]):
        if i < 15:
            rsi.append(0)
        window = df.iloc[i-15:i]
        changes = window['VWAP'].pct_change()
        avg_gain = changes[changes > 0].sum() / 14
        avg_loss = np.abs(changes[changes < 0].sum()) / 14
        rsi.append( 100 - 100 / (1 + avg_gain/avg_loss) )
    return rsi


def BollingerBands(df):
    """
    Calculates bollinger bands which is basically just 20 day lookback volatility
    https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_bands
    """
    bb = []
    for i in range(df.shape[0]):
        start_idx = max(0, i-20)
        window = df.iloc[i-20:i]
        vol = window['VWAP'].std()
        bb.append(vol)
    return bb
