"""
Various functions that may be helpful in the preprocessing steps of pipeline such as technical indicator calculations
Source/links for technical indicators:
    - 
"""

import numpy as np
import pandas as pd


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


def SMA(df, window):
    """
    Calculates exponential moving average given price data and sliding window size
    """
    sma = []
    for i in range(df.shape[0]):
        start_idx = max(0, i-window)
        sma.append(df.iloc[start_idx:i]['VWAP'].mean())
                       
    return sma


def EMA_5(df):
    return EMA(df, 5)


def EMA_20(df):
    return EMA(df, 20)


def EMA_50(df):
    return EMA(df, 50)


def SMA_5(df):
    return SMA(df, 5)


def SMA_20(df):
    return SMA(df, 20)


def SMA_50(df):
    return SMA(df, 50)


def RSI(df):
    """
    Calculates relative strength index on a 14-day lookback window (standard)
    https://www.investopedia.com/terms/r/rsi.asp
    """
    rsi = []
    for i in range(df.shape[0]):
        if i < 15:
            rsi.append(0)
        else:
            window = df.iloc[i-15:i]
            changes = window['VWAP'].pct_change()
            avg_gain = changes[changes > 0].sum() / 14
            avg_loss = np.abs(changes[changes < 0].sum()) / 14
            if avg_loss > 0:
                rsi.append( 100 - 100 / (1 + avg_gain/avg_loss) )
            else:
                rsi.append(100)
    return rsi


def BollingerBands(df):
    """
    Calculates bollinger bands which is basically just 20 day lookback volatility
    https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_bands
    """
    bb = []
    for i in range(df.shape[0]):
        start_idx = max(0, i-20)
        window = df.iloc[start_idx:i]
        vol = window['VWAP'].std()
        bb.append(vol)
    return bb


def StochasticOscillator(df):
    stochastics = []
    for i in range(df.shape[0]):
        if i < 14:
            stochastics.append(np.nan)
        else:
            price_window = df.iloc[i-14:i]['VWAP'].values
            close = price_window[-1]
            low = price_window.min()
            high = price_window.max()
            pct_k = 100 * (close - low) / (high - low)
            stochastics.append(pct_k)
    
    return stochastics