import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pykalman import KalmanFilter
from sklearn.linear_model import TheilSenRegressor

def prepare_df(df, freq='1s'):
    df['ts'] = pd.to_datetime(df['ts'], format='ISO8601')
    # Remove duplicate timestamps
    df = df.groupby('ts').agg({'price': 'mean', 'size': 'sum'}).sort_index()
    # Fill missing timestamps
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(full_index).ffill()
    return df

def resample_df(df, timeframe):
    rule = {'1s': 's', '1m': 'min', '5m': '5min'}[timeframe]
    resampled = df.resample(rule).agg({'price': 'mean', 'size': 'sum'})
    return resampled.ffill()

def hedge_ratio(a, b, reg_type='OLS'):
    a, b = a.align(b, join='inner')
    df = pd.concat([a, b], axis=1).dropna()

    if len(df) < 3:
        return np.nan

    a = df.iloc[:, 0]
    b = df.iloc[:, 1]

    # ---------- OLS ----------
    if reg_type == 'OLS':
        X = sm.add_constant(b, has_constant='add')
        model = sm.OLS(a, X).fit()

        # ✅ SAFE extraction
        params = model.params.values
        return params[-1]   # always returns slope

    # ---------- TLS ----------
    elif reg_type == 'TLS':
        # Subsample for large datasets to avoid memory issues
        if len(a) > 10000:
            step = len(a) // 10000 + 1
            a_sub = a[::step]
            b_sub = b[::step]
        else:
            a_sub = a
            b_sub = b
        x_mean = b_sub.mean()
        y_mean = a_sub.mean()
        x_c = b_sub - x_mean
        y_c = a_sub - y_mean
        A = np.column_stack([x_c, y_c])
        _, _, Vt = np.linalg.svd(A)
        return Vt[0, 1] / Vt[0, 0]

    # ---------- Huber ----------
    elif reg_type == 'Huber':
        X = sm.add_constant(b, has_constant='add')
        model = sm.RLM(a, X, M=sm.robust.norms.HuberT()).fit()
        return model.params.values[-1]

    # ---------- Theil-Sen ----------
    elif reg_type == 'Theil-Sen':
        tsr = TheilSenRegressor(random_state=42)
        tsr.fit(b.values.reshape(-1, 1), a.values)
        return tsr.coef_[0]

    # ---------- Non-linear ----------
    elif reg_type == 'Non-linear':
        coeffs = np.polyfit(b, a, 2)
        mean_b = b.mean()
        return 2 * coeffs[0] * mean_b + coeffs[1]

    # ---------- Kalman ----------
    elif reg_type == 'Kalman':
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=b.values.reshape(-1, 1),
            initial_state_mean=1.0,
            initial_state_covariance=1.0,
            observation_covariance=1.0,
            transition_covariance=0.01
        )
        state_means, _ = kf.filter(a.values)
        return state_means[-1, 0]

    else:
        raise ValueError("Unknown regression type")

def spread_zscore(a, b, beta, min_periods=10):
    a, b = a.align(b, join="inner")

    if np.isnan(beta):
        return pd.Series(dtype=float), pd.Series(dtype=float)

    spread = a - beta * b

    # Not enough data
    if spread.count() < min_periods:
        return spread, pd.Series(index=spread.index, dtype=float)

    std = spread.std()

    # Constant spread → no z-score
    if std == 0 or np.isnan(std):
        z = pd.Series(index=spread.index, dtype=float)
    else:
        z = (spread - spread.mean()) / std

    return spread, z


def rolling_corr(a, b, window):
    a, b = a.align(b, join='inner')
    return a.rolling(window).corr(b)

def adf_test(series):
    series = series.dropna()

    # Not enough data
    if len(series) < 10:
        return None, False

    # Constant or near-constant series → ADF undefined
    if series.nunique() <= 1:
        return None, False

    try:
        p = adfuller(series, autolag="AIC")[1]
        return p, p < 0.05
    except ValueError:
        return None, False


def backtest(spread, z, entry_threshold=2, exit_threshold=0):
    """
    Mini mean-reversion backtest: Enter short when z > entry_threshold, exit when z < exit_threshold.
    Assumes shorting the spread (bet on convergence).
    """
    position = 0  # 0: no position, -1: short spread
    trades = []
    entry_price = None
    entry_time = None
    for i in range(len(z)):
        if position == 0 and z.iloc[i] > entry_threshold:
            position = -1
            entry_price = spread.iloc[i]
            entry_time = spread.index[i]
        elif position == -1 and z.iloc[i] < exit_threshold:
            exit_price = spread.iloc[i]
            pnl = entry_price - exit_price  # Profit when spread decreases
            trades.append({
                'entry_time': entry_time,
                'exit_time': spread.index[i],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl
            })
            position = 0
            entry_price = None
            entry_time = None
    total_pnl = sum(t['pnl'] for t in trades) if trades else 0
    return trades, total_pnl
