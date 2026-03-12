"""Shared computation functions for BTC binary options pricing system."""

import numpy as np
import pandas as pd
from scipy import stats
from config import *


# =============================================================================
# Bar Construction
# =============================================================================

class BarBuilder:
    """Accumulates tick trades into time bars (1-second and 1-minute)."""

    def __init__(self):
        self._accum = {}

    def build_bars(self, trades: pd.DataFrame, bar_sec: int = 60) -> pd.DataFrame:
        """Build OHLCV bars from tick trades.

        Args:
            trades: DataFrame with columns [timestamp, price, amount, side].
                    side: 'buy' or 'sell' (taker side, from is_buyer_maker).
            bar_sec: Bar size in seconds.
        """
        trades = trades.copy()
        trades['bar'] = trades['timestamp'].dt.floor(f'{bar_sec}s')
        grouped = trades.groupby('bar')

        bars = pd.DataFrame({
            'open': grouped['price'].first(),
            'high': grouped['price'].max(),
            'low': grouped['price'].min(),
            'close': grouped['price'].last(),
            'volume': grouped['amount'].sum(),
            'buy_volume': grouped.apply(
                lambda g: g.loc[g['side'] == 'buy', 'amount'].sum()
            ),
            'sell_volume': grouped.apply(
                lambda g: g.loc[g['side'] == 'sell', 'amount'].sum()
            ),
            'trade_count': grouped['price'].count(),
            'vwap': grouped.apply(
                lambda g: (g['price'] * g['amount']).sum() / g['amount'].sum()
            ),
        })
        bars.index.name = 'timestamp'
        return bars


# =============================================================================
# Feature Engineering: TOFI (Trade-Initiated Order Flow Imbalance)
# =============================================================================

def compute_tofi(bars: pd.DataFrame, windows_sec: list = None,
                 bar_sec: int = 60, prefix: str = '') -> pd.DataFrame:
    """Compute TOFI variants: (V_buy - V_sell) / (V_buy + V_sell).

    Returns DataFrame with tofi columns for each window.
    """
    if windows_sec is None:
        windows_sec = TOFI_WINDOWS_SEC

    result = pd.DataFrame(index=bars.index)
    for w in windows_sec:
        n = max(1, w // bar_sec)
        buy_sum = bars['buy_volume'].rolling(n, min_periods=1).sum()
        sell_sum = bars['sell_volume'].rolling(n, min_periods=1).sum()
        total = buy_sum + sell_sum
        label = f'{prefix}tofi_{w // 60}m'
        result[label] = (buy_sum - sell_sum) / total.replace(0, np.nan)

    # Momentum: delta of 5m TOFI over 1 bar
    key_5m = f'{prefix}tofi_5m'
    if key_5m in result.columns:
        result[f'{prefix}tofi_momentum'] = result[key_5m].diff()

    return result


def compute_tofi_divergence(spot_tofi: pd.DataFrame,
                            perp_tofi: pd.DataFrame) -> pd.Series:
    """Perp TOFI_5m minus spot TOFI_5m."""
    return perp_tofi.get('perp_tofi_5m', pd.Series(dtype=float)) - \
           spot_tofi.get('spot_tofi_5m', pd.Series(dtype=float))


# =============================================================================
# Feature Engineering: Cross-Exchange Basis
# =============================================================================

def compute_basis_features(perp_bars: pd.DataFrame,
                           spot_bars: pd.DataFrame) -> pd.DataFrame:
    """Compute perpetual-spot basis features.

    Both DataFrames must share the same index (aligned timestamps).
    """
    basis = (perp_bars['close'] - spot_bars['close']) / spot_bars['close']
    result = pd.DataFrame({'perp_spot_basis': basis}, index=basis.index)

    # Z-scores
    for w, label in [(60, '1h'), (1440, '24h')]:
        roll = basis.rolling(w, min_periods=max(1, w // 4))
        result[f'basis_zscore_{label}'] = (basis - roll.mean()) / roll.std().replace(0, np.nan)

    # Momentum
    for w, label in [(5, '5m'), (15, '15m')]:
        result[f'basis_momentum_{label}'] = basis.diff(w)

    # Acceleration
    result['basis_accel'] = result['basis_momentum_5m'].diff()

    return result


# =============================================================================
# Feature Engineering: Funding Rate & Liquidation
# =============================================================================

def compute_funding_features(funding_rate: pd.Series,
                             oi: pd.Series,
                             liquidations: pd.DataFrame = None,
                             tofi_5m: pd.Series = None,
                             mark_price: pd.Series = None,
                             index_price: pd.Series = None) -> pd.DataFrame:
    """Compute funding rate, OI, and liquidation features."""
    result = pd.DataFrame(index=funding_rate.index)
    result['funding_rate_predicted'] = funding_rate

    # Z-score over 7 days (assuming 1-min bars: 7*1440)
    w = 7 * 1440
    roll = funding_rate.rolling(w, min_periods=max(1, w // 4))
    result['funding_rate_zscore'] = (funding_rate - roll.mean()) / roll.std().replace(0, np.nan)

    # Premium momentum
    if mark_price is not None and index_price is not None:
        premium = (mark_price - index_price) / index_price
        result['premium_momentum_5m'] = premium.diff(5)

    # OI change
    if oi is not None:
        oi_avg_24h = oi.rolling(1440, min_periods=60).mean()
        result['oi_change_5m'] = oi.diff(5) / oi_avg_24h.replace(0, np.nan)

        # OI-TOFI divergence: OI rising while TOFI near zero
        if tofi_5m is not None:
            aligned_tofi = tofi_5m.reindex(oi.index, method='ffill')
            result['oi_tofi_divergence'] = result['oi_change_5m'].abs() * (1 - aligned_tofi.abs())

    # Liquidation features
    if liquidations is not None and len(liquidations) > 0:
        avg_vol = liquidations['amount'].rolling(1440, min_periods=60).mean()
        result['liquidation_intensity_1m'] = liquidations['amount'] / avg_vol.replace(0, np.nan)
        if 'side' in liquidations.columns:
            long_liq = liquidations.loc[liquidations['side'] == 'long', 'amount']
            short_liq = liquidations.loc[liquidations['side'] == 'short', 'amount']
            total_liq = long_liq.add(short_liq, fill_value=0)
            result['liquidation_imbalance'] = (long_liq.subtract(short_liq, fill_value=0)) / \
                                               total_liq.replace(0, np.nan)

    return result


# =============================================================================
# Feature Engineering: Realized Higher Moments
# =============================================================================

def compute_realized_moments(second_bars: pd.DataFrame,
                             windows_sec: list = None) -> pd.DataFrame:
    """Compute realized skewness and kurtosis from 1-second log returns."""
    if windows_sec is None:
        windows_sec = MOMENT_WINDOWS_SEC

    log_ret = np.log(second_bars['close'] / second_bars['close'].shift(1))
    result = pd.DataFrame(index=second_bars.index)

    for w in windows_sec:
        label = f'{w // 60}m' if w >= 60 else f'{w}s'
        roll = log_ret.rolling(w, min_periods=max(30, w // 4))
        mean = roll.mean()
        std = roll.std().replace(0, np.nan)
        standardized = (log_ret - mean) / std

        result[f'realized_skew_{label}'] = standardized.pow(3).rolling(w, min_periods=30).mean()
        result[f'realized_kurt_{label}'] = standardized.pow(4).rolling(w, min_periods=30).mean()

    return result


# =============================================================================
# Feature Engineering: Microstructure (HAR-RV, VPIN, Kyle's Lambda)
# =============================================================================

def compute_har_rv(bars: pd.DataFrame) -> pd.DataFrame:
    """HAR-RV: realized volatility at daily, weekly, monthly lags."""
    log_ret = np.log(bars['close'] / bars['close'].shift(1))
    rv_1m = log_ret ** 2

    result = pd.DataFrame(index=bars.index)
    # For 1-min bars: daily=1440, weekly=10080, monthly=43200
    for n, label in [(1440, 'daily'), (10080, 'weekly'), (43200, 'monthly')]:
        result[f'rv_{label}'] = rv_1m.rolling(n, min_periods=max(1, n // 4)).sum()

    # Current 5m, 15m, 1h RV
    for n, label in [(5, '5m'), (15, '15m'), (60, '1h')]:
        result[f'rv_{label}'] = rv_1m.rolling(n, min_periods=1).sum()

    return result


def compute_vpin(bars: pd.DataFrame, bucket_size: int = None,
                 n_buckets: int = None) -> pd.Series:
    """Volume-synchronized probability of informed trading (VPIN).

    Simplified: uses buy/sell volume ratio per time bar as proxy.
    """
    if bucket_size is None:
        bucket_size = VPIN_BUCKET_SIZE
    if n_buckets is None:
        n_buckets = VPIN_N_BUCKETS

    buy_vol = bars['buy_volume']
    sell_vol = bars['sell_volume']
    total_vol = buy_vol + sell_vol
    imbalance = (buy_vol - sell_vol).abs()

    # Bucket by cumulative volume
    cum_vol = total_vol.cumsum()
    bucket_id = (cum_vol // bucket_size).astype(int)
    bucket_imbalance = imbalance.groupby(bucket_id).sum()
    bucket_total = total_vol.groupby(bucket_id).sum()

    vpin_raw = bucket_imbalance.rolling(n_buckets, min_periods=1).sum() / \
               bucket_total.rolling(n_buckets, min_periods=1).sum()

    # Map back to bar index
    vpin = vpin_raw.reindex(bucket_id).values
    return pd.Series(vpin, index=bars.index, name='vpin')


def compute_kyles_lambda(bars: pd.DataFrame, window: int = 60) -> pd.Series:
    """Kyle's Lambda: price impact per unit of signed volume."""
    log_ret = np.log(bars['close'] / bars['close'].shift(1))
    signed_vol = bars['buy_volume'] - bars['sell_volume']

    # Rolling regression: Δp = λ * signed_vol + ε
    cov = log_ret.rolling(window, min_periods=10).cov(signed_vol)
    var = signed_vol.rolling(window, min_periods=10).var()
    return (cov / var.replace(0, np.nan)).rename('kyles_lambda')


def compute_spread_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Spread-related features from OHLC bars."""
    result = pd.DataFrame(index=bars.index)
    result['hl_spread'] = (bars['high'] - bars['low']) / bars['close']
    result['hl_spread_5m'] = result['hl_spread'].rolling(5, min_periods=1).mean()
    result['hl_spread_15m'] = result['hl_spread'].rolling(15, min_periods=1).mean()
    return result


def compute_trade_intensity(bars: pd.DataFrame) -> pd.DataFrame:
    """Trade count and volume intensity features."""
    result = pd.DataFrame(index=bars.index)
    result['trade_count_zscore'] = (
        bars['trade_count'] - bars['trade_count'].rolling(60, min_periods=10).mean()
    ) / bars['trade_count'].rolling(60, min_periods=10).std().replace(0, np.nan)
    result['volume_zscore'] = (
        bars['volume'] - bars['volume'].rolling(60, min_periods=10).mean()
    ) / bars['volume'].rolling(60, min_periods=10).std().replace(0, np.nan)
    result['buy_ratio'] = bars['buy_volume'] / bars['volume'].replace(0, np.nan)
    return result


def compute_microstructure_features(bars: pd.DataFrame) -> pd.DataFrame:
    """All symmetric microstructure features combined."""
    parts = [
        compute_har_rv(bars),
        compute_spread_features(bars),
        compute_trade_intensity(bars),
    ]
    vpin = compute_vpin(bars)
    kl = compute_kyles_lambda(bars)
    result = pd.concat(parts + [vpin, kl], axis=1)
    return result


# =============================================================================
# Regime Detection Helpers
# =============================================================================

def rolling_variance_ratio(returns: pd.Series, q: int = None,
                           window: int = 60) -> pd.Series:
    """Variance ratio: VR(q) = Var(q-period) / (q * Var(1-period))."""
    if q is None:
        q = VARIANCE_RATIO_Q
    var_1 = returns.rolling(window, min_periods=10).var()
    q_ret = returns.rolling(q).sum()
    var_q = q_ret.rolling(window, min_periods=10).var()
    return (var_q / (q * var_1).replace(0, np.nan)).rename('variance_ratio')


def rolling_hurst(returns: pd.Series, window: int = None) -> pd.Series:
    """Hurst exponent via rescaled range (R/S) analysis."""
    if window is None:
        window = HURST_WINDOW

    def _hurst(x):
        if len(x) < 20:
            return np.nan
        mean = x.mean()
        deviations = np.cumsum(x - mean)
        r = deviations.max() - deviations.min()
        s = x.std()
        if s == 0:
            return np.nan
        return np.log(r / s) / np.log(len(x))

    return returns.rolling(window, min_periods=20).apply(_hurst, raw=True).rename('hurst')


def rolling_autocorrelation(returns: pd.Series, window: int = None) -> pd.Series:
    """Rolling lag-1 autocorrelation."""
    if window is None:
        window = AUTOCORR_WINDOW
    return returns.rolling(window, min_periods=10).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=True
    ).rename('autocorr_lag1')


# =============================================================================
# Directional Probability / Strike Probability
# =============================================================================

def compute_strike_probability(pi, mu, sigma, nu, strike_return=0.0):
    """P(r > strike_return) from Student-t mixture.

    Args:
        pi: mixture weights, shape (K,)
        mu: location params, shape (K,)
        sigma: scale params, shape (K,)
        nu: degrees of freedom, shape (K,)
        strike_return: log-return threshold (0 for Polymarket up/down)

    Returns:
        Probability that return exceeds strike_return.
    """
    prob = 0.0
    for k in range(len(pi)):
        z = (strike_return - mu[k]) / sigma[k]
        cdf_val = stats.t.cdf(z, df=nu[k])
        prob += pi[k] * (1 - cdf_val)
    return prob


# =============================================================================
# Market Making
# =============================================================================

def reservation_price(p, q, gamma, time_remaining):
    """Modified Avellaneda-Stoikov reservation price for binary options.

    Args:
        p: fair value of binary option [0, 1]
        q: current inventory (positive = long)
        gamma: risk aversion
        time_remaining: fraction of 15-min window remaining
    """
    variance = p * (1 - p)
    return p - q * gamma * variance * time_remaining


def compute_spread(p, gamma, k_param, time_remaining):
    """AS spread + gamma-risk premium for binary options.

    Args:
        p: fair value [0, 1]
        gamma: risk aversion
        k_param: order arrival intensity parameter
        time_remaining: fraction of window remaining
    """
    variance = p * (1 - p)
    as_spread = gamma * variance * time_remaining + (2 / gamma) * np.log(1 + gamma / k_param)

    # Binary gamma premium (increases near expiry for ATM)
    if time_remaining > 0.01:
        z = (p - 0.5) / (0.25 * np.sqrt(time_remaining))
        binary_gamma = np.exp(-0.5 * z ** 2) / (np.sqrt(2 * np.pi) * 0.25 * np.sqrt(time_remaining))
    else:
        binary_gamma = 1e6

    total = as_spread + GAMMA_PREMIUM_COEFF * abs(binary_gamma)
    return max(total, MIN_SPREAD)


def compute_quotes(res_price, half_spread, alpha, skew_sensitivity=None):
    """Compute bid/ask with alpha-based skewing.

    Args:
        res_price: reservation price
        half_spread: half the total spread
        alpha: directional signal (P(up) - 0.5)
        skew_sensitivity: how aggressively to lean into alpha
    """
    if skew_sensitivity is None:
        skew_sensitivity = SKEW_SENSITIVITY
    skew = alpha * skew_sensitivity
    bid = res_price - half_spread + skew
    ask = res_price + half_spread + skew

    if ask - bid < MIN_SPREAD:
        mid = (bid + ask) / 2
        bid = mid - MIN_SPREAD / 2
        ask = mid + MIN_SPREAD / 2

    return np.clip(bid, 0, 1), np.clip(ask, 0, 1)


def check_toxicity(vpin_current, vpin_rolling_mean, vpin_rolling_std):
    """VPIN-based toxicity check for quoting decisions."""
    if vpin_rolling_std == 0:
        return 'NORMAL'
    zscore = (vpin_current - vpin_rolling_mean) / vpin_rolling_std
    if zscore > VPIN_WITHDRAW_ZSCORE:
        return 'WITHDRAW'
    elif zscore > VPIN_WIDEN_MAX_ZSCORE:
        return 'WIDEN_MAX'
    elif zscore > VPIN_WIDEN_ZSCORE:
        return 'WIDEN'
    return 'NORMAL'


def time_spread_multiplier(time_remaining, p):
    """Time-dependent spread multiplier from the quoting zones table."""
    atm = 0.40 <= p <= 0.60
    near_atm = 0.20 <= p <= 0.80
    # far_otm = p < 0.20 or p > 0.80

    if time_remaining > 10 / 15:
        return 1.0
    elif time_remaining > 5 / 15:
        return 2.0 if atm else (1.5 if near_atm else 0.8)
    elif time_remaining > 1 / 15:
        return 4.0 if atm else (2.0 if near_atm else 1.0)
    else:
        return np.inf if atm else (3.0 if near_atm else 1.0)


# =============================================================================
# Backtest Helpers
# =============================================================================

def compute_backtest_metrics(pnl: pd.Series, predictions: pd.Series,
                             actuals: pd.Series) -> dict:
    """Compute the 7 target metrics for backtest evaluation."""
    brier = ((predictions - actuals) ** 2).mean()
    accuracy = ((predictions > 0.5) == actuals).mean()

    # Sharpe (annualized from 15-min periods: 365.25 * 96 periods/day)
    periods_per_year = 365.25 * 96
    sharpe = pnl.mean() / pnl.std() * np.sqrt(periods_per_year) if pnl.std() > 0 else 0

    # Max drawdown (rolling 24h = 96 periods)
    cum_pnl = pnl.cumsum()
    rolling_max = cum_pnl.rolling(96, min_periods=1).max()
    drawdown = (rolling_max - cum_pnl) / rolling_max.replace(0, np.nan).abs().clip(lower=1e-8)
    max_dd = drawdown.max()

    # Profit factor
    wins = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    profit_factor = wins / losses if losses > 0 else np.inf

    return {
        'brier_score': brier,
        'accuracy': accuracy,
        'sharpe_ratio': sharpe,
        'max_drawdown_24h': max_dd,
        'profit_factor': profit_factor,
    }
