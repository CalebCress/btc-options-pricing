# Implementation Plan: Direction-Conditional Binary Options Trading System

## Executive Summary

This document provides a complete development plan for transitioning from a volatility-based symmetric pricer to a direction-conditional, regime-aware trading system for 15-minute Bitcoin binary options on Polymarket. The system combines directional probability estimation, regime detection, and alpha-integrated market making into a unified architecture. Development is organized into six phases over approximately 14–18 weeks, with each phase producing independently testable and deployable components.

The core architectural shift: **stop treating this as a market making problem with a volatility model, and start treating it as a directional trading problem that provides liquidity as a byproduct of alpha generation.** This reframing, formalized by Cartea & Wang (2020) and empirically validated by Levitt (2004), is the single most important change.

-----

## Phase 0: Data Infrastructure Expansion (Weeks 1–2)

### Objective

Extend the existing Tardis tick data pipeline to ingest the additional data sources required for directional features, cross-exchange signals, and regime detection.

### 0.1 New Data Sources Required

**Binance USDT Perpetual (BTC/USDT-PERP)**
This is the single most critical addition. Alexander & Heck (2020, *Journal of Financial Stability*) demonstrated that perpetual swaps on unregulated exchanges dominate BTC price discovery. Binance perpetual is the #1 price leader, and the Bouchaud et al. fragmentation study achieved R² up to 25.9% predicting one exchange’s returns from Binance perpetual features at sub-second horizons. Post-2024 ETF introduction, Bitcoin ETFs now lead price discovery roughly 85% of the time (Springer 2025), so ETF data should also be considered.

Required fields from Binance perpetual:

- Tick-level trades with `is_buyer_maker` flag (aggressor classification)
- Order book snapshots: minimum 10 levels bid/ask, ideally 20 levels (per Lensky & Hao 2023, who used 20 LOB levels for their CNN-Aggr volatility model)
- Mark price and index price (for basis computation)
- Funding rate: both the current predicted rate and the 8-hourly settlement rate
- Open interest: total and per-side if available
- Liquidation feed: forced liquidation events with size, side, and price

**Spot Exchange Reference Price**
A composite spot price from Coinbase + Kraken + Bitstamp for the perpetual-spot basis calculation. Coinbase is preferred as the primary reference because Cont et al.’s OFI framework achieved R² ≈ 0.40 for contemporaneous BTC returns specifically on Coinbase data.

**Polymarket Order Book (CLOB)**
The Polymarket CLOB data for the specific 15-minute up/down contracts being traded:

- Resting order sizes at each price level
- Trade-level data with timestamps
- Own order status and fill notifications

### 0.2 Pipeline Architecture

Extend the existing stateful minute-bar pipeline to produce two parallel outputs:

1. **Minute-bar features** (existing pipeline, extended): All current microstructure features (VPIN, Kyle’s Lambda, HAR-RV, trade flow analysis, etc.) plus the new directional features computed at 1-minute granularity.
1. **Second-bar features** (new pipeline): A subset of fast-updating features computed at 1-second granularity for real-time regime detection and quote adjustment. This includes OFI, microprice, and cross-exchange basis.

The second-bar pipeline must be stateful across boundaries just like the existing minute pipeline. Use the same tracker pattern — persistent accumulators that carry state across bar boundaries.

### 0.3 Data Alignment

Cross-exchange features require careful timestamp alignment. Binance and Coinbase timestamps can differ by 50–200ms. For 1-second bar construction this matters; for 1-minute bars it generally does not. Apply the following:

- Normalize all timestamps to UTC microseconds
- For cross-exchange features at second granularity, use arrival-time alignment (assign each event to the second-bar of its arrival time) rather than exchange-time alignment
- For minute bars, this is negligible — use simple truncation to minute boundary

-----

## Phase 1: Directional Feature Engineering (Weeks 2–4)

### Objective

Implement the directional feature set that transforms the model from symmetric volatility pricing to conditional directional probability estimation. The research identifies a clear hierarchy: anti-symmetric features (those that flip sign when bid/ask sides are swapped) predict direction, while symmetric features predict volatility (Neural Networks for Direct Bitcoin Strike Probability Estimation).

### 1.1 Trade-Initiated Order Flow Imbalance (TOFI) — Highest Priority

**Literature basis:** Cont, Kukanov & Stoikov (2014, *Journal of Financial Econometrics*) established OFI as the strongest single predictor of short-term price changes. Bieganowski & Ślepaczuk (2025) confirmed OFI is the #1 feature by SHAP importance across five crypto assets at 1-second frequency.

**Construction:**

```
TOFI(t, Δ) = (V_buy(t, Δ) - V_sell(t, Δ)) / (V_buy(t, Δ) + V_sell(t, Δ))
```

Where `V_buy` and `V_sell` are aggregate volumes of buyer-initiated and seller-initiated trades over window Δ. In crypto, use the `is_buyer_maker` flag directly from the exchange API — no Lee-Ready classification needed.

**Feature variants to compute:**

|Feature                    |Window (Δ)|Source         |Rationale                          |
|---------------------------|----------|---------------|-----------------------------------|
|`tofi_spot_1m`             |1 min     |Spot (Coinbase)|Immediate flow pressure            |
|`tofi_spot_5m`             |5 min     |Spot           |Medium-term flow                   |
|`tofi_spot_15m`            |15 min    |Spot           |Full-window flow                   |
|`tofi_perp_1m`             |1 min     |Binance perp   |Leading indicator (perp leads spot)|
|`tofi_perp_5m`             |5 min     |Binance perp   |Medium-term perp flow              |
|`tofi_perp_15m`            |15 min    |Binance perp   |Full-window perp flow              |
|`tofi_momentum_spot`       |Δ(5m TOFI)|Spot           |Acceleration of flow imbalance     |
|`tofi_momentum_perp`       |Δ(5m TOFI)|Binance perp   |Acceleration of perp flow          |
|`tofi_perp_spot_divergence`|5 min     |Both           |Perp TOFI minus spot TOFI          |

**Implementation notes:**

- Compute in the existing stateful tracker pattern. Maintain running buy/sell volume accumulators per window.
- The `tofi_momentum` features are the first derivative: `tofi_5m(t) - tofi_5m(t-1m)`. This captures whether flow pressure is accelerating.
- The `tofi_perp_spot_divergence` is critical: when perpetual aggression leads spot aggression, it signals directional conviction with high confidence. When they diverge (perp buying but spot selling), it signals noise.

### 1.2 Cross-Exchange Basis and Lead-Lag Signals

**Literature basis:** Alexander & Heck (2020) showed perpetual swaps dominate BTC price discovery. The perpetual-spot basis encodes leveraged trader positioning and directional conviction.

**Construction:**

```
basis(t) = (P_perp(t) - P_spot(t)) / P_spot(t)
```

**Feature variants:**

|Feature             |Definition                           |Rationale                               |
|--------------------|-------------------------------------|----------------------------------------|
|`perp_spot_basis`   |Raw basis at current time            |Leveraged positioning signal            |
|`basis_zscore_1h`   |Z-score of basis over trailing 1 hour|Normalized extremity                    |
|`basis_zscore_24h`  |Z-score over trailing 24 hours       |Regime-level extremity                  |
|`basis_momentum_5m` |Δ(basis) over 5 min                  |Basis expansion/contraction rate        |
|`basis_momentum_15m`|Δ(basis) over 15 min                 |Full-window basis change                |
|`basis_accel`       |Δ(basis_momentum_5m)                 |Second derivative — momentum of momentum|

**Interpretation for the model:**

- Sustained positive basis expansion → long-biased positioning building → upward pressure
- Extreme positive basis with stalling momentum → overextended longs → mean-reversion pressure (potential short signal)
- Extreme negative basis → shorts overextended → potential squeeze (upward)

### 1.3 Funding Rate and Liquidation Proximity

**Literature basis:** Coinbase Institutional (2024) concluded funding rates are trailing indicators at short horizons, but extreme values have counter-directional predictive power. Easley et al. (2024) found crypto VPIN levels of 0.45–0.47 (double traditional markets), and Kitvanitphasu et al. (2025) showed VPIN predicts future BTC price jumps. The October 2025 cascade demonstrated the mechanical predictability of liquidation feedback loops.

**Feature variants:**

|Feature                   |Definition                                           |Rationale                                            |
|--------------------------|-----------------------------------------------------|-----------------------------------------------------|
|`funding_rate_predicted`  |Real-time predicted funding rate                     |Current positioning cost                             |
|`funding_rate_zscore`     |Z-score over trailing 7 days                         |Extremity relative to recent history                 |
|`premium_momentum_5m`     |Δ(mark_price - index_price) / index_price over 5 min |Real-time funding pressure direction                 |
|`open_interest_change_5m` |Δ(OI) over 5 min, normalized by 24h avg OI           |Position building/unwinding                          |
|`oi_tofi_divergence`      |OI rising while TOFI near zero                       |Pre-cascade signature: rising OI with weak spot flows|
|`liquidation_intensity_1m`|Volume of liquidations in trailing 1 min / avg volume|Cascade detection                                    |
|`liquidation_imbalance`   |(Long_liq - Short_liq) / (Long_liq + Short_liq)      |Direction of cascade                                 |

**Critical implementation detail for liquidation proximity:**
Liquidation prices cluster at predictable levels based on leverage and entry price. Estimate the distance from current price to dense liquidation zones:

```
liq_proximity_long = (current_price - estimated_long_liq_zone) / current_price
liq_proximity_short = (estimated_short_liq_zone - current_price) / current_price
```

The exact liquidation prices require open interest distribution data (available from some aggregators). As an approximation, use the funding rate extremity and OI concentration as proxies.

### 1.4 Realized Higher Moments

**Literature basis:** Jia, Liu & Yan (2021) found that for 84 cryptocurrencies using 5-minute data, volatility and kurtosis positively predict next-period returns while skewness exhibits negative return predictability. A 2025 MDPI study of 10-minute BTC returns found q-Gaussian distributions provide the best fit across all time scales.

**Feature variants:**

|Feature            |Window|Definition                               |
|-------------------|------|-----------------------------------------|
|`realized_skew_5m` |5 min |Third standardized moment of tick returns|
|`realized_skew_15m`|15 min|Same, over full window                   |
|`realized_skew_1h` |1 hour|Longer-horizon skewness context          |
|`realized_kurt_5m` |5 min |Fourth standardized moment               |
|`realized_kurt_15m`|15 min|Same, over full window                   |
|`realized_kurt_1h` |1 hour|Longer-horizon kurtosis context          |

Compute from log-returns of 1-second mid-price changes within each window. Require minimum 30 observations per window for statistical stability. Negative realized skewness should shift model probability mass toward the left tail.

### 1.5 Existing Features to Retain (Volatility Backbone)

All existing features remain as the volatility estimation backbone. The HAR-RV framework (RV at daily, weekly, monthly lags) provides the foundation, with VPIN, Kyle’s Lambda, spread dynamics, and trade intensity as the microstructure layer. These features are symmetric — they predict magnitude, not direction. The new directional features complement them by breaking the symmetry.

### 1.6 Feature Summary and Expected Dimensionality

|Feature Group                   |Count  |Type   |Signal Class                |
|--------------------------------|-------|-------|----------------------------|
|Existing HAR-RV + microstructure|~44    |Tabular|Symmetric (volatility)      |
|TOFI variants                   |10     |Tabular|Anti-symmetric (directional)|
|Cross-exchange basis            |6      |Tabular|Anti-symmetric (directional)|
|Funding/liquidation/OI          |7      |Tabular|Mixed (regime + directional)|
|Realized higher moments         |6      |Tabular|Mixed (distributional shape)|
|**Total**                       |**~73**|       |                            |

Plus the derived XGBoost model outputs (σ̂, P_BS, uncertainty) fed as features to the neural network = ~76 total input dimensions.

-----

## Phase 2: Regime Detection System (Weeks 4–6)

### Objective

Build a real-time regime classifier that distinguishes mean-reverting, trending, and transitional market states, enabling the trading system to adjust behavior dynamically.

### 2.1 Why Regime Detection Is Make-or-Break

The current system’s failure mode is operating a mean-reversion strategy during trending regimes. The literature confirms this is not a random occurrence — Bitcoin exhibits both behaviors at different times:

- De Nicola (2021, *Ledger*): significant negative first-order autocorrelation at 1–4 hour horizons (genuine mean reversion)
- Wen et al. (2022, *North American Journal of Economics and Finance*): evidence of both intraday momentum and reversal, with dominant regime shifting based on jumps and liquidity
- Corbet & Katsiampa (2020): asymmetric mean reversion — negative returns revert faster, positive returns show persistence

### 2.2 Three-Tier Signal Architecture

**Tier 1: Fast signals (1-second update, 1–60 second lookback)**

These drive real-time quote adjustment:

|Signal                       |Construction                                                                                         |Interpretation                    |
|-----------------------------|-----------------------------------------------------------------------------------------------------|----------------------------------|
|Microprice divergence        |`microprice - midprice`, where microprice = (bid × ask_size + ask × bid_size) / (bid_size + ask_size)|Positive divergence = buy pressure|
|Order book imbalance velocity|Δ(bid_size_L1:L5 / ask_size_L1:L5) per second                                                        |Rate of change of imbalance       |
|Trade arrival asymmetry      |Ratio of buy/sell trade count in trailing 10 seconds                                                 |Immediate flow direction          |

**Tier 2: Medium signals (1-minute update, 1–60 minute lookback)**

These drive regime classification and spread adjustment:

|Signal                          |Construction                                                      |Interpretation                                                  |
|--------------------------------|------------------------------------------------------------------|----------------------------------------------------------------|
|Rolling variance ratio          |VR(q) = Var(q-period returns) / (q × Var(1-period returns)), q = 5|VR < 1 → mean reversion; VR > 1 → trending (Lo & MacKinlay 1988)|
|Rolling Hurst exponent          |Rescaled range (R/S) analysis over trailing 60 1-minute bars      |H < 0.5 → mean reverting; H > 0.5 → trending                    |
|VPIN current level              |Your existing VPIN calculation                                    |Toxicity/informed flow detection                                |
|VPIN rate of change             |Δ(VPIN) over 5 minutes                                            |Rising VPIN = deteriorating conditions                          |
|Autocorrelation of 1-min returns|Rolling lag-1 autocorrelation over trailing 30 bars               |Negative → reverting; Positive → trending                       |

**Tier 3: Slow signals (15-minute update, 1–24 hour lookback)**

These provide regime context:

|Signal                     |Construction                                      |Interpretation                          |
|---------------------------|--------------------------------------------------|----------------------------------------|
|HMM posterior probabilities|4-state non-homogeneous HMM (per Koki et al. 2022)|P(bull), P(bear), P(calm), P(transition)|
|Realized vol regime        |Current 1h RV relative to 24h rolling distribution|Percentile rank of current volatility   |
|Funding rate regime        |funding_rate_zscore > ±2σ                         |Extreme positioning                     |
|OI/volume ratio            |24h OI change / 24h volume                        |Position accumulation vs. turnover      |

### 2.3 Hidden Markov Model Specification

Koki, Leonardos & Piliouras (2022, *Research in International Business and Finance*) found that a 4-state non-homogeneous HMM produced the best one-step-ahead crypto forecasts. Implement this as the primary slow regime signal.

**States:** Bull (trending up), Bear (trending down), Calm (mean-reverting), Transition (high uncertainty)

**Observable:** Vector of [1-min return, realized_vol_5m, tofi_5m, basis_momentum_5m]

**Emission distribution:** Multivariate Student-t per state (to handle fat tails)

**Transition matrix:** Time-varying, conditioned on hour-of-day and funding rate regime. Bitcoin exhibits strong intraday seasonality in regime transitions.

**Training:** Fit via EM algorithm on trailing 30-day rolling window. Refit daily. Use `hmmlearn` or `pomegranate` library in Python.

**Output:** At each minute, compute posterior probability vector [P(bull), P(bear), P(calm), P(transition)]. Feed this as a 4-dimensional input to the main model AND use it directly for quoting behavior.

### 2.4 Composite Regime Score

Combine tiers into a single regime score for quoting decisions:

```
regime_score = w1 * variance_ratio_signal + w2 * hurst_signal + w3 * hmm_trend_prob + w4 * autocorr_signal
```

Where each component is mapped to [-1, +1]: negative = mean-reverting, positive = trending. Calibrate weights via walk-forward optimization on historical P&L.

**Quoting behavior mapping:**

|Regime Score|Interpretation       |Quoting Behavior                                                 |
|------------|---------------------|-----------------------------------------------------------------|
|< -0.3      |Strong mean reversion|Tightest spreads, max size, symmetric quotes                     |
|-0.3 to 0.3 |Uncertain / mixed    |Moderate spreads, moderate size                                  |
|0.3 to 0.6  |Mild trending        |Wide spreads, reduced size, skew quotes in trend direction       |
|> 0.6       |Strong trending      |Widest spreads or cease quoting; consider directional trades only|

Campi & Zabaljauregui (2019/2020, *Applied Mathematical Finance*) proved that when the exact regime is unknown, optimal spreads should be biased wider than full-information spreads. This “regime uncertainty premium” should be added whenever regime confidence is low (e.g., HMM posterior entropy is high).

-----

## Phase 3: Directional Probability Model (Weeks 5–9)

### Objective

Replace symmetric volatility-based pricing with a conditional directional probability estimator. This is the core model change.

### 3.1 Architecture: LSTM-MDN with Student-t Components

Per the project research (Neural Networks for Direct Bitcoin Strike Probability Estimation), the LSTM-MDN architecture is the strongest empirically validated choice at 15-minute horizons. The distributional approach is strictly superior to binary classification because a single model handles all strike levels simultaneously, outputs are automatically monotonic across the strike surface (preventing arbitrage violations), and the model learns richer structure.

**Architecture specification:**

```
Input: [batch, sequence_length=60, features=76]
  ↓
Layer Normalization
  ↓
LSTM (2 layers, hidden_dim=128, dropout=0.2)
  ↓
Dense (128 → 64, ReLU)
  ↓
Dense (64 → K*(1 + 1 + 1 + 1))  [K components × (weight + mean + scale + df)]
  ↓
Output head splits:
  - π_k: softmax over K weights (sum to 1)
  - μ_k: unconstrained (location parameters)
  - σ_k: softplus (positive scale parameters)
  - ν_k: softplus + 2.1 (degrees of freedom > 2 for finite variance)
```

**Key design choices:**

- **K = 4 mixture components**: Sufficient to capture bull, bear, calm, and transition regimes. The LSTM-GTN from 2025 used a similar Gaussian-t mixture structure.
- **Student-t components, not Gaussian**: Bitcoin intraday returns exhibit excess kurtosis 5–15× normal. Typical BTC intraday degrees of freedom fall in ν ≈ 3–6.
- **Sequence length = 60**: 60 one-minute bars = 1 hour of history. Provides sufficient context for regime detection while keeping compute tractable.
- **Input normalization**: Layer normalization, not batch normalization, for financial time series (batch statistics are non-stationary).

**Strike probability computation:**

```python
def compute_strike_probability(pi, mu, sigma, nu, strike_return):
    """
    P(r > ln(K/S)) = Σ_k π_k * (1 - F_t(ln(K/S); μ_k, σ_k, ν_k))
    
    strike_return = ln(K/S) where K is the strike price, S is current price
    For Polymarket up/down: strike_return = 0 (close above current price)
    """
    prob = 0
    for k in range(K):
        # Standardize
        z = (strike_return - mu[k]) / sigma[k]
        # Student-t CDF
        cdf_val = scipy.stats.t.cdf(z, df=nu[k])
        prob += pi[k] * (1 - cdf_val)
    return prob
```

For the Polymarket up/down contract specifically, `strike_return = 0` (the question is whether price closes above the opening price of the 15-minute window), so this simplifies to evaluating the mixture CDF at zero.

### 3.2 XGBoost Integration as Prior

Feed three derived features from the existing XGBoost volatility model:

1. **Raw σ̂₁₅ₘ**: The volatility prediction itself
1. **Black-Scholes implied probability**: P_BS = Φ(d₂) where d₂ = ln(S/K) / (σ̂ · √τ). For up/down contracts, d₂ = 0 / (σ̂ · √(15/1440)) which simplifies to 0, giving P_BS = 0.50 always for ATM. But for non-zero drift estimation, include a drift term: d₂ = (μ̂ - 0.5σ̂²)·τ / (σ̂·√τ)
1. **XGBoost prediction uncertainty**: Use quantile regression XGBoost (fit at τ = 0.1 and τ = 0.9) and feed the spread (q₉₀ - q₁₀) as an uncertainty feature

The neural network then learns corrections to this Black-Scholes baseline — where the BS probability is wrong due to heavy tails, skewness, microstructure effects, or directional flow. This dramatically accelerates convergence per Garcia & Gençay (2000).

### 3.3 Training Configuration

**Loss function: Negative log-likelihood of the Student-t mixture**

```python
def mixture_nll_loss(pi, mu, sigma, nu, target_return):
    """
    NLL = -log(Σ_k π_k * f_t(target; μ_k, σ_k, ν_k))
    """
    log_probs = []
    for k in range(K):
        z = (target_return - mu[k]) / sigma[k]
        log_pdf = scipy.stats.t.logpdf(z, df=nu[k]) - torch.log(sigma[k])
        log_probs.append(torch.log(pi[k]) + log_pdf)
    return -torch.logsumexp(torch.stack(log_probs), dim=0).mean()
```

For the binary classification head (used for calibration evaluation), add a secondary focal loss objective:

```python
def focal_loss(pred_prob, target, gamma_low=5, gamma_high=3):
    """
    FLSD-53: γ = 5 for p ∈ [0, 0.2), γ = 3 for p ∈ [0.2, 1)
    Per Mukhoti et al. (NeurIPS 2020)
    """
    gamma = torch.where(pred_prob < 0.2, gamma_low, gamma_high)
    bce = F.binary_cross_entropy(pred_prob, target, reduction='none')
    focal_weight = (1 - torch.exp(-bce)) ** gamma
    return (focal_weight * bce).mean()
```

**Combined loss:** `L = L_NLL + λ * L_focal` where λ = 0.1 (focal loss as regularizer)

**Optimizer:** AdamW with weight decay 1e-4, learning rate 1e-3 with cosine annealing to 1e-5

**Training data:** 3–6 month sliding window. Bitcoin microstructure evolves rapidly — exchange dominance, fee structures, and participant composition shift, making very old data potentially harmful.

### 3.4 Validation: Combinatorial Purged Cross-Validation

Per the project research, use CPCV (Combinatorial Purged Cross-Validation) for model selection rather than standard walk-forward:

- N = 10 sequential groups
- k = 2 test groups per combination → C(10,2) = 45 backtest paths
- Purge gap: ≥ 15 minutes (matching the label horizon)
- Embargo: 30–60 minutes after each test fold boundary
- Primary metric: Brier score (calibration), secondary: log-loss, AUC

### 3.5 Calibration: Venn-ABERS Predictors

Post-hoc calibration using Venn-ABERS (distribution-free calibration validity):

1. Reserve the most recent 3–7 days as calibration set
1. Apply isotonic regression twice: once assuming test object is class 0, once class 1
1. Produces interval [p₀, p₁] — use the midpoint as the calibrated probability
1. Calibrate separately per moneyness bucket: ATM (|d| < 0.5σ), near-OTM (0.5σ < |d| < 1.5σ), deep-OTM (|d| > 1.5σ)
1. Refresh daily

**Production monitoring:** Track rolling 4-hour Brier score. If it exceeds 2× the training baseline for more than 2 hours, trigger immediate recalibration (re-fit the Venn-ABERS layer on the most recent 3 days).

-----

## Phase 4: Alpha-Integrated Market Making Engine (Weeks 8–12)

### Objective

Implement the Cartea & Wang (2020) alpha-integrated market making framework, adapted for binary options with the Avellaneda-Stoikov modifications.

### 4.1 Core Framework: Cartea & Wang (2020)

The key insight from Cartea & Wang (*International Journal of Theoretical and Applied Finance*, 2020): the market maker uses an alpha signal for three simultaneous purposes:

1. **Minimizing adverse selection** by widening/skewing when informed flow is detected
1. **Executing directional trades** in anticipation of price changes
1. **Managing inventory** through asymmetric quote placement

**Mode switching based on alpha signal strength:**

|Alpha Signal Strength               |Behavior                                          |
|------------------------------------|--------------------------------------------------|
|Near zero (                         |α                                                 |
|Moderate positive (α > threshold_1) |Post only buy limit orders, cancel sell orders    |
|Moderate negative (α < -threshold_1)|Post only sell limit orders, cancel buy orders    |
|Strong positive (α > threshold_2)   |Send market buy orders — full directional trading |
|Strong negative (α < -threshold_2)  |Send market sell orders — full directional trading|

The alpha signal is the directional probability estimate from Phase 3 minus 0.50 (the neutral prior). When the model estimates P(up) = 0.58, α = +0.08.

Threshold calibration: threshold_1 and threshold_2 should be determined by the expected adverse selection cost per trade and the model’s calibrated accuracy at different confidence levels. Start with threshold_1 ≈ 0.03 (53% probability) and threshold_2 ≈ 0.08 (58% probability), then optimize via simulation.

### 4.2 Modified Avellaneda-Stoikov for Binary Options

The standard Avellaneda-Stoikov reservation price is `r(s, q, t) = s − q·γ·σ²·(T−t)`. For binary options, five modifications are required (from the adverse selection research):

**Modification 1: Bounded variance**

Replace constant σ² with `σ²(p) ∝ p(1−p)`:

```python
def reservation_price(p, q, gamma, time_remaining):
    """
    p: current binary option fair value [0, 1]
    q: current inventory (positive = long)
    gamma: risk aversion parameter
    time_remaining: fraction of 15 minutes remaining
    """
    variance = p * (1 - p)
    return p - q * gamma * variance * time_remaining
```

**Modification 2: Gamma-risk premium**

Binary option delta diverges near expiry for ATM contracts. Add explicit gamma premium:

```python
def compute_spread(p, q, gamma, k, time_remaining, gamma_premium_coeff):
    variance = p * (1 - p)
    
    # Standard AS spread
    as_spread = gamma * variance * time_remaining + (2 / gamma) * np.log(1 + gamma / k)
    
    # Binary gamma approximation (increases dramatically near expiry for ATM)
    # Gamma of digital ≈ φ(d₂) / (S·σ·√τ) which → ∞ as τ → 0 for ATM
    if time_remaining > 0.01:  # Avoid division by zero
        binary_gamma = np.exp(-0.5 * ((p - 0.5) / (0.25 * np.sqrt(time_remaining)))**2) / \
                       (np.sqrt(2 * np.pi) * 0.25 * np.sqrt(time_remaining))
    else:
        binary_gamma = 1e6  # Force wide spread
    
    gamma_premium = gamma_premium_coeff * abs(binary_gamma)
    
    total_spread = as_spread + gamma_premium
    return max(total_spread, MIN_SPREAD)
```

**Modification 3: Elevated risk aversion**

With no hedging instrument, γ must be far higher than for hedgeable instruments:

```python
gamma_binary = 1.0 / (max_acceptable_inventory * max(p, 1 - p))
```

**Modification 4: Hard inventory bounds**

```python
Q_max = capital / max(p, 1 - p)

if q >= Q_max:
    cancel_all_bids()  # Stop buying
if q <= -Q_max:
    cancel_all_offers()  # Stop selling
```

**Modification 5: Time-dependent quoting zones**

|Time Remaining|ATM (p ∈ [0.40, 0.60])|Near-ATM (p ∈ [0.20, 0.80])|Far-OTM (p < 0.20 or p > 0.80)    |
|--------------|----------------------|---------------------------|----------------------------------|
|> 10 min      |Normal spread         |Normal spread              |Normal spread                     |
|5–10 min      |2× spread             |1.5× spread                |0.8× spread (tighten for theta)   |
|1–5 min       |4× spread or withdraw |2× spread                  |Normal spread                     |
|< 1 min       |**CEASE QUOTING**     |3× spread or withdraw      |Can continue (exploit convergence)|

The far-from-ATM zone near expiry is the *best* opportunity: outcomes are nearly certain, gamma is low, and the market maker can tighten spreads to harvest time decay as prices converge toward 0 or 1.

### 4.3 Toxicity-Triggered Withdrawal

Monitor VPIN in real time. Easley, López de Prado & O’Hara (2012, *Review of Financial Studies*) showed VPIN anticipated the Flash Crash ~1 hour before the event. In crypto, where VPIN levels are already 2× higher than traditional markets (Easley et al. 2024 found 0.45–0.47 vs. 0.22–0.23), the withdrawal threshold should be correspondingly sensitive.

```python
def check_toxicity(vpin_current, vpin_rolling_mean, vpin_rolling_std):
    vpin_zscore = (vpin_current - vpin_rolling_mean) / vpin_rolling_std
    
    if vpin_zscore > 3.0:
        return "WITHDRAW"       # Cease all quoting
    elif vpin_zscore > 2.0:
        return "WIDEN_MAX"      # Maximum spread, minimum size
    elif vpin_zscore > 1.5:
        return "WIDEN"          # 2× normal spread
    else:
        return "NORMAL"
```

### 4.4 Quote Skewing Based on Alpha

When the directional model has a non-zero signal, skew the bid and ask placement:

```python
def compute_quotes(reservation_price, half_spread, alpha, skew_sensitivity):
    """
    alpha > 0: model expects price to rise → skew bid up (more aggressive buying)
    alpha < 0: model expects price to fall → skew ask down (more aggressive selling)
    """
    skew = alpha * skew_sensitivity
    
    bid = reservation_price - half_spread + skew
    ask = reservation_price + half_spread + skew
    
    # Enforce minimum spread
    if ask - bid < MIN_SPREAD:
        mid = (bid + ask) / 2
        bid = mid - MIN_SPREAD / 2
        ask = mid + MIN_SPREAD / 2
    
    return bid, ask
```

The `skew_sensitivity` parameter controls how aggressively the quotes lean into the directional signal. Start conservatively (skew_sensitivity = 0.5) and increase via walk-forward optimization.

-----

## Phase 5: Ensemble and Production System (Weeks 10–14)

### 5.1 Deep Ensemble

Wrap the LSTM-MDN in a deep ensemble of 3–5 independently trained models (Lakshminarayanan et al. showed this improves both accuracy and calibration significantly):

- Train each model with different random seeds
- Use different random subsets of training data (80% sample per model)
- Average the mixture parameters at inference time: π̄ = (1/M) Σ π_m, etc.
- Ensemble disagreement (variance across models) serves as an additional uncertainty signal — when models disagree, widen spreads

### 5.2 Two-Level Stacking

Per the project research, implement a two-level stacking design:

**Level 0 (base models):**

1. LSTM-MDN (primary, from Phase 3)
1. XGBoost binary classifier (existing model adapted for directional prediction)
1. LightGBM binary classifier (trained on same features, different algorithm)
1. Temporal Fusion Transformer (optional, for interpretability)

**Level 1 (meta-learner):**

- Logistic regression combining out-of-fold predictions from all Level 0 models
- Generate out-of-fold predictions via purged walk-forward splits to prevent information leakage
- The ZestFinance study found stacking 4 XGBoost + 2 neural network models achieved 21% lower AUC variance

### 5.3 Production Update Strategy

**Weekly:** Retrain the global LSTM-MDN ensemble on a 3–6 month sliding window

**Daily:** Recalibrate the Venn-ABERS layer using the most recent 3–7 days

**Real-time:** Monitor rolling 4-hour Brier score. If it exceeds 2× baseline for > 2 hours, trigger immediate Venn-ABERS recalibration.

**Feature drift monitoring:** Track distribution statistics (mean, variance, 5th/95th percentiles) of all input features. Alert if any feature drifts > 3σ from its training distribution. Prefer scale-invariant features (returns, normalized OFI, vol ratios) over raw price levels to minimize drift.

-----

## Phase 6: Simulation, Backtesting, and Deployment (Weeks 12–18)

### 6.1 Simulation Framework

Before live deployment, build a high-fidelity simulator of the Polymarket CLOB:

**Components:**

- Historical replay of Polymarket order book states
- Realistic fill model: limit orders fill only when the market crosses through your price level (not just touches)
- Latency model: add realistic round-trip latency (Polymarket is on-chain, latency is variable and can be 1–15 seconds)
- Slippage model: for market orders, model the impact on a thin binary options book

**Metrics to track in simulation:**

|Metric                                     |Target                                         |Why It Matters                       |
|-------------------------------------------|-----------------------------------------------|-------------------------------------|
|Brier score                                |< 0.22 (better than naive 0.25)                |Calibration quality                  |
|Directional accuracy                       |> 53%                                          |Minimum for profitability after costs|
|Sharpe ratio (15-min windows)              |> 1.5 annualized                               |Risk-adjusted return                 |
|Max drawdown (rolling 24h)                 |< 5% of capital                                |Survival constraint                  |
|Win rate × avg win / (loss rate × avg loss)|> 1.2                                          |Profit factor                        |
|VPIN-conditioned P&L                       |Positive in all VPIN quartiles                 |Not dependent on low-toxicity periods|
|Regime-conditioned P&L                     |Positive in mean-reverting; <0 loss in trending|Regime adaptation working            |

### 6.2 Walk-Forward Backtest Protocol

1. **Training window:** 3 months
1. **Validation window:** 2 weeks (for hyperparameter selection and Venn-ABERS calibration)
1. **Test window:** 2 weeks (out-of-sample P&L evaluation)
1. **Step forward:** 2 weeks
1. **Repeat:** Across all available data

This produces multiple independent test windows covering different regime types. Evaluate whether the strategy is profitable across bull, bear, and ranging markets.

### 6.3 Deployment Sequence

**Stage 1 — Paper trading (2 weeks):** Run the full system in production reading live data and generating quotes, but do not execute. Log all would-be trades. Compare paper P&L to simulation expectations.

**Stage 2 — Micro-live (2 weeks):** Deploy with 5% of intended capital. Verify that actual fills, latency, and slippage match simulation assumptions. Monitor for any systematic deviation.

**Stage 3 — Ramp (4 weeks):** Increase to 25% → 50% → 100% of intended capital, spending 1 week at each level. At each stage, verify that P&L scales linearly (no capacity constraints being hit).

-----

## Appendix A: Key Academic References

|Paper                                              |Relevance                      |Key Finding                                                                         |
|---------------------------------------------------|-------------------------------|------------------------------------------------------------------------------------|
|Cartea & Wang (2020), *IJTAF*                      |Alpha-integrated market making |Optimal hybrid: provide liquidity when α ≈ 0, switch to directional when α is strong|
|Avellaneda & Stoikov (2008), *Quantitative Finance*|Optimal market making framework|Reservation price and spread formulas under inventory risk                          |
|Cont, Kukanov & Stoikov (2014), *JFE*              |Order flow imbalance           |OFI achieves R² ≈ 65–70% for contemporaneous mid-price changes                      |
|Alexander & Heck (2020), *JFS*                     |Cross-exchange price discovery |Perpetual swaps on unregulated exchanges dominate BTC price discovery               |
|Levitt (2004), *Economic Journal*                  |Bookmaker behavior             |Bookmakers take directional risk, not balanced books — 23% higher profit            |
|Easley, López de Prado & O’Hara (2012), *RFS*      |VPIN                           |Flow toxicity metric predicts market stress ~1 hour before Flash Crash              |
|Easley et al. (2024), Cornell working paper        |Crypto VPIN                    |VPIN in crypto (0.45–0.47) is double traditional markets                            |
|Koki, Leonardos & Piliouras (2022), *RIBF*         |HMM for crypto                 |4-state non-homogeneous HMM best for one-step-ahead crypto forecasts                |
|Campi & Zabaljauregui (2019/2020), *AMF*           |Regime-uncertain market making |Optimal spreads wider under regime uncertainty                                      |
|Milionis et al. (2023), *Econometrica*             |Loss-Versus-Rebalancing        |Passive liquidity provision against a moving price is systematically losing         |
|Bieganowski & Ślepaczuk (2025)                     |Crypto OFI ranking             |OFI is #1 SHAP feature across five crypto assets at 1-second frequency              |
|Jha et al. (2020)                                  |Deep learning on LOB           |71% walk-forward accuracy on BTC direction at 2-second horizons                     |
|Lensky & Hao (2023)                                |Order flow image encoding      |CNN-Aggr achieves best RMSPE (0.851) for BTC volatility prediction                  |
|Mukhoti et al. (2020), *NeurIPS*                   |Focal loss calibration         |Focal loss significantly outperforms cross-entropy on calibration                   |
|Lakshminarayanan et al. (2017), *NeurIPS*          |Deep ensembles                 |3–5 model ensembles improve both accuracy and calibration                           |
|Schittenkopf et al. (2000)                         |MDN for options pricing        |MDNs outperform BS and GARCH for density extraction                                 |
|Tetlock (2008), *Econometrica*                     |Prediction market efficiency   |Limit orders in prediction markets have negative expected returns                   |
|Tsang & Yang (2026), arXiv                         |Polymarket microstructure      |Kyle’s Lambda declined by >10× as market matured                                    |
|Kitvanitphasu et al. (2025), *RIBF*                |VPIN and BTC jumps             |VPIN significantly predicts future Bitcoin price jumps                              |

## Appendix B: Technology Stack

|Component        |Library/Tool                         |Notes                                    |
|-----------------|-------------------------------------|-----------------------------------------|
|LSTM-MDN         |PyTorch                              |Native mixed-precision training for speed|
|XGBoost          |xgboost, lightgbm                    |Existing infrastructure                  |
|HMM              |hmmlearn or pomegranate              |4-state with Student-t emissions         |
|Venn-ABERS       |venn-abers (PyPI) or custom isotonic |Daily recalibration                      |
|CPCV             |Custom implementation                |Based on de Prado’s framework            |
|Data pipeline    |Existing Tardis pipeline + extensions|Add Binance perp, Coinbase spot feeds    |
|Student-t CDF    |scipy.stats.t                        |For strike probability computation       |
|Johnson SU       |scipy.stats.johnsonsu                |Analytical fast-path for sanity checking |
|Normalizing flows|nflows or zuko (PyTorch)             |Optional ensemble member                 |
|Backtesting      |Custom event-driven simulator        |Must model CLOB fill mechanics           |

## Appendix C: Development Timeline Summary

|Phase                           |Weeks|Deliverable                                |Dependencies|
|--------------------------------|-----|-------------------------------------------|------------|
|0: Data infrastructure          |1–2  |Binance perp + Coinbase ingestion pipeline |—           |
|1: Directional features         |2–4  |73-feature dataset with directional signals|Phase 0     |
|2: Regime detection             |4–6  |Real-time 3-tier regime classifier         |Phase 1     |
|3: Directional probability model|5–9  |LSTM-MDN trained and calibrated            |Phase 1     |
|4: Market making engine         |8–12 |Alpha-integrated quoting system            |Phases 2, 3 |
|5: Ensemble + production        |10–14|Deep ensemble with Venn-ABERS              |Phase 4     |
|6: Simulation + deployment      |12–18|Live system with staged capital ramp       |Phase 5     |