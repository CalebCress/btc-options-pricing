"""All constants and hyperparameters for the BTC binary options pricing system."""

# --- Data ---
BINANCE_SPOT_SYMBOL = "BTCUSDT"
BINANCE_PERP_SYMBOL = "BTCUSDT"  # Tardis uses same symbol, different exchange id
BAR_SIZES_SEC = [1, 60]

# --- Feature Windows ---
TOFI_WINDOWS_SEC = [60, 300, 900]  # 1m, 5m, 15m
BASIS_ZSCORE_WINDOWS_SEC = [3600, 86400]  # 1h, 24h
BASIS_MOMENTUM_WINDOWS_SEC = [300, 900]  # 5m, 15m
MOMENT_WINDOWS_SEC = [300, 900, 3600]  # 5m, 15m, 1h

# --- VPIN ---
VPIN_BUCKET_SIZE = 50  # trades per bucket
VPIN_N_BUCKETS = 50  # rolling window of buckets

# --- Regime Detection ---
VARIANCE_RATIO_Q = 5
HURST_WINDOW = 60  # 1-min bars
AUTOCORR_WINDOW = 30  # 1-min bars
HMM_N_STATES = 4  # Bull, Bear, Calm, Transition
HMM_TRAIN_DAYS = 30

# --- Model ---
SEQUENCE_LENGTH = 60
HIDDEN_DIM = 128
N_LSTM_LAYERS = 2
K_COMPONENTS = 4
DROPOUT = 0.2
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
FOCAL_LAMBDA = 0.1
ENSEMBLE_SIZE = 5
MAX_EPOCHS = 100
PATIENCE = 10

# --- Market Making ---
ALPHA_THRESHOLD_1 = 0.03  # one-sided quoting
ALPHA_THRESHOLD_2 = 0.08  # directional trading
SKEW_SENSITIVITY = 0.5
MIN_SPREAD = 0.01
GAMMA_PREMIUM_COEFF = 0.1
VPIN_WITHDRAW_ZSCORE = 3.0
VPIN_WIDEN_MAX_ZSCORE = 2.0
VPIN_WIDEN_ZSCORE = 1.5

# --- Backtest ---
TRAIN_MONTHS = 3
VAL_WEEKS = 2
TEST_WEEKS = 2
STEP_WEEKS = 2
PURGE_MINUTES = 15
EMBARGO_MINUTES = 60
LATENCY_RANGE_SEC = (1, 15)  # Polymarket on-chain latency

# --- Targets ---
TARGET_BRIER = 0.22
TARGET_ACCURACY = 0.53
TARGET_SHARPE = 1.5
TARGET_MAX_DD = 0.05
TARGET_PROFIT_FACTOR = 1.2
