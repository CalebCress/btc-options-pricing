package livetracker

import (
	"math"
)

// Feature indices — ordered to match the Python feature vector.
const (
	// TOFI (9 features)
	fSpotTOFI1m = iota
	fSpotTOFI5m
	fSpotTOFI15m
	fSpotTOFIMomentum
	fPerpTOFI1m
	fPerpTOFI5m
	fPerpTOFI15m
	fPerpTOFIMomentum
	fTOFIDivergence

	// Basis (6 features)
	fBasis
	fBasisZScore1h
	fBasisZScore24h
	fBasisMomentum5m
	fBasisMomentum15m
	fBasisAccel

	// Funding/OI (7 features)
	fFundingRate
	fFundingRateZScore
	fPremiumMomentum5m
	fOIChange5m
	fOITOFIDivergence
	fLiqIntensity1m
	fLiqImbalance

	// Realized Moments (6 features)
	fRealizedSkew5m
	fRealizedSkew15m
	fRealizedSkew60m
	fRealizedKurt5m
	fRealizedKurt15m
	fRealizedKurt60m

	// Microstructure - HAR-RV (6 features)
	fRV5m
	fRV15m
	fRV1h
	fRVDaily
	fRVWeekly
	fRVMonthly

	// Spread (3 features)
	fHLSpread
	fHLSpread5m
	fHLSpread15m

	// Trade intensity (3 features)
	fTradeCountZScore
	fVolumeZScore
	fBuyRatio

	// VPIN + Kyle's Lambda (2 features)
	fVPIN
	fKylesLambda

	// Prices (3, used internally and by model)
	fSpotClose
	fPerpClose
	fLogReturn1m

	// XGBoost-derived (3 features, filled by inference server)
	fXGBProb
	fXGBUncertainty
	fBSImpliedProb

	numFeatureSlots // must equal NumFeatures (76)
)

// featureNames maps indices to human-readable names.
var featureNames = [NumFeatures]string{
	fSpotTOFI1m: "spot_tofi_1m", fSpotTOFI5m: "spot_tofi_5m", fSpotTOFI15m: "spot_tofi_15m",
	fSpotTOFIMomentum: "spot_tofi_momentum",
	fPerpTOFI1m: "perp_tofi_1m", fPerpTOFI5m: "perp_tofi_5m", fPerpTOFI15m: "perp_tofi_15m",
	fPerpTOFIMomentum: "perp_tofi_momentum",
	fTOFIDivergence: "tofi_perp_spot_divergence",

	fBasis: "perp_spot_basis", fBasisZScore1h: "basis_zscore_1h", fBasisZScore24h: "basis_zscore_24h",
	fBasisMomentum5m: "basis_momentum_5m", fBasisMomentum15m: "basis_momentum_15m",
	fBasisAccel: "basis_accel",

	fFundingRate: "funding_rate_predicted", fFundingRateZScore: "funding_rate_zscore",
	fPremiumMomentum5m: "premium_momentum_5m", fOIChange5m: "oi_change_5m",
	fOITOFIDivergence: "oi_tofi_divergence", fLiqIntensity1m: "liquidation_intensity_1m",
	fLiqImbalance: "liquidation_imbalance",

	fRealizedSkew5m: "realized_skew_5m", fRealizedSkew15m: "realized_skew_15m",
	fRealizedSkew60m: "realized_skew_60m",
	fRealizedKurt5m: "realized_kurt_5m", fRealizedKurt15m: "realized_kurt_15m",
	fRealizedKurt60m: "realized_kurt_60m",

	fRV5m: "rv_5m", fRV15m: "rv_15m", fRV1h: "rv_1h",
	fRVDaily: "rv_daily", fRVWeekly: "rv_weekly", fRVMonthly: "rv_monthly",

	fHLSpread: "hl_spread", fHLSpread5m: "hl_spread_5m", fHLSpread15m: "hl_spread_15m",
	fTradeCountZScore: "trade_count_zscore", fVolumeZScore: "volume_zscore",
	fBuyRatio: "buy_ratio",
	fVPIN: "vpin", fKylesLambda: "kyles_lambda",

	fSpotClose: "spot_close", fPerpClose: "perp_close", fLogReturn1m: "log_return_1m",

	fXGBProb: "xgb_prob", fXGBUncertainty: "xgb_uncertainty", fBSImpliedProb: "bs_implied_prob",
}

func init() {
	if numFeatureSlots != NumFeatures {
		panic("feature slot count mismatch: update NumFeatures or feature indices")
	}
}

// tofiState tracks rolling buy/sell volume sums for TOFI computation.
type tofiState struct {
	buySum  []*RollingSum // one per window
	sellSum []*RollingSum
	prev5m  float64 // previous 5m TOFI for momentum
}

func newTOFIState(windows []int) *tofiState {
	ts := &tofiState{
		buySum:  make([]*RollingSum, len(windows)),
		sellSum: make([]*RollingSum, len(windows)),
		prev5m:  math.NaN(),
	}
	for i, w := range windows {
		ts.buySum[i] = NewRollingSum(w)
		ts.sellSum[i] = NewRollingSum(w)
	}
	return ts
}

func (ts *tofiState) update(buyVol, sellVol float64) {
	for i := range ts.buySum {
		ts.buySum[i].Push(buyVol)
		ts.sellSum[i].Push(sellVol)
	}
}

func (ts *tofiState) tofi(windowIdx int) float64 {
	b := ts.buySum[windowIdx].Sum()
	s := ts.sellSum[windowIdx].Sum()
	total := b + s
	if total == 0 {
		return math.NaN()
	}
	return (b - s) / total
}

// FeatureComputer maintains all rolling state and computes the 73 base features
// (+ 3 XGBoost-derived features populated externally) as 1-min bars arrive.
type FeatureComputer struct {
	cfg Config

	// TOFI state
	spotTOFI *tofiState
	perpTOFI *tofiState

	// Basis
	basisStats1h   *RollingStats
	basisStats24h  *RollingStats
	basisDiff      *DiffBuffer
	basisMom5mDiff *DiffBuffer // for acceleration
	lastSpotClose  float64
	lastPerpClose  float64
	prevSpotClose  float64 // for log return

	// Funding/OI
	fundingStats  *RollingStats
	premiumDiff   *DiffBuffer
	oiDiff        *DiffBuffer
	oiStats24h    *RollingStats
	liqStats24h   *RollingStats

	// Last ticker values
	lastFundingRate  float64
	lastOI           float64
	lastMarkPrice    float64
	lastIndexPrice   float64
	lastLiqAmount    float64
	lastLiqLong      float64
	lastLiqShort     float64

	// Realized moments (from 1-second bars)
	secLogReturns [3]*RollingStats // 5m(300), 15m(900), 60m(3600) of standardized^3 and ^4
	secRawStats   [3]*RollingStats // mean/std of log returns per window
	// We maintain separate buffers for skew and kurt running sums
	secSkew [3]*RollingSum
	secKurt [3]*RollingSum
	secStdz [3]*RingBuffer // buffer of standardized returns for skew/kurt
	prevSecClose float64

	// HAR-RV
	rv1mSq      float64 // current bar's log return squared
	rvSum5m     *RollingSum
	rvSum15m    *RollingSum
	rvSum1h     *RollingSum
	rvSumDaily  *RollingSum
	rvSumWeekly *RollingSum
	rvSumMonthly *RollingSum

	// Spread
	hlSpreadBuf5m  *RollingSum
	hlSpreadBuf15m *RollingSum

	// Trade intensity
	tradeCountStats *RollingStats
	volumeStats     *RollingStats

	// VPIN
	vpin *VPINCalculator

	// Kyle's Lambda
	logRetBuf   *RingBuffer
	signedVolBuf *RingBuffer
	kylesWindow int

	// Sequence buffer (60 bars × NumFeatures)
	seqBuf   *RingBuffer // stores flat features; each "element" is actually a bar index
	seqSlots [][NumFeatures]float64
	seqHead  int
	seqCount int

	// Current feature values
	current [NumFeatures]float64
	barCount int
}

// NewFeatureComputer creates a feature computer with the given config.
func NewFeatureComputer(cfg Config) *FeatureComputer {
	fc := &FeatureComputer{
		cfg: cfg,

		spotTOFI: newTOFIState(cfg.TOFIWindows),
		perpTOFI: newTOFIState(cfg.TOFIWindows),

		basisStats1h:  NewRollingStats(cfg.BasisZScoreWindows[0]),
		basisStats24h: NewRollingStats(cfg.BasisZScoreWindows[1]),
		basisDiff:     NewDiffBuffer(15),
		basisMom5mDiff: NewDiffBuffer(1),

		fundingStats: NewRollingStats(cfg.FundingZScoreWindow),
		premiumDiff:  NewDiffBuffer(5),
		oiDiff:       NewDiffBuffer(5),
		oiStats24h:   NewRollingStats(1440),
		liqStats24h:  NewRollingStats(1440),

		rvSum5m:      NewRollingSum(5),
		rvSum15m:     NewRollingSum(15),
		rvSum1h:      NewRollingSum(60),
		rvSumDaily:   NewRollingSum(1440),
		rvSumWeekly:  NewRollingSum(10080),
		rvSumMonthly: NewRollingSum(43200),

		hlSpreadBuf5m:  NewRollingSum(5),
		hlSpreadBuf15m: NewRollingSum(15),

		tradeCountStats: NewRollingStats(60),
		volumeStats:     NewRollingStats(60),

		vpin: NewVPINCalculator(cfg.VPINBucketSize, cfg.VPINNumBuckets),

		logRetBuf:    NewRingBuffer(60),
		signedVolBuf: NewRingBuffer(60),
		kylesWindow:  60,

		seqSlots:  make([][NumFeatures]float64, cfg.SequenceLength),
		prevSecClose: math.NaN(),
	}

	// Initialize realized moment buffers for each window
	for i, wSec := range cfg.MomentWindowsSec {
		fc.secRawStats[i] = NewRollingStats(wSec)
		fc.secSkew[i] = NewRollingSum(wSec)
		fc.secKurt[i] = NewRollingSum(wSec)
		fc.secStdz[i] = NewRingBuffer(wSec)
	}

	return fc
}

// OnSpotBar processes a new 1-minute spot bar.
func (fc *FeatureComputer) OnSpotBar(b Bar) {
	fc.prevSpotClose = fc.lastSpotClose
	fc.lastSpotClose = b.Close

	// TOFI
	fc.spotTOFI.update(b.BuyVolume, b.SellVolume)
	fc.current[fSpotTOFI1m] = fc.spotTOFI.tofi(0)
	fc.current[fSpotTOFI5m] = fc.spotTOFI.tofi(1)
	fc.current[fSpotTOFI15m] = fc.spotTOFI.tofi(2)

	cur5m := fc.spotTOFI.tofi(1)
	fc.current[fSpotTOFIMomentum] = cur5m - fc.spotTOFI.prev5m
	fc.spotTOFI.prev5m = cur5m

	// Log return
	if fc.prevSpotClose > 0 {
		fc.current[fLogReturn1m] = math.Log(b.Close / fc.prevSpotClose)
	} else {
		fc.current[fLogReturn1m] = 0
	}

	// HAR-RV (squared log return)
	rv := fc.current[fLogReturn1m] * fc.current[fLogReturn1m]
	fc.rvSum5m.Push(rv)
	fc.rvSum15m.Push(rv)
	fc.rvSum1h.Push(rv)
	fc.rvSumDaily.Push(rv)
	fc.rvSumWeekly.Push(rv)
	fc.rvSumMonthly.Push(rv)
	fc.current[fRV5m] = fc.rvSum5m.Sum()
	fc.current[fRV15m] = fc.rvSum15m.Sum()
	fc.current[fRV1h] = fc.rvSum1h.Sum()
	fc.current[fRVDaily] = fc.rvSumDaily.Sum()
	fc.current[fRVWeekly] = fc.rvSumWeekly.Sum()
	fc.current[fRVMonthly] = fc.rvSumMonthly.Sum()

	// Spread features
	hlSpread := 0.0
	if b.Close > 0 {
		hlSpread = (b.High - b.Low) / b.Close
	}
	fc.current[fHLSpread] = hlSpread
	fc.hlSpreadBuf5m.Push(hlSpread)
	fc.hlSpreadBuf15m.Push(hlSpread)
	fc.current[fHLSpread5m] = fc.hlSpreadBuf5m.Sum() / float64(fc.hlSpreadBuf5m.Len())
	fc.current[fHLSpread15m] = fc.hlSpreadBuf15m.Sum() / float64(fc.hlSpreadBuf15m.Len())

	// Trade intensity
	fc.tradeCountStats.Push(float64(b.TradeCount))
	fc.volumeStats.Push(b.Volume)
	fc.current[fTradeCountZScore] = fc.tradeCountStats.ZScore(float64(b.TradeCount))
	fc.current[fVolumeZScore] = fc.volumeStats.ZScore(b.Volume)
	if b.Volume > 0 {
		fc.current[fBuyRatio] = b.BuyVolume / b.Volume
	} else {
		fc.current[fBuyRatio] = math.NaN()
	}

	// VPIN
	fc.vpin.OnBar(b.BuyVolume, b.SellVolume)
	fc.current[fVPIN] = fc.vpin.Value()

	// Kyle's Lambda (rolling regression of log_ret vs signed_vol)
	signedVol := b.BuyVolume - b.SellVolume
	fc.logRetBuf.Push(fc.current[fLogReturn1m])
	fc.signedVolBuf.Push(signedVol)
	fc.current[fKylesLambda] = fc.computeKylesLambda()

	fc.current[fSpotClose] = b.Close

	// Update basis if we have perp data
	fc.updateBasis()

	// Store in sequence buffer
	fc.pushSequence()
}

// OnPerpBar processes a new 1-minute perp bar.
func (fc *FeatureComputer) OnPerpBar(b Bar) {
	fc.lastPerpClose = b.Close

	fc.perpTOFI.update(b.BuyVolume, b.SellVolume)
	fc.current[fPerpTOFI1m] = fc.perpTOFI.tofi(0)
	fc.current[fPerpTOFI5m] = fc.perpTOFI.tofi(1)
	fc.current[fPerpTOFI15m] = fc.perpTOFI.tofi(2)

	cur5m := fc.perpTOFI.tofi(1)
	fc.current[fPerpTOFIMomentum] = cur5m - fc.perpTOFI.prev5m
	fc.perpTOFI.prev5m = cur5m

	// TOFI divergence
	fc.current[fTOFIDivergence] = fc.current[fPerpTOFI5m] - fc.current[fSpotTOFI5m]

	fc.current[fPerpClose] = b.Close

	fc.updateBasis()
}

// OnSecondBar processes a 1-second spot bar for realized moment computation.
func (fc *FeatureComputer) OnSecondBar(b Bar) {
	if math.IsNaN(fc.prevSecClose) || fc.prevSecClose == 0 {
		fc.prevSecClose = b.Close
		return
	}
	logRet := math.Log(b.Close / fc.prevSecClose)
	fc.prevSecClose = b.Close

	for i := range fc.cfg.MomentWindowsSec {
		fc.secRawStats[i].Push(logRet)

		mean := fc.secRawStats[i].Mean()
		std := fc.secRawStats[i].Std()
		stdz := 0.0
		if !math.IsNaN(std) && std > 0 {
			stdz = (logRet - mean) / std
		}

		// Push standardized value and track its cube and 4th power
		oldStdz, didEvict := fc.secStdz[i].Push(stdz)
		fc.secSkew[i].Push(stdz * stdz * stdz)
		fc.secKurt[i].Push(stdz * stdz * stdz * stdz)

		// Subtract evicted values (handled by RollingSum internally)
		_ = oldStdz
		_ = didEvict

		n := fc.secStdz[i].Len()
		if n >= 30 {
			fc.current[fRealizedSkew5m+i] = fc.secSkew[i].Sum() / float64(n)
			fc.current[fRealizedKurt5m+3+i] = fc.secKurt[i].Sum() / float64(n)
		}
	}
}

// OnTicker processes a derivative ticker update.
func (fc *FeatureComputer) OnTicker(t TickerUpdate) {
	fc.lastFundingRate = t.FundingRate
	fc.lastOI = t.OpenInterest
	fc.lastMarkPrice = t.MarkPrice
	fc.lastIndexPrice = t.IndexPrice

	// Funding rate
	fc.current[fFundingRate] = t.FundingRate
	fc.fundingStats.Push(t.FundingRate)
	fc.current[fFundingRateZScore] = fc.fundingStats.ZScore(t.FundingRate)

	// Premium momentum
	if t.IndexPrice > 0 {
		premium := (t.MarkPrice - t.IndexPrice) / t.IndexPrice
		fc.premiumDiff.Push(premium)
		fc.current[fPremiumMomentum5m] = fc.premiumDiff.Diff(5)
	}

	// OI change
	fc.oiDiff.Push(t.OpenInterest)
	fc.oiStats24h.Push(t.OpenInterest)
	oiAvg := fc.oiStats24h.Mean()
	if !math.IsNaN(oiAvg) && oiAvg > 0 {
		fc.current[fOIChange5m] = fc.oiDiff.Diff(5) / oiAvg
	}

	// OI-TOFI divergence
	perpTOFI5m := fc.current[fPerpTOFI5m]
	oiChange := fc.current[fOIChange5m]
	if !math.IsNaN(oiChange) && !math.IsNaN(perpTOFI5m) {
		fc.current[fOITOFIDivergence] = math.Abs(oiChange) * (1 - math.Abs(perpTOFI5m))
	}
}

// OnLiquidation processes a liquidation event.
func (fc *FeatureComputer) OnLiquidation(l LiquidationUpdate) {
	fc.lastLiqAmount += l.Amount
	if l.Side == "sell" { // long liquidated
		fc.lastLiqLong += l.Amount
	} else {
		fc.lastLiqShort += l.Amount
	}

	fc.liqStats24h.Push(l.Amount)
	avgLiq := fc.liqStats24h.Mean()
	if !math.IsNaN(avgLiq) && avgLiq > 0 {
		fc.current[fLiqIntensity1m] = l.Amount / avgLiq
	}

	total := fc.lastLiqLong + fc.lastLiqShort
	if total > 0 {
		fc.current[fLiqImbalance] = (fc.lastLiqLong - fc.lastLiqShort) / total
	}
}

// SetXGBFeatures sets the XGBoost-derived features from the inference response.
func (fc *FeatureComputer) SetXGBFeatures(xgbProb, xgbUncertainty, bsImpliedProb float64) {
	fc.current[fXGBProb] = xgbProb
	fc.current[fXGBUncertainty] = xgbUncertainty
	fc.current[fBSImpliedProb] = bsImpliedProb
}

// Snapshot returns the current feature vector with names.
func (fc *FeatureComputer) Snapshot() Features {
	return Features{
		Values: fc.current,
		Names:  featureNames,
		Ready:  fc.barCount >= fc.cfg.SequenceLength,
	}
}

// Sequence returns the flat feature sequence for model inference.
// Returns nil if not enough bars have been accumulated.
func (fc *FeatureComputer) Sequence() []float32 {
	if fc.seqCount < fc.cfg.SequenceLength {
		return nil
	}
	out := make([]float32, fc.cfg.SequenceLength*NumFeatures)
	for i := 0; i < fc.cfg.SequenceLength; i++ {
		idx := (fc.seqHead - fc.cfg.SequenceLength + i + len(fc.seqSlots)) % len(fc.seqSlots)
		for j := 0; j < NumFeatures; j++ {
			out[i*NumFeatures+j] = float32(fc.seqSlots[idx][j])
		}
	}
	return out
}

// BarCount returns the number of 1-minute spot bars processed.
func (fc *FeatureComputer) BarCount() int {
	return fc.barCount
}

// --- internal helpers ---

func (fc *FeatureComputer) updateBasis() {
	if fc.lastSpotClose == 0 || fc.lastPerpClose == 0 {
		return
	}
	basis := (fc.lastPerpClose - fc.lastSpotClose) / fc.lastSpotClose
	fc.current[fBasis] = basis

	fc.basisStats1h.Push(basis)
	fc.basisStats24h.Push(basis)
	fc.current[fBasisZScore1h] = fc.basisStats1h.ZScore(basis)
	fc.current[fBasisZScore24h] = fc.basisStats24h.ZScore(basis)

	fc.basisDiff.Push(basis)
	mom5m := fc.basisDiff.Diff(5)
	fc.current[fBasisMomentum5m] = mom5m
	fc.current[fBasisMomentum15m] = fc.basisDiff.Diff(15)

	fc.basisMom5mDiff.Push(mom5m)
	fc.current[fBasisAccel] = fc.basisMom5mDiff.Diff(1)
}

func (fc *FeatureComputer) pushSequence() {
	fc.seqSlots[fc.seqHead] = fc.current
	fc.seqHead = (fc.seqHead + 1) % len(fc.seqSlots)
	if fc.seqCount < len(fc.seqSlots) {
		fc.seqCount++
	}
	fc.barCount++
}

func (fc *FeatureComputer) computeKylesLambda() float64 {
	n := fc.logRetBuf.Len()
	if n < 10 {
		return math.NaN()
	}

	// Compute rolling covariance and variance.
	var sumLR, sumSV, sumLR2, sumSV2, sumProd float64
	for i := 0; i < n; i++ {
		lr := fc.logRetBuf.Get(i)
		sv := fc.signedVolBuf.Get(i)
		sumLR += lr
		sumSV += sv
		sumLR2 += lr * lr
		sumSV2 += sv * sv
		sumProd += lr * sv
	}
	fn := float64(n)
	meanLR := sumLR / fn
	meanSV := sumSV / fn
	cov := sumProd/fn - meanLR*meanSV
	varSV := sumSV2/fn - meanSV*meanSV

	if varSV == 0 {
		return math.NaN()
	}
	return cov / varSV
}
