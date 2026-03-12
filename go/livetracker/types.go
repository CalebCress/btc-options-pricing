package livetracker

import "time"

// Source identifies where a trade originated.
type Source int

const (
	SourceSpot Source = iota
	SourcePerp
)

// Trade represents a single tick-level trade from an exchange.
type Trade struct {
	Timestamp    time.Time
	Price        float64
	Amount       float64
	IsBuyerMaker bool   // true = taker sold (sell side), false = taker bought (buy side)
	Source       Source
}

// Bar represents an OHLCV bar aggregated from trades.
type Bar struct {
	Timestamp  time.Time
	Open       float64
	High       float64
	Low        float64
	Close      float64
	Volume     float64
	BuyVolume  float64
	SellVolume float64
	TradeCount int
	VWAP       float64
}

// TickerUpdate carries derivative instrument metadata from the exchange.
type TickerUpdate struct {
	Timestamp    time.Time
	FundingRate  float64
	OpenInterest float64
	MarkPrice    float64
	IndexPrice   float64
}

// LiquidationUpdate carries a single liquidation event.
type LiquidationUpdate struct {
	Timestamp time.Time
	Side      string  // "buy" (short liquidated) or "sell" (long liquidated)
	Price     float64
	Amount    float64
}

// NumFeatures is the total number of features in the feature vector.
// 9 TOFI + 6 basis + 7 funding/OI + 6 realized moments + 6 HAR-RV +
// 3 spread + 3 trade intensity + 2 VPIN/Kyle + 3 tier1 regime +
// 4 tier2 regime + 4 HMM posteriors + 1 regime score + 3 prices + 3 XGBoost = 60
const NumFeatures = 60

// PriceTarget holds the latest model prediction.
type PriceTarget struct {
	ProbUp       float64    // calibrated P(return > 0 in next 15 min)
	Alpha        float64    // ProbUp - 0.5
	MixturePi    [4]float64 // mixture weights
	MixtureMu    [4]float64 // location params
	MixtureSigma [4]float64 // scale params
	MixtureNu    [4]float64 // degrees of freedom
	Timestamp    time.Time  // when inference was run
	Stale        bool       // true if >2 min since last inference
}

// Features holds a snapshot of the current feature vector.
type Features struct {
	Values    [NumFeatures]float64
	Names     [NumFeatures]string
	Timestamp time.Time
	Ready     bool // false until 60 1-min bars have been accumulated
}
