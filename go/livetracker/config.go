package livetracker

import "time"

// Config holds all parameters for the live tracking engine.
type Config struct {
	// InferenceAddr is the gRPC address of the Python inference server (e.g. "localhost:50051").
	InferenceAddr string

	// SequenceLength is the number of 1-min bars in the model input sequence.
	SequenceLength int

	// KComponents is the number of Student-t mixture components.
	KComponents int

	// TOFI rolling windows in number of 1-min bars.
	TOFIWindows []int

	// Basis z-score rolling windows in number of 1-min bars.
	BasisZScoreWindows []int

	// Basis momentum diff windows in number of 1-min bars.
	BasisMomentumWindows []int

	// Realized moment windows in seconds (computed from 1-sec bars).
	MomentWindowsSec []int

	// VPIN parameters.
	VPINBucketSize int
	VPINNumBuckets int

	// Funding rate z-score window in 1-min bars (7 days = 10080).
	FundingZScoreWindow int

	// StaleThreshold is how long before a prediction is marked stale.
	StaleThreshold time.Duration

	// Regime detection parameters.
	VarianceRatioQ      int
	VarianceRatioWindow int
	HurstWindow         int
	AutocorrWindow      int
	HMMNStates          int

	// HMMParams are the pre-trained HMM parameters (nil if no HMM).
	HMMParams *HMMParams
}

// DefaultConfig returns configuration matching the Python config.py.
func DefaultConfig() Config {
	return Config{
		InferenceAddr:        "localhost:50051",
		SequenceLength:       60,
		KComponents:          4,
		TOFIWindows:          []int{1, 5, 15},
		BasisZScoreWindows:   []int{60, 1440},
		BasisMomentumWindows: []int{5, 15},
		MomentWindowsSec:     []int{300, 900, 3600},
		VPINBucketSize:       50,
		VPINNumBuckets:       50,
		FundingZScoreWindow:  10080,
		StaleThreshold:       2 * time.Minute,

		VarianceRatioQ:      5,
		VarianceRatioWindow: 60,
		HurstWindow:         60,
		AutocorrWindow:      30,
		HMMNStates:          4,
	}
}
