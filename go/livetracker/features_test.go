package livetracker

import (
	"math"
	"testing"
	"time"
)

func TestFeatureNames_AllPopulated(t *testing.T) {
	for i, name := range featureNames {
		if name == "" {
			t.Errorf("feature index %d has empty name", i)
		}
	}
}

func TestFeatureComputer_TOFI(t *testing.T) {
	cfg := DefaultConfig()
	fc := NewFeatureComputer(cfg)

	// Push 5 bars with known buy/sell volumes (spot)
	t0 := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	for i := 0; i < 5; i++ {
		fc.OnSpotBar(Bar{
			Timestamp:  t0.Add(time.Duration(i) * time.Minute),
			Open:       100, High: 101, Low: 99, Close: 100,
			BuyVolume:  3, // consistently more buying
			SellVolume: 1,
			Volume:     4,
			TradeCount: 10,
		})
	}

	// TOFI_1m should be (3-1)/(3+1) = 0.5
	tofi1m := fc.current[fSpotTOFI1m]
	if !almostEqual(tofi1m, 0.5, 1e-10) {
		t.Errorf("expected spot_tofi_1m=0.5, got %v", tofi1m)
	}

	// TOFI_5m should also be 0.5 (5 bars of same data, sum=15/5)
	tofi5m := fc.current[fSpotTOFI5m]
	if !almostEqual(tofi5m, 0.5, 1e-10) {
		t.Errorf("expected spot_tofi_5m=0.5, got %v", tofi5m)
	}
}

func TestFeatureComputer_Basis(t *testing.T) {
	cfg := DefaultConfig()
	fc := NewFeatureComputer(cfg)

	t0 := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

	// Push spot bar
	fc.OnSpotBar(Bar{
		Timestamp: t0, Open: 100, High: 101, Low: 99, Close: 100,
		BuyVolume: 1, SellVolume: 1, Volume: 2, TradeCount: 5,
	})
	// Push perp bar at slightly higher price
	fc.OnPerpBar(Bar{
		Timestamp: t0, Open: 100.5, High: 101.5, Low: 99.5, Close: 100.5,
		BuyVolume: 1, SellVolume: 1, Volume: 2, TradeCount: 5,
	})

	// basis = (100.5 - 100) / 100 = 0.005
	basis := fc.current[fBasis]
	if !almostEqual(basis, 0.005, 1e-10) {
		t.Errorf("expected basis=0.005, got %v", basis)
	}
}

func TestFeatureComputer_VPIN(t *testing.T) {
	cfg := DefaultConfig()
	cfg.VPINBucketSize = 10
	cfg.VPINNumBuckets = 3
	fc := NewFeatureComputer(cfg)

	t0 := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

	// Push bars with enough volume to fill buckets
	for i := 0; i < 20; i++ {
		fc.OnSpotBar(Bar{
			Timestamp:  t0.Add(time.Duration(i) * time.Minute),
			Open:       100, High: 101, Low: 99, Close: 100,
			BuyVolume:  8, // heavily buying
			SellVolume: 2,
			Volume:     10,
			TradeCount: 5,
		})
	}

	vpin := fc.current[fVPIN]
	if math.IsNaN(vpin) {
		t.Error("expected VPIN to be computed, got NaN")
	}
	// With buy=8, sell=2: imbalance = |8-2| = 6, total = 10
	// VPIN should be 6/10 = 0.6
	if !almostEqual(vpin, 0.6, 0.01) {
		t.Errorf("expected VPIN≈0.6, got %v", vpin)
	}
}

func TestFeatureComputer_Sequence(t *testing.T) {
	cfg := DefaultConfig()
	cfg.SequenceLength = 3 // short sequence for testing
	fc := NewFeatureComputer(cfg)

	t0 := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

	// Not enough bars yet
	if seq := fc.Sequence(); seq != nil {
		t.Error("expected nil sequence before enough bars")
	}

	for i := 0; i < 3; i++ {
		fc.OnSpotBar(Bar{
			Timestamp:  t0.Add(time.Duration(i) * time.Minute),
			Open:       100, High: 101, Low: 99, Close: float64(100 + i),
			BuyVolume:  1, SellVolume: 1, Volume: 2, TradeCount: 5,
		})
	}

	seq := fc.Sequence()
	if seq == nil {
		t.Fatal("expected non-nil sequence after enough bars")
	}
	expectedLen := 3 * NumFeatures
	if len(seq) != expectedLen {
		t.Errorf("expected sequence length %d, got %d", expectedLen, len(seq))
	}
}

func TestFeatureComputer_KylesLambda(t *testing.T) {
	cfg := DefaultConfig()
	fc := NewFeatureComputer(cfg)

	t0 := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

	// Push 15 bars with correlated returns and signed volume
	for i := 0; i < 15; i++ {
		price := 100.0 + float64(i)*0.1
		fc.OnSpotBar(Bar{
			Timestamp:  t0.Add(time.Duration(i) * time.Minute),
			Open:       price, High: price + 0.5, Low: price - 0.5, Close: price,
			BuyVolume:  float64(5 + i), // increasing buy volume with price
			SellVolume: 3,
			Volume:     float64(8 + i),
			TradeCount: 10,
		})
	}

	kl := fc.current[fKylesLambda]
	if math.IsNaN(kl) {
		t.Error("expected Kyle's Lambda to be computed, got NaN")
	}
	// With positive correlation between returns and signed volume, lambda should be positive
	if kl <= 0 {
		t.Errorf("expected positive Kyle's Lambda, got %v", kl)
	}
}
