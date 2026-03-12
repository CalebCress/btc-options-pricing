package livetracker

import (
	"math"
	"testing"
	"time"
)

func TestEngine_FeaturesOnly(t *testing.T) {
	cfg := DefaultConfig()
	cfg.InferenceAddr = "" // no inference server
	cfg.SequenceLength = 5

	e, err := New(cfg)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
	defer e.Close()

	t0 := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

	// Feed trades to fill 5 bars
	for bar := 0; bar < 6; bar++ {
		for tick := 0; tick < 10; tick++ {
			ts := t0.Add(time.Duration(bar)*time.Minute + time.Duration(tick)*time.Second)
			e.OnTrade(Trade{
				Timestamp:    ts,
				Price:        float64(50000 + bar*10 + tick),
				Amount:       0.1,
				IsBuyerMaker: tick%3 == 0, // every 3rd trade is seller-initiated
				Source:       SourceSpot,
			})
			e.OnTrade(Trade{
				Timestamp:    ts,
				Price:        float64(50001 + bar*10 + tick),
				Amount:       0.2,
				IsBuyerMaker: tick%2 == 0,
				Source:       SourcePerp,
			})
		}
	}

	f := e.Features()
	if !f.Ready {
		t.Error("expected features to be ready after 5+ bars")
	}

	// Check that some features are non-zero
	if f.Values[fSpotTOFI1m] == 0 && f.Values[fSpotTOFI5m] == 0 {
		t.Error("expected non-zero TOFI features")
	}
	if f.Values[fBasis] == 0 {
		t.Error("expected non-zero basis")
	}

	// PriceTarget should be stale (no inference server)
	pt := e.PriceTarget()
	if !pt.Stale {
		t.Error("expected stale price target without inference server")
	}
}

func TestEngine_OnTicker(t *testing.T) {
	cfg := DefaultConfig()
	cfg.InferenceAddr = ""

	e, err := New(cfg)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
	defer e.Close()

	e.OnTicker(TickerUpdate{
		Timestamp:    time.Now(),
		FundingRate:  0.0001,
		OpenInterest: 1e9,
		MarkPrice:    50000,
		IndexPrice:   49990,
	})

	f := e.Features()
	if f.Values[fFundingRate] != 0.0001 {
		t.Errorf("expected funding_rate=0.0001, got %v", f.Values[fFundingRate])
	}
}

func TestEngine_PriceStrike(t *testing.T) {
	cfg := DefaultConfig()
	cfg.InferenceAddr = ""

	e, err := New(cfg)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
	defer e.Close()

	// No prediction yet
	p := e.PriceStrike(0)
	if !math.IsNaN(p) {
		t.Errorf("expected NaN before any prediction, got %v", p)
	}

	// Manually set a target to test strike pricing
	e.mu.Lock()
	e.target = PriceTarget{
		ProbUp:       0.55,
		Alpha:        0.05,
		MixturePi:    [4]float64{0.25, 0.25, 0.25, 0.25},
		MixtureMu:    [4]float64{0.001, -0.001, 0.0005, -0.0005},
		MixtureSigma: [4]float64{0.01, 0.01, 0.005, 0.005},
		MixtureNu:    [4]float64{5, 5, 5, 5},
		Timestamp:    time.Now(),
	}
	e.mu.Unlock()

	// P(return > 0) should be close to 0.5 (symmetric-ish mixture)
	pATM := e.PriceStrike(0)
	if pATM < 0.3 || pATM > 0.7 {
		t.Errorf("expected P(up) near 0.5, got %v", pATM)
	}

	// P(return > very negative) should be close to 1
	pDeepITM := e.PriceStrike(-0.1)
	if pDeepITM < 0.9 {
		t.Errorf("expected P(r > -0.1) near 1, got %v", pDeepITM)
	}

	// P(return > very positive) should be close to 0
	pDeepOTM := e.PriceStrike(0.1)
	if pDeepOTM > 0.1 {
		t.Errorf("expected P(r > 0.1) near 0, got %v", pDeepOTM)
	}
}

func TestEngine_Concurrent(t *testing.T) {
	cfg := DefaultConfig()
	cfg.InferenceAddr = ""

	e, err := New(cfg)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
	defer e.Close()

	done := make(chan struct{})

	// Writer goroutine
	go func() {
		t0 := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
		for i := 0; i < 1000; i++ {
			ts := t0.Add(time.Duration(i) * time.Millisecond * 100)
			e.OnTrade(Trade{
				Timestamp: ts, Price: 50000, Amount: 0.1,
				Source: SourceSpot,
			})
		}
		close(done)
	}()

	// Reader goroutine
	for i := 0; i < 100; i++ {
		_ = e.Features()
		_ = e.PriceTarget()
	}

	<-done
}
