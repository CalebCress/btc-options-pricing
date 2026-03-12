package livetracker

import (
	"context"
	"log"
	"math"
	"sync"
	"time"

	"github.com/CalebCress/btc-options-pricing/go/livetracker/internal/studentt"
)

// Engine is the main entry point for live feature tracking and prediction.
// It is safe for concurrent use from multiple goroutines.
type Engine struct {
	cfg Config

	mu       sync.RWMutex
	features *FeatureComputer

	// Bar aggregators
	spot1m *BarAggregator
	spot1s *BarAggregator
	perp1m *BarAggregator

	// Inference
	inference *InferenceClient
	target    PriceTarget

	// Background inference
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
	inferCh chan struct{} // signals that new data is ready for inference
}

// New creates a new Engine. If cfg.InferenceAddr is non-empty, it connects to
// the Python inference server. Pass an empty InferenceAddr to run without inference
// (features-only mode).
func New(cfg Config) (*Engine, error) {
	ctx, cancel := context.WithCancel(context.Background())

	e := &Engine{
		cfg:      cfg,
		features: NewFeatureComputer(cfg),
		ctx:      ctx,
		cancel:   cancel,
		inferCh:  make(chan struct{}, 1),
	}

	// Wire bar aggregators to feature computer
	e.spot1m = NewBarAggregator(time.Minute, func(b Bar) {
		e.features.OnSpotBar(b)
		// Signal inference
		select {
		case e.inferCh <- struct{}{}:
		default:
		}
	})
	e.spot1s = NewBarAggregator(time.Second, func(b Bar) {
		e.features.OnSecondBar(b)
	})
	e.perp1m = NewBarAggregator(time.Minute, func(b Bar) {
		e.features.OnPerpBar(b)
	})

	// Connect to inference server if configured
	if cfg.InferenceAddr != "" {
		ic, err := NewInferenceClient(cfg.InferenceAddr)
		if err != nil {
			cancel()
			return nil, err
		}
		e.inference = ic

		// Start background inference goroutine
		e.wg.Add(1)
		go e.inferenceLoop()
	}

	return e, nil
}

// OnTrade processes a trade from the exchange. Thread-safe.
func (e *Engine) OnTrade(t Trade) {
	e.mu.Lock()
	defer e.mu.Unlock()

	switch t.Source {
	case SourceSpot:
		e.spot1m.AddTrade(t)
		e.spot1s.AddTrade(t)
	case SourcePerp:
		e.perp1m.AddTrade(t)
	}
}

// OnTicker processes a derivative ticker update. Thread-safe.
func (e *Engine) OnTicker(t TickerUpdate) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.features.OnTicker(t)
}

// OnLiquidation processes a liquidation event. Thread-safe.
func (e *Engine) OnLiquidation(l LiquidationUpdate) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.features.OnLiquidation(l)
}

// Features returns a snapshot of the current feature vector. Thread-safe.
func (e *Engine) Features() Features {
	e.mu.RLock()
	defer e.mu.RUnlock()
	f := e.features.Snapshot()
	f.Timestamp = time.Now()
	return f
}

// PriceTarget returns the latest model prediction. Thread-safe.
// Returns a target with Stale=true if no inference has been run or if
// the last inference is older than cfg.StaleThreshold.
func (e *Engine) PriceTarget() PriceTarget {
	e.mu.RLock()
	defer e.mu.RUnlock()
	pt := e.target
	if pt.Timestamp.IsZero() || time.Since(pt.Timestamp) > e.cfg.StaleThreshold {
		pt.Stale = true
	}
	return pt
}

// PriceStrike computes P(return > strikeReturn) from the latest mixture distribution.
// strikeReturn is in log-return space (0 = ATM).
func (e *Engine) PriceStrike(strikeReturn float64) float64 {
	e.mu.RLock()
	defer e.mu.RUnlock()

	pt := e.target
	if pt.Timestamp.IsZero() {
		return math.NaN()
	}

	prob := 0.0
	for k := 0; k < 4; k++ {
		if pt.MixtureSigma[k] == 0 {
			continue
		}
		z := (strikeReturn - pt.MixtureMu[k]) / pt.MixtureSigma[k]
		cdf := studentt.CDF(z, pt.MixtureNu[k])
		prob += pt.MixturePi[k] * (1 - cdf)
	}
	return prob
}

// Close shuts down the engine, flushing any pending bars and closing connections.
func (e *Engine) Close() error {
	e.cancel()

	// Flush pending bars
	e.mu.Lock()
	e.spot1m.Flush()
	e.spot1s.Flush()
	e.perp1m.Flush()
	e.mu.Unlock()

	e.wg.Wait()

	if e.inference != nil {
		return e.inference.Close()
	}
	return nil
}

// inferenceLoop runs in a background goroutine, calling the inference server
// whenever new bar data is available.
func (e *Engine) inferenceLoop() {
	defer e.wg.Done()
	for {
		select {
		case <-e.ctx.Done():
			return
		case <-e.inferCh:
			e.runInference()
		}
	}
}

func (e *Engine) runInference() {
	e.mu.RLock()
	seq := e.features.Sequence()
	barCount := e.features.BarCount()
	e.mu.RUnlock()

	if seq == nil || barCount < e.cfg.SequenceLength {
		return
	}

	tsUs := time.Now().UnixMicro()
	pt, xgbProb, xgbUnc, bsProb, err := e.inference.Predict(e.ctx, seq, tsUs)
	if err != nil {
		log.Printf("inference error: %v", err)
		return
	}

	e.mu.Lock()
	e.target = *pt
	e.features.SetXGBFeatures(xgbProb, xgbUnc, bsProb)
	e.mu.Unlock()
}
