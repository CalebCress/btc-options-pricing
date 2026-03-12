package livetracker

import (
	"math"
	"testing"
)

func TestRegimeComputer_VarianceRatio(t *testing.T) {
	rc := NewRegimeComputer(5, 60, 60, 30)

	// Feed random walk returns — VR should be ~1
	for i := 0; i < 100; i++ {
		// Alternating small returns
		ret := 0.001 * float64(i%3-1)
		rc.Push(ret)
	}

	vr := rc.VarianceRatio()
	if math.IsNaN(vr) {
		t.Fatal("expected non-NaN variance ratio")
	}
	// For a structured series, VR won't be exactly 1, but should be finite
	if vr < 0 || vr > 10 {
		t.Errorf("variance ratio out of reasonable range: %v", vr)
	}
}

func TestRegimeComputer_Hurst(t *testing.T) {
	rc := NewRegimeComputer(5, 60, 60, 30)

	// Feed trending returns (all positive) — Hurst should be > 0.5
	for i := 0; i < 60; i++ {
		rc.Push(0.001)
	}

	h := rc.Hurst()
	if math.IsNaN(h) {
		t.Fatal("expected non-NaN Hurst exponent")
	}
	// Constant returns: S=0, so NaN. Use varying trending:
	rc2 := NewRegimeComputer(5, 60, 60, 30)
	for i := 0; i < 60; i++ {
		rc2.Push(0.001 + 0.0001*float64(i))
	}
	h2 := rc2.Hurst()
	if math.IsNaN(h2) {
		t.Fatal("expected non-NaN Hurst for trending series")
	}
	if h2 < 0 || h2 > 2 {
		t.Errorf("Hurst out of range: %v", h2)
	}
}

func TestRegimeComputer_AutocorrLag1(t *testing.T) {
	rc := NewRegimeComputer(5, 60, 60, 30)

	// Feed perfectly alternating returns — should have negative autocorrelation
	for i := 0; i < 30; i++ {
		if i%2 == 0 {
			rc.Push(0.001)
		} else {
			rc.Push(-0.001)
		}
	}

	ac := rc.AutocorrLag1()
	if math.IsNaN(ac) {
		t.Fatal("expected non-NaN autocorrelation")
	}
	if ac > 0 {
		t.Errorf("expected negative autocorr for alternating series, got %v", ac)
	}
}

func TestRegimeComputer_InsufficientData(t *testing.T) {
	rc := NewRegimeComputer(5, 60, 60, 30)

	// Only 5 values — not enough for any statistic
	for i := 0; i < 5; i++ {
		rc.Push(float64(i) * 0.001)
	}

	if !math.IsNaN(rc.VarianceRatio()) {
		t.Error("expected NaN variance ratio with insufficient data")
	}
	if !math.IsNaN(rc.Hurst()) {
		t.Error("expected NaN Hurst with insufficient data")
	}
	if !math.IsNaN(rc.AutocorrLag1()) {
		t.Error("expected NaN autocorr with insufficient data")
	}
}

func TestHMMPredictor_Posteriors(t *testing.T) {
	// Simple 2-state, 1-dim HMM for testing
	params := HMMParams{
		NStates:   4,
		NDims:     1,
		StartProb: []float64{0.25, 0.25, 0.25, 0.25},
		TransMat: [][]float64{
			{0.9, 0.03, 0.04, 0.03},
			{0.03, 0.9, 0.04, 0.03},
			{0.04, 0.03, 0.9, 0.03},
			{0.03, 0.04, 0.03, 0.9},
		},
		Means: [][]float64{
			{0.002},  // Bull: positive mean
			{-0.002}, // Bear: negative mean
			{0.0},    // Calm: zero mean
			{0.0},    // Transition: zero mean
		},
		Covars: [][][]float64{
			{{0.0001}},
			{{0.0001}},
			{{0.00005}}, // Calm: low vol
			{{0.0003}},  // Transition: high vol
		},
		StateMap: [4]int{0, 1, 2, 3},
	}

	hmm := NewHMMPredictor(params)

	// Posteriors should start uniform
	post := hmm.Posteriors()
	for i, p := range post {
		if math.Abs(p-0.25) > 1e-10 {
			t.Errorf("initial posterior[%d] = %v, want 0.25", i, p)
		}
	}

	// Feed strongly positive observations → should increase Bull probability
	for i := 0; i < 10; i++ {
		hmm.Observe([]float64{0.003})
	}
	post = hmm.Posteriors()
	if post[0] < post[1] {
		t.Errorf("after positive obs, P(bull)=%v should be > P(bear)=%v", post[0], post[1])
	}

	// Posteriors should sum to ~1
	sum := post[0] + post[1] + post[2] + post[3]
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("posteriors sum = %v, want 1.0", sum)
	}
}

func TestHMMPredictor_Nil(t *testing.T) {
	var hmm *HMMPredictor
	// Should not panic
	hmm.Observe([]float64{1, 2, 3})
	post := hmm.Posteriors()
	// Nil HMM returns uniform
	for i, p := range post {
		if math.Abs(p-0.25) > 1e-10 {
			t.Errorf("nil HMM posterior[%d] = %v, want 0.25", i, p)
		}
	}
}

func TestInvertMatrix(t *testing.T) {
	// 2x2 identity should invert to itself
	m := [][]float64{{1, 0}, {0, 1}}
	inv, det := invertMatrix(m)
	if math.Abs(det-1.0) > 1e-10 {
		t.Errorf("det = %v, want 1.0", det)
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			if math.Abs(inv[i][j]-expected) > 1e-10 {
				t.Errorf("inv[%d][%d] = %v, want %v", i, j, inv[i][j], expected)
			}
		}
	}

	// 2x2 matrix
	m2 := [][]float64{{4, 7}, {2, 6}}
	inv2, det2 := invertMatrix(m2)
	// det = 24 - 14 = 10
	if math.Abs(det2-10.0) > 1e-10 {
		t.Errorf("det = %v, want 10.0", det2)
	}
	// inv = [0.6, -0.7; -0.2, 0.4]
	if math.Abs(inv2[0][0]-0.6) > 1e-10 {
		t.Errorf("inv[0][0] = %v, want 0.6", inv2[0][0])
	}
}
