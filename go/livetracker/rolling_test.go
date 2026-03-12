package livetracker

import (
	"math"
	"testing"
)

func almostEqual(a, b, tol float64) bool {
	if math.IsNaN(a) && math.IsNaN(b) {
		return true
	}
	return math.Abs(a-b) < tol
}

func TestRingBuffer_Basic(t *testing.T) {
	rb := NewRingBuffer(3)
	if rb.Len() != 0 || rb.Cap() != 3 {
		t.Fatalf("expected len=0, cap=3")
	}

	rb.Push(1)
	rb.Push(2)
	rb.Push(3)
	if rb.Len() != 3 {
		t.Fatalf("expected len=3, got %d", rb.Len())
	}
	if rb.Get(0) != 1 || rb.Get(1) != 2 || rb.Get(2) != 3 {
		t.Fatalf("unexpected values: %v %v %v", rb.Get(0), rb.Get(1), rb.Get(2))
	}

	// Eviction
	evicted, did := rb.Push(4)
	if !did || evicted != 1 {
		t.Fatalf("expected eviction of 1, got %v %v", evicted, did)
	}
	if rb.Get(0) != 2 || rb.Get(1) != 3 || rb.Get(2) != 4 {
		t.Fatalf("after eviction: %v %v %v", rb.Get(0), rb.Get(1), rb.Get(2))
	}
}

func TestRingBuffer_Last_NthFromEnd(t *testing.T) {
	rb := NewRingBuffer(5)
	for i := 1; i <= 5; i++ {
		rb.Push(float64(i))
	}
	if rb.Last() != 5 {
		t.Fatalf("Last: expected 5, got %v", rb.Last())
	}
	if rb.NthFromEnd(0) != 5 || rb.NthFromEnd(2) != 3 || rb.NthFromEnd(4) != 1 {
		t.Fatalf("NthFromEnd: unexpected values")
	}
	if !math.IsNaN(rb.NthFromEnd(5)) {
		t.Fatalf("NthFromEnd(5) should be NaN")
	}
}

func TestRollingSum(t *testing.T) {
	rs := NewRollingSum(3)
	rs.Push(1)
	rs.Push(2)
	rs.Push(3)
	if !almostEqual(rs.Sum(), 6, 1e-10) {
		t.Fatalf("expected sum=6, got %v", rs.Sum())
	}
	rs.Push(4) // evicts 1
	if !almostEqual(rs.Sum(), 9, 1e-10) {
		t.Fatalf("expected sum=9, got %v", rs.Sum())
	}
}

func TestRollingStats(t *testing.T) {
	rs := NewRollingStats(4)
	vals := []float64{2, 4, 4, 4}
	for _, v := range vals {
		rs.Push(v)
	}
	// mean = 3.5, std = sqrt(((2-3.5)^2+(4-3.5)^2*3)/3) = sqrt(3/3) = 1.0
	if !almostEqual(rs.Mean(), 3.5, 1e-10) {
		t.Fatalf("expected mean=3.5, got %v", rs.Mean())
	}
	if !almostEqual(rs.Std(), 1.0, 1e-10) {
		t.Fatalf("expected std=1.0, got %v", rs.Std())
	}

	// Z-score: (5 - 3.5) / 1.0 = 1.5
	if !almostEqual(rs.ZScore(5), 1.5, 1e-10) {
		t.Fatalf("expected zscore=1.5, got %v", rs.ZScore(5))
	}
}

func TestDiffBuffer(t *testing.T) {
	db := NewDiffBuffer(5)
	for i := 1; i <= 6; i++ {
		db.Push(float64(i * 10))
	}
	// Last = 60, diff(1) = 60-50=10, diff(3) = 60-30=30
	if !almostEqual(db.Diff(1), 10, 1e-10) {
		t.Fatalf("expected diff(1)=10, got %v", db.Diff(1))
	}
	if !almostEqual(db.Diff(3), 30, 1e-10) {
		t.Fatalf("expected diff(3)=30, got %v", db.Diff(3))
	}
	if !math.IsNaN(db.Diff(6)) {
		t.Fatalf("expected NaN for lag > count")
	}
}
