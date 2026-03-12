package livetracker

import "math"

// RingBuffer is a fixed-capacity circular buffer of float64 values.
// Zero allocation after initialization.
type RingBuffer struct {
	data  []float64
	head  int // next write position
	count int
	cap   int
}

// NewRingBuffer creates a ring buffer with the given capacity.
func NewRingBuffer(capacity int) *RingBuffer {
	return &RingBuffer{
		data: make([]float64, capacity),
		cap:  capacity,
	}
}

// Push adds a value, evicting the oldest if at capacity. Returns the evicted value and whether one was evicted.
func (rb *RingBuffer) Push(v float64) (evicted float64, didEvict bool) {
	if rb.count == rb.cap {
		evicted = rb.data[rb.head]
		didEvict = true
	}
	rb.data[rb.head] = v
	rb.head = (rb.head + 1) % rb.cap
	if rb.count < rb.cap {
		rb.count++
	}
	return
}

// Len returns the number of elements currently stored.
func (rb *RingBuffer) Len() int { return rb.count }

// Cap returns the buffer capacity.
func (rb *RingBuffer) Cap() int { return rb.cap }

// Get returns the i-th element (0 = oldest).
func (rb *RingBuffer) Get(i int) float64 {
	idx := (rb.head - rb.count + i) % rb.cap
	if idx < 0 {
		idx += rb.cap
	}
	return rb.data[idx]
}

// Last returns the most recently pushed value.
func (rb *RingBuffer) Last() float64 {
	if rb.count == 0 {
		return math.NaN()
	}
	idx := (rb.head - 1 + rb.cap) % rb.cap
	return rb.data[idx]
}

// NthFromEnd returns the n-th value from the end (0 = last, 1 = second-to-last).
func (rb *RingBuffer) NthFromEnd(n int) float64 {
	if n >= rb.count {
		return math.NaN()
	}
	idx := (rb.head - 1 - n + rb.cap*2) % rb.cap
	return rb.data[idx]
}

// RollingSum maintains a running sum over the last N values.
type RollingSum struct {
	buf *RingBuffer
	sum float64
}

// NewRollingSum creates a rolling sum with the given window size.
func NewRollingSum(window int) *RollingSum {
	return &RollingSum{buf: NewRingBuffer(window)}
}

// Push adds a value, maintaining the rolling sum.
func (rs *RollingSum) Push(v float64) {
	evicted, did := rs.buf.Push(v)
	rs.sum += v
	if did {
		rs.sum -= evicted
	}
}

// Sum returns the current rolling sum.
func (rs *RollingSum) Sum() float64 { return rs.sum }

// Len returns the number of values in the window.
func (rs *RollingSum) Len() int { return rs.buf.Len() }

// RollingStats maintains online mean and variance using a simple sum/sumSq approach.
// For our use case (z-scores over rolling windows), this is sufficient and fast.
type RollingStats struct {
	buf   *RingBuffer
	sum   float64
	sumSq float64
}

// NewRollingStats creates rolling statistics with the given window size.
func NewRollingStats(window int) *RollingStats {
	return &RollingStats{buf: NewRingBuffer(window)}
}

// Push adds a value.
func (rs *RollingStats) Push(v float64) {
	evicted, did := rs.buf.Push(v)
	rs.sum += v
	rs.sumSq += v * v
	if did {
		rs.sum -= evicted
		rs.sumSq -= evicted * evicted
	}
}

// Len returns the number of values in the window.
func (rs *RollingStats) Len() int { return rs.buf.Len() }

// Mean returns the rolling mean.
func (rs *RollingStats) Mean() float64 {
	n := rs.buf.Len()
	if n == 0 {
		return math.NaN()
	}
	return rs.sum / float64(n)
}

// Variance returns the rolling sample variance.
func (rs *RollingStats) Variance() float64 {
	n := rs.buf.Len()
	if n < 2 {
		return math.NaN()
	}
	mean := rs.sum / float64(n)
	return (rs.sumSq/float64(n) - mean*mean) * float64(n) / float64(n-1)
}

// Std returns the rolling sample standard deviation.
func (rs *RollingStats) Std() float64 {
	v := rs.Variance()
	if math.IsNaN(v) || v < 0 {
		return math.NaN()
	}
	return math.Sqrt(v)
}

// ZScore returns the z-score of value v against the rolling distribution.
func (rs *RollingStats) ZScore(v float64) float64 {
	s := rs.Std()
	if math.IsNaN(s) || s == 0 {
		return math.NaN()
	}
	return (v - rs.Mean()) / s
}

// DiffBuffer maintains a history to compute value[t] - value[t-lag].
type DiffBuffer struct {
	buf *RingBuffer
}

// NewDiffBuffer creates a diff buffer that can compute diffs up to maxLag.
func NewDiffBuffer(maxLag int) *DiffBuffer {
	return &DiffBuffer{buf: NewRingBuffer(maxLag + 1)}
}

// Push adds a value.
func (db *DiffBuffer) Push(v float64) {
	db.buf.Push(v)
}

// Diff returns value[now] - value[now - lag].
func (db *DiffBuffer) Diff(lag int) float64 {
	if db.buf.Len() <= lag {
		return math.NaN()
	}
	return db.buf.Last() - db.buf.NthFromEnd(lag)
}

// Last returns the most recent value.
func (db *DiffBuffer) Last() float64 {
	return db.buf.Last()
}
