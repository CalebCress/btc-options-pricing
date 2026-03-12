package livetracker

import "math"

// VPINCalculator computes Volume-Synchronized Probability of Informed Trading.
// It buckets trades by cumulative volume and computes the rolling imbalance ratio.
type VPINCalculator struct {
	bucketSize int
	nBuckets   int

	// Current bucket accumulation.
	cumVol     float64
	bucketImb  float64 // |buy - sell| for current bucket
	bucketVol  float64 // total volume for current bucket

	// Rolling window over completed buckets.
	imbBuf *RingBuffer // per-bucket |buy-sell|
	volBuf *RingBuffer // per-bucket total volume
	imbSum float64
	volSum float64

	lastVPIN float64
}

// NewVPINCalculator creates a VPIN calculator with the given bucket size and window.
func NewVPINCalculator(bucketSize, nBuckets int) *VPINCalculator {
	return &VPINCalculator{
		bucketSize: bucketSize,
		nBuckets:   nBuckets,
		imbBuf:     NewRingBuffer(nBuckets),
		volBuf:     NewRingBuffer(nBuckets),
		lastVPIN:   math.NaN(),
	}
}

// OnBar updates VPIN with a new bar's buy/sell volume.
func (v *VPINCalculator) OnBar(buyVol, sellVol float64) {
	totalVol := buyVol + sellVol
	imbalance := math.Abs(buyVol - sellVol)

	v.bucketImb += imbalance
	v.bucketVol += totalVol
	v.cumVol += totalVol

	// Close buckets as cumulative volume crosses thresholds.
	for v.cumVol >= float64(v.bucketSize) {
		v.closeBucket()
		v.cumVol -= float64(v.bucketSize)
	}
}

func (v *VPINCalculator) closeBucket() {
	// Evict oldest bucket from rolling sums.
	evImb, didImb := v.imbBuf.Push(v.bucketImb)
	evVol, didVol := v.volBuf.Push(v.bucketVol)
	v.imbSum += v.bucketImb
	v.volSum += v.bucketVol
	if didImb {
		v.imbSum -= evImb
	}
	if didVol {
		v.volSum -= evVol
	}

	// Compute VPIN.
	if v.volSum > 0 {
		v.lastVPIN = v.imbSum / v.volSum
	}

	// Reset bucket accumulators.
	v.bucketImb = 0
	v.bucketVol = 0
}

// Value returns the current VPIN. NaN if not enough data.
func (v *VPINCalculator) Value() float64 {
	return v.lastVPIN
}
