package livetracker

import "math"

// RegimeComputer computes Tier-2 regime signals:
// variance ratio, Hurst exponent (R/S), and lag-1 autocorrelation.
type RegimeComputer struct {
	// Variance Ratio
	vrQ      int
	vrWindow int
	retBuf   *RingBuffer  // raw 1-min log returns
	qRetBuf  *RingBuffer  // q-period cumulative returns

	// Hurst exponent
	hurstWindow int
	hurstBuf    *RingBuffer

	// Autocorrelation
	acWindow int
	acBuf    *RingBuffer
}

// NewRegimeComputer creates a RegimeComputer matching the Python config.
func NewRegimeComputer(vrQ, vrWindow, hurstWindow, acWindow int) *RegimeComputer {
	return &RegimeComputer{
		vrQ:         vrQ,
		vrWindow:    vrWindow,
		retBuf:      NewRingBuffer(vrWindow + vrQ), // need q extra for q-period returns
		qRetBuf:     NewRingBuffer(vrWindow),
		hurstWindow: hurstWindow,
		hurstBuf:    NewRingBuffer(hurstWindow),
		acWindow:    acWindow,
		acBuf:       NewRingBuffer(acWindow),
	}
}

// Push adds a new 1-min log return and updates all regime signals.
func (rc *RegimeComputer) Push(logReturn float64) {
	rc.retBuf.Push(logReturn)
	rc.hurstBuf.Push(logReturn)
	rc.acBuf.Push(logReturn)

	// Compute q-period cumulative return for variance ratio
	n := rc.retBuf.Len()
	if n >= rc.vrQ {
		qRet := 0.0
		for i := n - rc.vrQ; i < n; i++ {
			qRet += rc.retBuf.Get(i)
		}
		rc.qRetBuf.Push(qRet)
	}
}

// VarianceRatio returns VR(q) = Var(q-period returns) / (q * Var(1-period returns)).
// Values > 1 indicate trending, < 1 indicate mean-reversion.
func (rc *RegimeComputer) VarianceRatio() float64 {
	n := rc.retBuf.Len()
	if n < 10 {
		return math.NaN()
	}

	// Var of 1-period returns (from last vrWindow values)
	start := 0
	if n > rc.vrWindow {
		start = n - rc.vrWindow
	}
	count := n - start
	if count < 10 {
		return math.NaN()
	}

	var sum, sumSq float64
	for i := start; i < n; i++ {
		v := rc.retBuf.Get(i)
		sum += v
		sumSq += v * v
	}
	fn := float64(count)
	mean := sum / fn
	var1 := sumSq/fn - mean*mean
	if var1 <= 0 {
		return math.NaN()
	}

	// Var of q-period returns
	nq := rc.qRetBuf.Len()
	if nq < 10 {
		return math.NaN()
	}
	startQ := 0
	if nq > rc.vrWindow {
		startQ = nq - rc.vrWindow
	}
	countQ := nq - startQ
	if countQ < 10 {
		return math.NaN()
	}

	var sumQ, sumSqQ float64
	for i := startQ; i < startQ+countQ; i++ {
		v := rc.qRetBuf.Get(i)
		sumQ += v
		sumSqQ += v * v
	}
	fnQ := float64(countQ)
	meanQ := sumQ / fnQ
	varQ := sumSqQ/fnQ - meanQ*meanQ

	return varQ / (float64(rc.vrQ) * var1)
}

// Hurst returns the Hurst exponent via rescaled range (R/S) analysis.
// H < 0.5: mean-reverting, H = 0.5: random walk, H > 0.5: trending.
func (rc *RegimeComputer) Hurst() float64 {
	n := rc.hurstBuf.Len()
	if n < 20 {
		return math.NaN()
	}

	// Compute mean
	var sum float64
	for i := 0; i < n; i++ {
		sum += rc.hurstBuf.Get(i)
	}
	mean := sum / float64(n)

	// Cumulative deviations
	var cumDev, maxDev, minDev float64
	var sumSq float64
	for i := 0; i < n; i++ {
		v := rc.hurstBuf.Get(i)
		d := v - mean
		cumDev += d
		sumSq += d * d
		if cumDev > maxDev {
			maxDev = cumDev
		}
		if cumDev < minDev {
			minDev = cumDev
		}
	}

	r := maxDev - minDev
	s := math.Sqrt(sumSq / float64(n))
	if s == 0 {
		return math.NaN()
	}

	return math.Log(r/s) / math.Log(float64(n))
}

// AutocorrLag1 returns the rolling lag-1 autocorrelation.
func (rc *RegimeComputer) AutocorrLag1() float64 {
	n := rc.acBuf.Len()
	if n < 10 {
		return math.NaN()
	}

	// Compute mean
	var sum float64
	for i := 0; i < n; i++ {
		sum += rc.acBuf.Get(i)
	}
	mean := sum / float64(n)

	// Lag-1 autocorrelation: sum((x[t]-mean)*(x[t-1]-mean)) / sum((x[t]-mean)^2)
	var num, den float64
	for i := 1; i < n; i++ {
		d0 := rc.acBuf.Get(i) - mean
		d1 := rc.acBuf.Get(i-1) - mean
		num += d0 * d1
		den += d0 * d0
	}
	if den == 0 {
		return math.NaN()
	}
	return num / den
}

// HMMPredictor runs the forward algorithm for a pre-trained 4-state Gaussian HMM.
// Parameters are loaded from the trained model (saved after offline fitting).
type HMMPredictor struct {
	nStates int
	nDims   int

	// Model parameters
	startProb []float64       // initial state distribution [nStates]
	transmat  [][]float64     // transition matrix [nStates][nStates]
	means     [][]float64     // emission means [nStates][nDims]
	covarsInv [][][]float64   // inverse covariance matrices [nStates][nDims][nDims]
	covarsDet []float64       // determinant of covariance matrices [nStates]

	// State label mapping (state index → canonical label)
	// 0=Bull, 1=Bear, 2=Calm, 3=Transition
	stateMap [4]int // maps canonical label to raw HMM state index

	// Forward algorithm state
	alpha []float64 // current forward probabilities [nStates]
}

// HMMParams holds the pre-trained HMM parameters.
type HMMParams struct {
	NStates   int
	NDims     int
	StartProb []float64
	TransMat  [][]float64
	Means     [][]float64
	Covars    [][][]float64 // covariance matrices [nStates][nDims][nDims]
	// State ordering: indices of [Bull, Bear, Calm, Transition] in the raw HMM states
	StateMap  [4]int
}

// NewHMMPredictor creates an HMM predictor from pre-trained parameters.
func NewHMMPredictor(p HMMParams) *HMMPredictor {
	h := &HMMPredictor{
		nStates:   p.NStates,
		nDims:     p.NDims,
		startProb: make([]float64, p.NStates),
		transmat:  make([][]float64, p.NStates),
		means:     make([][]float64, p.NStates),
		covarsInv: make([][][]float64, p.NStates),
		covarsDet: make([]float64, p.NStates),
		stateMap:  p.StateMap,
		alpha:     make([]float64, p.NStates),
	}

	copy(h.startProb, p.StartProb)
	copy(h.alpha, p.StartProb)

	for i := 0; i < p.NStates; i++ {
		h.transmat[i] = make([]float64, p.NStates)
		copy(h.transmat[i], p.TransMat[i])

		h.means[i] = make([]float64, p.NDims)
		copy(h.means[i], p.Means[i])

		// Compute inverse and determinant of covariance matrix
		h.covarsInv[i], h.covarsDet[i] = invertMatrix(p.Covars[i])
	}

	return h
}

// Observe updates the forward probabilities given a new observation vector.
// obs must have length nDims.
func (h *HMMPredictor) Observe(obs []float64) {
	if h == nil {
		return
	}

	// Prediction step: alpha_pred[j] = sum_i(alpha[i] * trans[i][j])
	pred := make([]float64, h.nStates)
	for j := 0; j < h.nStates; j++ {
		for i := 0; i < h.nStates; i++ {
			pred[j] += h.alpha[i] * h.transmat[i][j]
		}
	}

	// Update step: alpha[j] = pred[j] * P(obs | state=j)
	var total float64
	for j := 0; j < h.nStates; j++ {
		emission := h.gaussianPDF(j, obs)
		h.alpha[j] = pred[j] * emission
		total += h.alpha[j]
	}

	// Normalize
	if total > 0 {
		for j := 0; j < h.nStates; j++ {
			h.alpha[j] /= total
		}
	} else {
		// Reset to uniform if numerical underflow
		for j := 0; j < h.nStates; j++ {
			h.alpha[j] = 1.0 / float64(h.nStates)
		}
	}
}

// Posteriors returns the current state probabilities in canonical order:
// [P(Bull), P(Bear), P(Calm), P(Transition)].
func (h *HMMPredictor) Posteriors() [4]float64 {
	if h == nil {
		return [4]float64{0.25, 0.25, 0.25, 0.25}
	}
	var out [4]float64
	for label := 0; label < 4; label++ {
		rawIdx := h.stateMap[label]
		if rawIdx < len(h.alpha) {
			out[label] = h.alpha[rawIdx]
		}
	}
	return out
}

// gaussianPDF computes the multivariate Gaussian PDF for state j.
func (h *HMMPredictor) gaussianPDF(j int, obs []float64) float64 {
	d := h.nDims
	// Compute (obs - mean)
	diff := make([]float64, d)
	for i := 0; i < d; i++ {
		diff[i] = obs[i] - h.means[j][i]
	}

	// Compute diff^T * covInv * diff (Mahalanobis distance squared)
	var maha float64
	for i := 0; i < d; i++ {
		var row float64
		for k := 0; k < d; k++ {
			row += h.covarsInv[j][i][k] * diff[k]
		}
		maha += diff[i] * row
	}

	// PDF = (2π)^(-d/2) * |Σ|^(-1/2) * exp(-0.5 * maha)
	det := h.covarsDet[j]
	if det <= 0 {
		return 1e-300 // avoid NaN
	}
	logPdf := -0.5*float64(d)*math.Log(2*math.Pi) - 0.5*math.Log(det) - 0.5*maha
	return math.Exp(logPdf)
}

// invertMatrix computes the inverse and determinant of a square matrix.
// For small dimensions (≤5), uses cofactor expansion. For production HMM
// with d=5, this is efficient enough.
func invertMatrix(m [][]float64) (inv [][]float64, det float64) {
	n := len(m)
	if n == 0 {
		return nil, 0
	}

	// Augmented matrix [m | I]
	aug := make([][]float64, n)
	for i := range aug {
		aug[i] = make([]float64, 2*n)
		for j := 0; j < n; j++ {
			aug[i][j] = m[i][j]
		}
		aug[i][n+i] = 1
	}

	det = 1.0
	for col := 0; col < n; col++ {
		// Partial pivoting
		maxRow := col
		maxVal := math.Abs(aug[col][col])
		for row := col + 1; row < n; row++ {
			if math.Abs(aug[row][col]) > maxVal {
				maxVal = math.Abs(aug[row][col])
				maxRow = row
			}
		}
		if maxRow != col {
			aug[col], aug[maxRow] = aug[maxRow], aug[col]
			det = -det
		}

		pivot := aug[col][col]
		if math.Abs(pivot) < 1e-15 {
			return nil, 0 // singular
		}
		det *= pivot

		// Scale pivot row
		for j := 0; j < 2*n; j++ {
			aug[col][j] /= pivot
		}

		// Eliminate column
		for row := 0; row < n; row++ {
			if row == col {
				continue
			}
			factor := aug[row][col]
			for j := 0; j < 2*n; j++ {
				aug[row][j] -= factor * aug[col][j]
			}
		}
	}

	// Extract inverse
	inv = make([][]float64, n)
	for i := range inv {
		inv[i] = make([]float64, n)
		copy(inv[i], aug[i][n:])
	}
	return inv, det
}
