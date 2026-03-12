// Package studentt provides a Student's t-distribution CDF implementation.
package studentt

import "math"

// CDF computes the cumulative distribution function of the Student's t-distribution.
// Uses the regularized incomplete beta function.
func CDF(x, nu float64) float64 {
	if nu <= 0 {
		return math.NaN()
	}
	t2 := x * x
	p := 0.5 * regIncBeta(nu/2, 0.5, nu/(nu+t2))
	if x >= 0 {
		return 1 - p
	}
	return p
}

// regIncBeta computes the regularized incomplete beta function I_x(a, b)
// using a continued fraction expansion (Lentz's method).
func regIncBeta(a, b, x float64) float64 {
	if x < 0 || x > 1 {
		return math.NaN()
	}
	if x == 0 {
		return 0
	}
	if x == 1 {
		return 1
	}

	// Use symmetry relation for better convergence
	if x > (a+1)/(a+b+2) {
		return 1 - regIncBeta(b, a, 1-x)
	}

	lnBeta := lgamma(a) + lgamma(b) - lgamma(a+b)
	front := math.Exp(math.Log(x)*a+math.Log(1-x)*b-lnBeta) / a

	// Lentz's continued fraction
	f := betaCF(a, b, x)
	return front * f
}

func lgamma(x float64) float64 {
	v, _ := math.Lgamma(x)
	return v
}

// betaCF evaluates the continued fraction for the incomplete beta function.
func betaCF(a, b, x float64) float64 {
	const maxIter = 200
	const eps = 3e-14

	qab := a + b
	qap := a + 1
	qam := a - 1

	c := 1.0
	d := 1 - qab*x/qap
	if math.Abs(d) < 1e-30 {
		d = 1e-30
	}
	d = 1 / d
	h := d

	for m := 1; m <= maxIter; m++ {
		fm := float64(m)
		// Even step
		num := fm * (b - fm) * x / ((qam + 2*fm) * (a + 2*fm))
		d = 1 + num*d
		if math.Abs(d) < 1e-30 {
			d = 1e-30
		}
		c = 1 + num/c
		if math.Abs(c) < 1e-30 {
			c = 1e-30
		}
		d = 1 / d
		h *= d * c

		// Odd step
		num = -(a + fm) * (qab + fm) * x / ((a + 2*fm) * (qap + 2*fm))
		d = 1 + num*d
		if math.Abs(d) < 1e-30 {
			d = 1e-30
		}
		c = 1 + num/c
		if math.Abs(c) < 1e-30 {
			c = 1e-30
		}
		d = 1 / d
		delta := d * c
		h *= delta

		if math.Abs(delta-1) < eps {
			break
		}
	}
	return h
}
