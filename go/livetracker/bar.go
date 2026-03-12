package livetracker

import (
	"math"
	"time"
)

// barAccum accumulates trade data for a single bar.
type barAccum struct {
	open, high, low, last float64
	volume, buyVol, sellVol float64
	vwapNum                 float64 // Σ(price * amount)
	count                   int
	started                 bool
}

func (a *barAccum) reset() {
	*a = barAccum{}
}

func (a *barAccum) addTrade(price, amount float64, isBuyerMaker bool) {
	if !a.started {
		a.open = price
		a.high = price
		a.low = price
		a.started = true
	}
	if price > a.high {
		a.high = price
	}
	if price < a.low {
		a.low = price
	}
	a.last = price
	a.volume += amount
	a.vwapNum += price * amount
	a.count++
	if isBuyerMaker {
		a.sellVol += amount // maker was buyer → taker sold
	} else {
		a.buyVol += amount // maker was seller → taker bought
	}
}

func (a *barAccum) toBar(ts time.Time) Bar {
	vwap := a.last
	if a.volume > 0 {
		vwap = a.vwapNum / a.volume
	}
	return Bar{
		Timestamp:  ts,
		Open:       a.open,
		High:       a.high,
		Low:        a.low,
		Close:      a.last,
		Volume:     a.volume,
		BuyVolume:  a.buyVol,
		SellVolume: a.sellVol,
		TradeCount: a.count,
		VWAP:       vwap,
	}
}

// BarAggregator converts a stream of trades into time bars.
type BarAggregator struct {
	interval  time.Duration
	current   barAccum
	barStart  time.Time
	lastClose float64
	onBar     func(Bar)
	started   bool
}

// NewBarAggregator creates a bar aggregator with the given interval and callback.
func NewBarAggregator(interval time.Duration, onBar func(Bar)) *BarAggregator {
	return &BarAggregator{
		interval: interval,
		onBar:    onBar,
	}
}

// floorTime floors a timestamp to the bar interval.
func (ba *BarAggregator) floorTime(t time.Time) time.Time {
	d := ba.interval
	return t.Truncate(d)
}

// AddTrade processes a single trade. If the trade crosses a bar boundary,
// the current bar is emitted and a new bar starts. Empty gap bars are
// forward-filled from the last close.
func (ba *BarAggregator) AddTrade(t Trade) {
	ts := t.Timestamp
	barTime := ba.floorTime(ts)

	if !ba.started {
		ba.barStart = barTime
		ba.started = true
		ba.lastClose = t.Price
	}

	// Check if we've crossed into a new bar period.
	if barTime.After(ba.barStart) {
		// Emit the current bar (if it has trades).
		ba.emitBar(ba.barStart)

		// Fill any gap bars (empty bars between last bar and current trade).
		next := ba.barStart.Add(ba.interval)
		for next.Before(barTime) {
			ba.emitEmptyBar(next)
			next = next.Add(ba.interval)
		}

		ba.barStart = barTime
		ba.current.reset()
	}

	ba.current.addTrade(t.Price, t.Amount, t.IsBuyerMaker)
}

// Flush emits the current bar if it has any trades. Call this on shutdown or timer.
func (ba *BarAggregator) Flush() {
	if ba.current.started {
		ba.emitBar(ba.barStart)
		ba.current.reset()
		ba.barStart = ba.barStart.Add(ba.interval)
	}
}

func (ba *BarAggregator) emitBar(ts time.Time) {
	if !ba.current.started {
		ba.emitEmptyBar(ts)
		return
	}
	bar := ba.current.toBar(ts)
	ba.lastClose = bar.Close
	if ba.onBar != nil {
		ba.onBar(bar)
	}
}

func (ba *BarAggregator) emitEmptyBar(ts time.Time) {
	// Forward-fill: use last close price.
	bar := Bar{
		Timestamp:  ts,
		Open:       ba.lastClose,
		High:       ba.lastClose,
		Low:        ba.lastClose,
		Close:      ba.lastClose,
		Volume:     0,
		BuyVolume:  0,
		SellVolume: 0,
		TradeCount: 0,
		VWAP:       math.NaN(),
	}
	if ba.onBar != nil {
		ba.onBar(bar)
	}
}
