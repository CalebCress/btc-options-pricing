package livetracker

import (
	"testing"
	"time"
)

func TestBarAggregator_BasicBar(t *testing.T) {
	var bars []Bar
	agg := NewBarAggregator(time.Minute, func(b Bar) {
		bars = append(bars, b)
	})

	t0 := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

	// Three trades in the same minute
	agg.AddTrade(Trade{Timestamp: t0.Add(10 * time.Second), Price: 100, Amount: 1, IsBuyerMaker: false})
	agg.AddTrade(Trade{Timestamp: t0.Add(20 * time.Second), Price: 105, Amount: 2, IsBuyerMaker: true})
	agg.AddTrade(Trade{Timestamp: t0.Add(30 * time.Second), Price: 102, Amount: 3, IsBuyerMaker: false})

	// Trade in next minute triggers bar emission
	agg.AddTrade(Trade{Timestamp: t0.Add(61 * time.Second), Price: 103, Amount: 1, IsBuyerMaker: false})

	if len(bars) != 1 {
		t.Fatalf("expected 1 bar, got %d", len(bars))
	}

	b := bars[0]
	if b.Open != 100 || b.High != 105 || b.Low != 100 || b.Close != 102 {
		t.Fatalf("OHLC mismatch: %v %v %v %v", b.Open, b.High, b.Low, b.Close)
	}
	if b.TradeCount != 3 {
		t.Fatalf("expected 3 trades, got %d", b.TradeCount)
	}
	if !almostEqual(b.Volume, 6, 1e-10) {
		t.Fatalf("expected volume=6, got %v", b.Volume)
	}
	// Buy volume: trades 1 and 3 (IsBuyerMaker=false → taker bought)
	if !almostEqual(b.BuyVolume, 4, 1e-10) {
		t.Fatalf("expected buy_vol=4, got %v", b.BuyVolume)
	}
	// Sell volume: trade 2 (IsBuyerMaker=true → taker sold)
	if !almostEqual(b.SellVolume, 2, 1e-10) {
		t.Fatalf("expected sell_vol=2, got %v", b.SellVolume)
	}
}

func TestBarAggregator_GapFill(t *testing.T) {
	var bars []Bar
	agg := NewBarAggregator(time.Minute, func(b Bar) {
		bars = append(bars, b)
	})

	t0 := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

	// Trade at minute 0
	agg.AddTrade(Trade{Timestamp: t0.Add(10 * time.Second), Price: 100, Amount: 1})

	// Trade at minute 3 (gap of 2 minutes)
	agg.AddTrade(Trade{Timestamp: t0.Add(3*time.Minute + 10*time.Second), Price: 105, Amount: 1})

	// Should have: bar[0] (minute 0 real), bar[1] (minute 1 fill), bar[2] (minute 2 fill)
	if len(bars) != 3 {
		t.Fatalf("expected 3 bars (1 real + 2 gap fill), got %d", len(bars))
	}

	// Gap bars should be forward-filled with last close
	for i := 1; i <= 2; i++ {
		if bars[i].Close != 100 || bars[i].TradeCount != 0 {
			t.Fatalf("gap bar %d: expected close=100 with 0 trades, got close=%v trades=%d",
				i, bars[i].Close, bars[i].TradeCount)
		}
	}
}

func TestBarAggregator_Flush(t *testing.T) {
	var bars []Bar
	agg := NewBarAggregator(time.Minute, func(b Bar) {
		bars = append(bars, b)
	})

	t0 := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	agg.AddTrade(Trade{Timestamp: t0.Add(10 * time.Second), Price: 100, Amount: 1})

	if len(bars) != 0 {
		t.Fatalf("expected 0 bars before flush")
	}

	agg.Flush()
	if len(bars) != 1 {
		t.Fatalf("expected 1 bar after flush, got %d", len(bars))
	}
}
