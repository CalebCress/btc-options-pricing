package livetracker

import (
	"context"
	"fmt"
	"time"

	pb "github.com/CalebCress/btc-options-pricing/go/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// InferenceClient wraps the gRPC connection to the Python inference server.
type InferenceClient struct {
	conn   *grpc.ClientConn
	client pb.InferenceServiceClient
}

// NewInferenceClient connects to the inference server at the given address.
func NewInferenceClient(addr string) (*InferenceClient, error) {
	conn, err := grpc.NewClient(addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to inference server at %s: %w", addr, err)
	}
	return &InferenceClient{
		conn:   conn,
		client: pb.NewInferenceServiceClient(conn),
	}, nil
}

// Predict sends a feature sequence to the model and returns mixture params + calibrated probability.
func (ic *InferenceClient) Predict(ctx context.Context, features []float32, timestampUs int64) (*PriceTarget, float64, float64, float64, error) {
	req := &pb.PredictRequest{
		Features:    features,
		TimestampUs: timestampUs,
	}

	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	resp, err := ic.client.Predict(ctx, req)
	if err != nil {
		return nil, 0, 0, 0, fmt.Errorf("inference failed: %w", err)
	}

	pt := &PriceTarget{
		ProbUp:    float64(resp.ProbUp),
		Alpha:     float64(resp.ProbUp) - 0.5,
		Timestamp: time.Now(),
	}

	for i := 0; i < 4 && i < len(resp.Pi); i++ {
		pt.MixturePi[i] = float64(resp.Pi[i])
		pt.MixtureMu[i] = float64(resp.Mu[i])
		pt.MixtureSigma[i] = float64(resp.Sigma[i])
		pt.MixtureNu[i] = float64(resp.Nu[i])
	}

	return pt, float64(resp.XgbProb), float64(resp.XgbUncertainty), float64(resp.BsImpliedProb), nil
}

// Close shuts down the gRPC connection.
func (ic *InferenceClient) Close() error {
	if ic.conn != nil {
		return ic.conn.Close()
	}
	return nil
}
