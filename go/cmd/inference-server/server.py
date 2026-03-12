"""gRPC inference server for LSTM-MDN ensemble.

Usage:
    python server.py --models-dir ../../models --port 50051

Loads:
    - 5x LSTM-MDN PyTorch models (lstm_mdn_0.pt .. lstm_mdn_4.pt)
    - XGBoost binary classifier (xgb_classifier.pkl)
    - XGBoost quantile regressors (xgb_q10.pkl, xgb_q90.pkl)
    - Isotonic regressors for calibration (isotonic_*.pkl)
"""

import argparse
import logging
import pickle
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add proto path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import proto.inference_pb2 as pb2
import proto.inference_pb2_grpc as pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Definition (must match training code) ---

SEQ_LEN = 60
N_FEATURES = 60
HIDDEN_DIM = 128
N_LAYERS = 2
K = 4  # mixture components
DROPOUT = 0.2


class LSTMMDN(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(N_FEATURES)
        self.lstm = nn.LSTM(
            N_FEATURES, HIDDEN_DIM,
            num_layers=N_LAYERS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.fc1 = nn.Linear(HIDDEN_DIM, 64)
        self.fc2 = nn.Linear(64, K * 4)  # pi, mu, sigma, nu

    def forward(self, x):
        x = self.norm(x)
        out, _ = self.lstm(x)
        h = F.relu(self.fc1(out[:, -1, :]))
        params = self.fc2(h)

        pi = F.softmax(params[:, :K], dim=1)
        mu = params[:, K:2*K]
        sigma = F.softplus(params[:, 2*K:3*K]) + 1e-6
        nu = F.softplus(params[:, 3*K:4*K]) + 2.1

        return pi, mu, sigma, nu


class InferenceServicer(pb2_grpc.InferenceServiceServicer):
    def __init__(self, models_dir: Path):
        self.models = []
        self.xgb_clf = None
        self.xgb_q10 = None
        self.xgb_q90 = None
        self.isotonic = {}

        # Load LSTM-MDN ensemble
        for i in range(5):
            model_path = models_dir / f"lstm_mdn_{i}.pt"
            if model_path.exists():
                model = LSTMMDN()
                model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
                model.eval()
                self.models.append(model)
                logger.info(f"Loaded model {i}")
            else:
                logger.warning(f"Model {model_path} not found, skipping")

        if not self.models:
            logger.warning("No models loaded! Server will return dummy predictions.")

        # Load XGBoost models
        for name in ["xgb_classifier", "xgb_q10", "xgb_q90"]:
            path = models_dir / f"{name}.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    setattr(self, name.replace("classifier", "clf"), pickle.load(f))
                logger.info(f"Loaded {name}")

        # Load isotonic regressors
        for bucket in ["atm", "near_otm", "far_otm"]:
            path = models_dir / f"isotonic_regressor_{bucket}.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    self.isotonic[bucket] = pickle.load(f)
                logger.info(f"Loaded isotonic_{bucket}")

    def Predict(self, request, context):
        features = np.array(request.features, dtype=np.float32)

        if len(features) != SEQ_LEN * N_FEATURES:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(
                f"Expected {SEQ_LEN * N_FEATURES} features, got {len(features)}"
            )
            return pb2.PredictResponse()

        x = torch.from_numpy(features.reshape(1, SEQ_LEN, N_FEATURES))

        # XGBoost prior (uses last bar's features)
        last_bar = features[-N_FEATURES:]
        xgb_prob = 0.5
        xgb_unc = 0.0
        bs_prob = 0.5

        if self.xgb_clf is not None:
            tabular = last_bar[:45].reshape(1, -1)  # base features (before prices/xgb)
            xgb_prob = float(self.xgb_clf.predict_proba(tabular)[0, 1])
            if self.xgb_q10 is not None and self.xgb_q90 is not None:
                q10 = float(self.xgb_q10.predict(tabular)[0])
                q90 = float(self.xgb_q90.predict(tabular)[0])
                xgb_unc = q90 - q10
            # Simple BS implied prob approximation
            bs_prob = xgb_prob  # placeholder; full BS calc in production

        # Ensemble inference
        if self.models:
            all_pi, all_mu, all_sigma, all_nu = [], [], [], []
            with torch.no_grad():
                for model in self.models:
                    pi, mu, sigma, nu = model(x)
                    all_pi.append(pi)
                    all_mu.append(mu)
                    all_sigma.append(sigma)
                    all_nu.append(nu)

            # Average across ensemble
            pi = torch.stack(all_pi).mean(0).squeeze().numpy()
            mu = torch.stack(all_mu).mean(0).squeeze().numpy()
            sigma = torch.stack(all_sigma).mean(0).squeeze().numpy()
            nu = torch.stack(all_nu).mean(0).squeeze().numpy()
        else:
            # Dummy prediction
            pi = np.array([0.25, 0.25, 0.25, 0.25])
            mu = np.array([0.0, 0.0, 0.0, 0.0])
            sigma = np.array([0.01, 0.01, 0.01, 0.01])
            nu = np.array([5.0, 5.0, 5.0, 5.0])

        # Compute P(up) from mixture
        from scipy import stats as sp_stats
        prob_up = 0.0
        for k in range(K):
            z = (0 - mu[k]) / sigma[k]
            cdf_val = sp_stats.t.cdf(z, df=nu[k])
            prob_up += pi[k] * (1 - cdf_val)

        # Calibrate if isotonic regressors available
        if self.isotonic:
            bucket = "atm" if abs(prob_up - 0.5) < 0.10 else (
                "near_otm" if abs(prob_up - 0.5) < 0.30 else "far_otm"
            )
            if bucket in self.isotonic:
                prob_up = float(
                    self.isotonic[bucket].predict(np.array([[prob_up]]))[0]
                )

        return pb2.PredictResponse(
            pi=pi.tolist(),
            mu=mu.tolist(),
            sigma=sigma.tolist(),
            nu=nu.tolist(),
            prob_up=float(prob_up),
            xgb_prob=float(xgb_prob),
            xgb_uncertainty=float(xgb_unc),
            bs_implied_prob=float(bs_prob),
        )


def serve(models_dir: str, port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServicer(Path(models_dir)), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"Inference server listening on port {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="../../models")
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args()
    serve(args.models_dir, args.port)
