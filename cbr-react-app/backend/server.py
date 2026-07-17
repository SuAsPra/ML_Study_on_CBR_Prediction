import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "final_cbr_model_with_pl.pkl"
IMPUTER_PATH = BASE_DIR / "imputer_with_pl.pkl"
FEATURES_PATH = BASE_DIR / "features_with_pl.pkl"

MODEL = joblib.load(MODEL_PATH)
IMPUTER = joblib.load(IMPUTER_PATH)
FEATURES = joblib.load(FEATURES_PATH)
EXPLAINER = shap.TreeExplainer(MODEL)


def _to_number(value: Any) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return np.nan
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return np.nan
    try:
        return float(match.group())
    except ValueError:
        return np.nan


def _safe_div(numerator: float, denominator: float) -> float:
    if np.isnan(numerator) or np.isnan(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def _is_cohesive(ucscs: str) -> bool:
    text = str(ucscs or "").strip().upper()
    if not text:
        return False

    if "CLAY" in text or "SILT" in text:
        return True

    cohesive_codes = {"CL", "CH", "ML", "MH", "OL", "OH", "PT", "CL-ML", "ML-CL"}
    parts = [part for part in re.split(r"[^A-Z-]+", text) if part]
    return any(part in cohesive_codes for part in parts)


def _clamp_cbr(value: float) -> float:
    return float(max(0.0, min(100.0, value)))


def _uncertainty_band(predicted_cbr: float) -> tuple[float, float, float]:
    # Heuristic uncertainty band for single-point inference display.
    delta = float(min(7.5, max(1.2, 0.14 * max(predicted_cbr, 1.0))))
    low = _clamp_cbr(predicted_cbr - delta)
    high = _clamp_cbr(predicted_cbr + delta)
    return delta, low, high


def _top_shap_factors(model_input_imputed_df: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
    shap_values = EXPLAINER.shap_values(model_input_imputed_df)
    values = np.array(shap_values, dtype=float).reshape(-1)
    feature_values = model_input_imputed_df.iloc[0].to_numpy(dtype=float)

    ranked_indices = np.argsort(np.abs(values))[::-1][:top_n]
    factors: list[dict[str, Any]] = []
    for idx in ranked_indices:
        shap_val = float(values[idx])
        factors.append(
            {
                "feature": FEATURES[idx],
                "feature_value": round(float(feature_values[idx]), 4),
                "shap_value": round(shap_val, 4),
                "direction": "increases CBR" if shap_val >= 0 else "decreases CBR",
                "impact_strength": round(abs(shap_val), 4),
            }
        )
    return factors


def predict_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required_inputs = {
        "ll": "LL (%)",
        "pl": "PL (%)",
        "omc": "OMC (%)",
        "mdd": "MDD (g/cm3)",
        "gravel": "Gravel (%)",
        "sand": "Sand (%)",
        "fines": "Fines",
    }
    numeric_inputs = {key: _to_number(payload.get(key)) for key in required_inputs}
    missing_inputs = [
        label for key, label in required_inputs.items() if np.isnan(numeric_inputs[key])
    ]
    if missing_inputs:
        raise ValueError("Compulsory inputs missing: " + ", ".join(missing_inputs))

    ll = numeric_inputs["ll"]
    pl = numeric_inputs["pl"]
    omc = numeric_inputs["omc"]
    mdd = numeric_inputs["mdd"]
    fines = numeric_inputs["fines"]
    sand = numeric_inputs["sand"]
    gravel = numeric_inputs["gravel"]

    model_row = {
        "LL": ll,
        "PL": pl,
        "OMC": omc,
        "MDD": mdd,
        "Fines": fines,
        "Sand": sand,
        "Gravel": gravel,
        "OMC_MDD": _safe_div(omc, mdd),
        "LL_MDD": _safe_div(ll, mdd),
        "LL_OMC": ll * omc if not np.isnan(ll) and not np.isnan(omc) else np.nan,
    }

    model_input = pd.DataFrame([model_row], columns=FEATURES)
    model_input_imputed = IMPUTER.transform(model_input)
    model_input_imputed_df = pd.DataFrame(model_input_imputed, columns=FEATURES)
    raw_prediction = float(MODEL.predict(model_input_imputed_df)[0])
    clamped_prediction = _clamp_cbr(raw_prediction)
    uncertainty, range_low, range_high = _uncertainty_band(clamped_prediction)
    top_factors = _top_shap_factors(model_input_imputed_df, top_n=5)

    is_cohesive = _is_cohesive(payload.get("ucscs", ""))
    warning = None
    if payload.get("ucscs") and not is_cohesive:
        warning = (
            "Model was trained on cohesive soils only (clay/silt). "
            "Prediction for non-cohesive soils may be unreliable."
        )

    return {
        "predicted_cbr": round(clamped_prediction, 4),
        "raw_predicted_cbr": round(raw_prediction, 4),
        "uncertainty_plus_minus": round(uncertainty, 4),
        "predicted_range_min": round(range_low, 4),
        "predicted_range_max": round(range_high, 4),
        "cohesive_input": is_cohesive,
        "warning": warning,
        "top_factors": top_factors,
        "used_features": FEATURES,
    }


class CBRRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, data: dict[str, Any]) -> None:
        response = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(response)

    def do_OPTIONS(self) -> None:
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if self.path == "/features":
            self._send_json(200, {"features": FEATURES})
            return
        self._send_json(
            200,
            {
                "message": "CBR Predictor API is running",
                "endpoints": ["/health", "/features", "/predict"],
            },
        )

    def do_POST(self) -> None:
        if self.path != "/predict":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            payload = json.loads(body.decode("utf-8"))
            result = predict_from_payload(payload)
            self._send_json(200, result)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON body"})
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    httpd = HTTPServer((host, port), CBRRequestHandler)
    print(f"CBR Predictor API running on http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
