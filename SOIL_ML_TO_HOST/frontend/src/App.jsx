import { useState } from "react";
import "./App.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

const INPUT_FIELDS = [
  { key: "ll", label: "LL (%)", type: "number", required: true },
  { key: "pl", label: "PL (%)", type: "number", required: true },
  { key: "pi", label: "PI (%)", type: "number", required: false },
  { key: "omc", label: "OMC (%)", type: "number", required: true },
  { key: "mdd", label: "MDD (g/cm3)", type: "number", required: true },
  { key: "aashto", label: "AASHTO", type: "text", required: false },
  { key: "ucscs", label: "UCSCS", type: "text", required: false },
  { key: "specific_gravity", label: "SPECIFIC GRAVITY ((g/cm3))", type: "number", required: false },
  { key: "gravel", label: "Gravel (%)", type: "number", required: true },
  { key: "sand", label: "Sand (%)", type: "number", required: true },
  { key: "fines", label: "Fines", type: "number", required: true }
];

const initialForm = INPUT_FIELDS.reduce((acc, item) => {
  acc[item.key] = "";
  return acc;
}, {});

function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const onChange = (key, value) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const onSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    const missingRequired = INPUT_FIELDS.filter(
      (field) => field.required && String(form[field.key]).trim() === ""
    );

    if (missingRequired.length > 0) {
      setError(
        `Please fill all compulsory fields: ${missingRequired
          .map((field) => field.label)
          .join(", ")}`
      );
      setLoading(false);
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form)
      });

      const raw = await response.text();
      let data = {};
      if (raw) {
        try {
          data = JSON.parse(raw);
        } catch {
          data = { error: raw };
        }
      }

      if (!response.ok) {
        throw new Error(data.error || "Prediction failed. Check if backend is running on port 8000.");
      }

      if (!raw) {
        throw new Error("Empty response from backend. Check backend logs.");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container">
      <section className="card">
        <h1>CBR Value Predictor</h1>
        <p className="subtext">
          Uses your saved <code>final_cbr_model_with_pl.pkl</code> (trained for cohesive soils).
        </p>
        <p className="legend">
          <strong>Compulsory:</strong> LL, PL, OMC, MDD, Gravel, Sand, Fines. <strong>Optional:</strong>{" "}
          PI, AASHTO, UCSCS, SPECIFIC GRAVITY.
        </p>

        <form className="grid" onSubmit={onSubmit}>
          {INPUT_FIELDS.map((field) => (
            <label key={field.key} className="field">
              <span>
                {field.label}{" "}
                <em className={field.required ? "badge required" : "badge optional"}>
                  {field.required ? "Compulsory" : "Optional"}
                </em>
              </span>
              <input
                type={field.type}
                value={form[field.key]}
                onChange={(e) => onChange(field.key, e.target.value)}
                step={field.type === "number" ? "any" : undefined}
                placeholder={`Enter ${field.label}`}
                required={field.required}
              />
            </label>
          ))}

          <button type="submit" disabled={loading}>
            {loading ? "Predicting..." : "Predict CBR"}
          </button>
        </form>

        {error && <p className="error">{error}</p>}

        {result && (
          <div className="result">
            <h2>Prediction Result</h2>
            <p>
              Predicted CBR: <strong>{result.predicted_cbr}</strong>
            </p>
            <p>
              CBR estimate range:{" "}
              <strong>
                {result.predicted_cbr} +/- {result.uncertainty_plus_minus}
              </strong>
            </p>
            <p>
              Expected interval:{" "}
              <strong>
                {result.predicted_range_min} to {result.predicted_range_max}
              </strong>
            </p>
            {result.warning && <p className="warning">{result.warning}</p>}

            {Array.isArray(result.top_factors) && result.top_factors.length > 0 && (
              <div className="factors">
                <h3>Top Factors Affecting CBR (SHAP)</h3>
                <ul>
                  {result.top_factors.map((factor) => (
                    <li key={factor.feature}>
                      <strong>{factor.feature}</strong> ({factor.feature_value}) {"->"}{" "}
                      <strong>{factor.direction}</strong> (SHAP {factor.shap_value})
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </section>
    </main>
  );
}

export default App;
