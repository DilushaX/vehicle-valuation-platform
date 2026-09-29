"use client";

import { useState } from "react";
import { Car, Sliders, Sparkles, CheckCircle2, AlertTriangle, ArrowRight, Copy, Check, BarChart2, ShieldAlert, Info, RefreshCw } from "lucide-react";

interface ValuationResponse {
  estimated_asking_price_lkr: number;
  prediction_range_lkr: {
    estimate: number;
    lower: number;
    upper: number;
    spread: number;
    percentile_lower: number;
    percentile_upper: number;
    method: string;
  };
  currency: string;
  model: {
    name: string;
    target_variable: string;
    target_transform: string;
    training_records: number;
    test_samples: number;
    status: string;
  };
  explanation: Array<{
    feature: string;
    value: string;
    contribution: number;
    direction: "positive" | "negative";
    description: string;
  }>;
  comparables: Array<{
    listing_id: string;
    category: string;
    brand: string;
    model: string;
    manufacture_year: number;
    mileage: number;
    engine_cc: number;
    fuel_type: string;
    transmission: string;
    district: string;
    condition: string;
    asking_price: number;
    similarity_score: number;
    similarity_percentage: number;
  }>;
  limitations: string[];
}

export default function ValuationPage() {
  const [formData, setFormData] = useState({
    category: "Cars",
    brand: "Toyota",
    model: "Premio",
    manufacture_year: 2016,
    mileage: 85000,
    engine_cc: 1500,
    fuel_type: "Petrol",
    transmission: "Automatic",
    district: "Colombo",
    condition: "Registered (Used)",
    top_k_factors: 4,
    top_k_comparables: 3,
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ValuationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copiedFingerprint, setCopiedFingerprint] = useState(false);

  const presets = [
    { label: "Toyota Premio 2016", category: "Cars", brand: "Toyota", model: "Premio", year: 2016, mileage: 85000, cc: 1500, fuel: "Petrol", trans: "Automatic", district: "Colombo" },
    { label: "Honda Vezel 2018", category: "SUVs", brand: "Honda", model: "Vezel", year: 2018, mileage: 65000, cc: 1500, fuel: "Hybrid", trans: "Automatic", district: "Gampaha" },
    { label: "Toyota Prado 2024", category: "SUVs", brand: "Toyota", model: "Land Cruiser Prado", year: 2024, mileage: 8000, cc: 2800, fuel: "Diesel", trans: "Automatic", district: "Colombo" },
    { label: "Suzuki Alto 2015", category: "Cars", brand: "Suzuki", model: "Alto", year: 2015, mileage: 95000, cc: 800, fuel: "Petrol", trans: "Manual", district: "Kandy" },
  ];

  const applyPreset = (p: typeof presets[0]) => {
    setFormData((prev) => ({
      ...prev,
      category: p.category,
      brand: p.brand,
      model: p.model,
      manufacture_year: p.year,
      mileage: p.mileage,
      engine_cc: p.cc,
      fuel_type: p.fuel,
      transmission: p.trans,
      district: p.district,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const res = await fetch("/api/py/api/valuation/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => null);
        throw new Error(errJson?.detail || `API request failed with status ${res.status}`);
      }

      const data: ValuationResponse = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to execute vehicle valuation.");
    } finally {
      setLoading(false);
    }
  };

  const formatLKR = (val?: number) => {
    if (val === undefined || isNaN(val)) return "—";
    return new Intl.NumberFormat("en-LK", {
      style: "currency",
      currency: "LKR",
      maximumFractionDigits: 0,
    }).format(val).replace("LKR", "Rs.");
  };

  return (
    <div style={{ maxWidth: "1400px", margin: "0 auto", padding: "2rem 1.5rem 4rem", width: "100%" }}>
      {/* Page Header */}
      <div style={{ marginBottom: "2rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.5rem" }}>
          <span className="badge badge-cyan">Phase 10.2 & 10.5 Workflow</span>
          <span className="badge badge-purple">RandomForest (300 Trees)</span>
        </div>
        <h1 style={{ fontFamily: "var(--font-heading)", fontSize: "2.2rem", fontWeight: 700 }}>
          Vehicle Valuation & Explainable AI
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", maxWidth: "800px", marginTop: "0.25rem" }}>
          Estimate market asking prices with empirical decision-tree dispersion intervals and exact Tree SHAP factor attributions in original Sri Lankan Rupees.
        </p>
      </div>

      {/* Preset Buttons */}
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1.75rem" }}>
        <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", fontWeight: 600 }}>Quick Presets:</span>
        {presets.map((p, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => applyPreset(p)}
            style={{
              padding: "0.35rem 0.75rem",
              borderRadius: "var(--radius-full)",
              background: "rgba(255, 255, 255, 0.05)",
              border: "1px solid var(--border-subtle)",
              color: "var(--text-secondary)",
              fontSize: "0.78rem",
              fontWeight: 500,
              cursor: "pointer",
              transition: "all 0.2s ease",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = "var(--accent-cyan)";
              e.currentTarget.style.color = "#ffffff";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = "var(--border-subtle)";
              e.currentTarget.style.color = "var(--text-secondary)";
            }}
          >
            {p.label}
          </button>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))", gap: "2rem", alignItems: "start" }}>
        {/* Valuation Input Form */}
        <div className="glass-panel" style={{ padding: "1.75rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "1.5rem" }}>
            <Sliders size={20} color="var(--accent-cyan)" />
            <h2 style={{ fontFamily: "var(--font-heading)", fontSize: "1.25rem", fontWeight: 600 }}>
              Vehicle Specifications
            </h2>
          </div>

          <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "1.1rem" }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
              <div>
                <label style={{ display: "block", fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "0.35rem", fontWeight: 500 }}>
                  Category
                </label>
                <select
                  className="input-field"
                  value={formData.category}
                  onChange={(e) => setFormData({ ...formData, category: e.target.value })}
                >
                  {["Cars", "SUVs", "Vans", "Pickups", "Motorbikes", "Three Wheelers", "Heavy-Duty", "Lorries"].map((c) => (
                    <option key={c} value={c} style={{ background: "#0b1120" }}>{c}</option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: "block", fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "0.35rem", fontWeight: 500 }}>
                  Condition
                </label>
                <select
                  className="input-field"
                  value={formData.condition}
                  onChange={(e) => setFormData({ ...formData, condition: e.target.value })}
                >
                  {["Registered (Used)", "Unregistered", "Brand New"].map((cond) => (
                    <option key={cond} value={cond} style={{ background: "#0b1120" }}>{cond}</option>
                  ))}
                </select>
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
              <div>
                <label style={{ display: "block", fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "0.35rem", fontWeight: 500 }}>
                  Brand / Make
                </label>
                <input
                  type="text"
                  className="input-field"
                  value={formData.brand}
                  onChange={(e) => setFormData({ ...formData, brand: e.target.value })}
                  required
                />
              </div>

              <div>
                <label style={{ display: "block", fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "0.35rem", fontWeight: 500 }}>
                  Model
                </label>
                <input
                  type="text"
                  className="input-field"
                  value={formData.model}
                  onChange={(e) => setFormData({ ...formData, model: e.target.value })}
                  required
                />
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
              <div>
                <label style={{ display: "block", fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "0.35rem", fontWeight: 500 }}>
                  Manufacture Year
                </label>
                <input
                  type="number"
                  min="1950"
                  max="2026"
                  className="input-field"
                  value={formData.manufacture_year}
                  onChange={(e) => setFormData({ ...formData, manufacture_year: parseInt(e.target.value, 10) || 2016 })}
                  required
                />
              </div>

              <div>
                <label style={{ display: "block", fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "0.35rem", fontWeight: 500 }}>
                  Mileage (km)
                </label>
                <input
                  type="number"
                  min="0"
                  max="1500000"
                  className="input-field"
                  value={formData.mileage}
                  onChange={(e) => setFormData({ ...formData, mileage: parseInt(e.target.value, 10) || 0 })}
                  required
                />
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
              <div>
                <label style={{ display: "block", fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "0.35rem", fontWeight: 500 }}>
                  Engine Capacity (cc)
                </label>
                <input
                  type="number"
                  min="0"
                  max="16000"
                  className="input-field"
                  value={formData.engine_cc}
                  onChange={(e) => setFormData({ ...formData, engine_cc: parseInt(e.target.value, 10) || 0 })}
                  required
                />
              </div>

              <div>
                <label style={{ display: "block", fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "0.35rem", fontWeight: 500 }}>
                  Fuel Type
                </label>
                <select
                  className="input-field"
                  value={formData.fuel_type}
                  onChange={(e) => setFormData({ ...formData, fuel_type: e.target.value })}
                >
                  {["Petrol", "Diesel", "Hybrid", "Electric"].map((f) => (
                    <option key={f} value={f} style={{ background: "#0b1120" }}>{f}</option>
                  ))}
                </select>
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
              <div>
                <label style={{ display: "block", fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "0.35rem", fontWeight: 500 }}>
                  Transmission
                </label>
                <select
                  className="input-field"
                  value={formData.transmission}
                  onChange={(e) => setFormData({ ...formData, transmission: e.target.value })}
                >
                  {["Automatic", "Manual", "Tiptronic"].map((t) => (
                    <option key={t} value={t} style={{ background: "#0b1120" }}>{t}</option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: "block", fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "0.35rem", fontWeight: 500 }}>
                  District
                </label>
                <select
                  className="input-field"
                  value={formData.district}
                  onChange={(e) => setFormData({ ...formData, district: e.target.value })}
                >
                  {[
                    "Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "Nuwara-Eliya",
                    "Galle", "Matara", "Hambantota", "Jaffna", "Kurunegala", "Puttalam",
                    "Anuradhapura", "Polonnaruwa", "Badulla", "Monaragala", "Ratnapura",
                    "Kegalle", "Trincomalee", "Batticaloa", "Ampara"
                  ].map((d) => (
                    <option key={d} value={d} style={{ background: "#0b1120" }}>{d}</option>
                  ))}
                </select>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="btn btn-primary"
              style={{ width: "100%", marginTop: "0.75rem", padding: "0.85rem", fontSize: "1rem" }}
            >
              {loading ? (
                <>
                  <RefreshCw size={18} className="status-dot-pulse" />
                  <span>Computing RandomForest Valuation...</span>
                </>
              ) : (
                <>
                  <Sparkles size={18} />
                  <span>Calculate Valuation</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Results Area */}
        <div>
          {error && (
            <div className="glass-panel" style={{
              padding: "1.25rem",
              background: "rgba(244, 63, 94, 0.08)",
              border: "1px solid rgba(244, 63, 94, 0.3)",
              marginBottom: "1.5rem",
              display: "flex",
              alignItems: "flex-start",
              gap: "0.75rem",
            }}>
              <AlertTriangle size={20} color="var(--accent-rose)" style={{ flexShrink: 0, marginTop: "2px" }} />
              <div>
                <strong style={{ color: "var(--accent-rose)", fontSize: "0.95rem" }}>Valuation Error</strong>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginTop: "0.25rem" }}>
                  {error}
                </p>
              </div>
            </div>
          )}

          {!result && !loading && !error && (
            <div className="glass-panel" style={{ padding: "3rem 2rem", textAlign: "center" }}>
              <div style={{
                width: "56px",
                height: "56px",
                borderRadius: "16px",
                background: "rgba(56, 189, 248, 0.1)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                margin: "0 auto 1.25rem",
              }}>
                <Car size={28} color="var(--accent-cyan)" />
              </div>
              <h3 style={{ fontFamily: "var(--font-heading)", fontSize: "1.25rem", fontWeight: 600 }}>
                Ready to Compute Valuation
              </h3>
              <p style={{ color: "var(--text-muted)", fontSize: "0.875rem", maxWidth: "420px", margin: "0.5rem auto 1.5rem" }}>
                Fill in the vehicle specifications or select a quick preset above, then click Calculate Valuation to inspect the asking price, 300-tree dispersion, and Tree SHAP factors.
              </p>
              <button
                type="button"
                onClick={() => applyPreset(presets[0])}
                className="btn btn-secondary"
                style={{ fontSize: "0.85rem" }}
              >
                Load {presets[0].label}
              </button>
            </div>
          )}

          {result && (
            <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
              {/* Hero Price Card */}
              <div className="glass-panel" style={{
                padding: "2rem",
                background: "linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(2, 132, 199, 0.15) 100%)",
                border: "1px solid rgba(56, 189, 248, 0.3)",
                boxShadow: "0 10px 30px -5px rgba(2, 132, 199, 0.25)",
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "0.75rem", marginBottom: "0.75rem" }}>
                  <span style={{ fontSize: "0.8rem", color: "var(--accent-cyan)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                    Estimated Market Asking Price
                  </span>
                  <span className="badge badge-purple">
                    {result.model.status}
                  </span>
                </div>

                <div style={{
                  fontFamily: "var(--font-heading)",
                  fontSize: "clamp(2rem, 4vw, 2.75rem)",
                  fontWeight: 800,
                  color: "#ffffff",
                  letterSpacing: "-0.02em",
                  lineHeight: 1.1,
                  marginBottom: "0.5rem",
                }}>
                  {formatLKR(result.estimated_asking_price_lkr)}
                </div>

                <div style={{ fontSize: "0.825rem", color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: "0.5rem", flexWrap: "wrap" }}>
                  <span>{formData.manufacture_year} {formData.brand} {formData.model}</span>
                  <span>•</span>
                  <span>{formData.mileage.toLocaleString()} km</span>
                  <span>•</span>
                  <span>{formData.fuel_type}</span>
                  <span>•</span>
                  <span>{formData.transmission}</span>
                </div>
              </div>

              {/* Indicative Range Card */}
              {result.prediction_range_lkr && (
                <div className="glass-panel" style={{ padding: "1.5rem" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      <BarChart2 size={18} color="var(--accent-amber)" />
                      <span style={{ fontWeight: 600, fontSize: "0.95rem" }}>Indicative Prediction Range</span>
                    </div>
                    <span className="badge badge-amber">10th – 90th Percentile Dispersion</span>
                  </div>

                  <div style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr 1fr",
                    gap: "0.75rem",
                    textAlign: "center",
                    padding: "1rem",
                    background: "rgba(10, 15, 29, 0.6)",
                    borderRadius: "var(--radius-md)",
                    marginBottom: "0.75rem",
                  }}>
                    <div>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Lower Bound (10th)</div>
                      <div style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-primary)", marginTop: "0.2rem" }}>
                        {formatLKR(result.prediction_range_lkr.lower)}
                      </div>
                    </div>
                    <div style={{ borderLeft: "1px solid var(--border-subtle)", borderRight: "1px solid var(--border-subtle)" }}>
                      <div style={{ fontSize: "0.75rem", color: "var(--accent-cyan)", fontWeight: 600 }}>Point Estimate</div>
                      <div style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--accent-cyan)", marginTop: "0.2rem" }}>
                        {formatLKR(result.prediction_range_lkr.estimate)}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Upper Bound (90th)</div>
                      <div style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-primary)", marginTop: "0.2rem" }}>
                        {formatLKR(result.prediction_range_lkr.upper)}
                      </div>
                    </div>
                  </div>

                  <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", lineHeight: 1.4 }}>
                    <Info size={12} style={{ display: "inline", verticalAlign: "middle", marginRight: "4px" }} />
                    Dispersion spread: {formatLKR(result.prediction_range_lkr.spread)}. Represents empirical tree disagreement across 300 decision trees, NOT a statistically guaranteed confidence interval.
                  </div>
                </div>
              )}

              {/* Tree SHAP Factor Attribution */}
              {result.explanation && result.explanation.length > 0 && (
                <div className="glass-panel" style={{ padding: "1.5rem" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      <Sparkles size={18} color="var(--accent-purple)" />
                      <span style={{ fontWeight: 600, fontSize: "0.95rem" }}>Tree SHAP Factor Attribution</span>
                    </div>
                    <span className="badge badge-purple">Additive Contributions</span>
                  </div>

                  <div style={{ display: "flex", flexDirection: "column", gap: "0.85rem" }}>
                    {result.explanation.map((exp, idx) => {
                      const isPos = exp.direction === "positive";
                      const maxAbs = Math.max(...result.explanation.map((e) => Math.abs(e.contribution))) || 1;
                      const barPct = Math.min(100, Math.round((Math.abs(exp.contribution) / maxAbs) * 100));

                      return (
                        <div key={idx} style={{
                          background: "rgba(10, 15, 29, 0.6)",
                          padding: "0.85rem 1rem",
                          borderRadius: "var(--radius-md)",
                          border: "1px solid var(--border-subtle)",
                        }}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.4rem" }}>
                            <div>
                              <strong style={{ fontSize: "0.85rem", textTransform: "capitalize", color: "#ffffff" }}>
                                {exp.feature.replace("_", " ")}
                              </strong>
                              <span style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginLeft: "0.5rem" }}>
                                ({exp.value})
                              </span>
                            </div>
                            <span style={{
                              fontSize: "0.85rem",
                              fontWeight: 700,
                              color: isPos ? "var(--accent-emerald)" : "var(--accent-rose)",
                            }}>
                              {isPos ? "+" : "-"}{Math.abs(exp.contribution).toFixed(4)}
                            </span>
                          </div>

                          {/* Contribution Bar */}
                          <div style={{ width: "100%", height: "6px", background: "rgba(255, 255, 255, 0.05)", borderRadius: "3px", overflow: "hidden", marginBottom: "0.4rem" }}>
                            <div style={{
                              width: `${barPct}%`,
                              height: "100%",
                              background: isPos ? "var(--accent-emerald)" : "var(--accent-rose)",
                              borderRadius: "3px",
                            }} />
                          </div>

                          <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
                            {exp.description}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Comparable Listings */}
              {result.comparables && result.comparables.length > 0 && (
                <div className="glass-panel" style={{ padding: "1.5rem" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      <Car size={18} color="var(--accent-cyan)" />
                      <span style={{ fontWeight: 600, fontSize: "0.95rem" }}>Top Matching Comparable Listings</span>
                    </div>
                    <span className="badge badge-cyan">{result.comparables.length} Verified Peers</span>
                  </div>

                  <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                    {result.comparables.map((comp, idx) => (
                      <div key={idx} style={{
                        background: "rgba(10, 15, 29, 0.6)",
                        padding: "0.85rem 1rem",
                        borderRadius: "var(--radius-md)",
                        border: "1px solid var(--border-subtle)",
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        flexWrap: "wrap",
                        gap: "0.75rem",
                      }}>
                        <div>
                          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                            <strong style={{ fontSize: "0.9rem", color: "#ffffff" }}>
                              {comp.manufacture_year} {comp.brand} {comp.model}
                            </strong>
                            <span className="badge badge-emerald">
                              {comp.similarity_percentage}% Match
                            </span>
                          </div>
                          <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
                            {comp.mileage ? `${comp.mileage.toLocaleString()} km` : "N/A"} • {comp.transmission} • {comp.district}
                          </div>
                        </div>

                        <div style={{ textAlign: "right" }}>
                          <div style={{ fontSize: "1rem", fontWeight: 700, color: "var(--accent-cyan)" }}>
                            {formatLKR(comp.asking_price)}
                          </div>
                          <div style={{ fontSize: "0.72rem", color: "var(--text-dim)" }}>
                            Ad ID: {comp.listing_id}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
