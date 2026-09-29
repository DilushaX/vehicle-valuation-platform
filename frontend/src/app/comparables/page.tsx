"use client";

import { useState } from "react";
import { Search, Sliders, Car, Sparkles, Filter, CheckCircle2, ChevronRight, BarChart2, RefreshCw, AlertCircle } from "lucide-react";

interface ComparableItem {
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
}

interface MarketSummary {
  min: number;
  median: number;
  mean: number;
  max: number;
  count: number;
}

export default function ComparablesPage() {
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
    top_k: 6,
  });

  const [weights, setWeights] = useState({
    brand: 0.25,
    model: 0.25,
    year: 0.14,
    mileage: 0.10,
    engine_cc: 0.08,
    transmission: 0.06,
    fuel_type: 0.04,
    district: 0.04,
    condition: 0.04,
  });

  const [showWeights, setShowWeights] = useState(false);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<ComparableItem[] | null>(null);
  const [marketSummary, setMarketSummary] = useState<MarketSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  const presets = [
    { label: "Toyota Premio 2016", category: "Cars", brand: "Toyota", model: "Premio", year: 2016, mileage: 85000, cc: 1500, fuel: "Petrol", trans: "Automatic", district: "Colombo" },
    { label: "Honda Vezel 2018", category: "SUVs", brand: "Honda", model: "Vezel", year: 2018, mileage: 65000, cc: 1500, fuel: "Hybrid", trans: "Automatic", district: "Gampaha" },
    { label: "Suzuki Alto 2015", category: "Cars", brand: "Suzuki", model: "Alto", year: 2015, mileage: 95000, cc: 800, fuel: "Petrol", trans: "Manual", district: "Kandy" },
    { label: "Hyundai H100 2000", category: "Vans", brand: "Hyundai", model: "H100", year: 2000, mileage: 260000, cc: 2700, fuel: "Diesel", trans: "Manual", district: "Gampaha" },
  ];

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const res = await fetch("/api/comparables", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...formData,
          weights,
        }),
      });

      if (!res.ok) throw new Error("Failed to search comparable vehicles");
      const data = await res.json();
      setResults(data.comparables);
      setMarketSummary(data.market_summary);
    } catch (err: any) {
      setError(err.message || "Failed to search comparables");
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
      {/* Header */}
      <div style={{ marginBottom: "2rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.5rem" }}>
          <span className="badge badge-purple">Phase 10.4 Dedicated Search</span>
          <span className="badge badge-cyan">Weighted Attribute Matching</span>
        </div>
        <h1 style={{ fontFamily: "var(--font-heading)", fontSize: "2.2rem", fontWeight: 700 }}>
          Comparable Vehicle Search Engine
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", maxWidth: "800px", marginTop: "0.25rem" }}>
          Retrieve peer vehicle listings matching query specifications through multi-attribute weighted similarity scoring across the verified marketplace.
        </p>
      </div>

      {/* Quick Presets */}
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1.75rem" }}>
        <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", fontWeight: 600 }}>Quick Queries:</span>
        {presets.map((p, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => {
              setFormData({
                ...formData,
                category: p.category,
                brand: p.brand,
                model: p.model,
                manufacture_year: p.year,
                mileage: p.mileage,
                engine_cc: p.cc,
                fuel_type: p.fuel,
                transmission: p.trans,
                district: p.district,
              });
            }}
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
          >
            {p.label}
          </button>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(350px, 1fr))", gap: "2rem", alignItems: "start" }}>
        {/* Search Specification Form */}
        <div className="glass-panel" style={{ padding: "1.75rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
              <Search size={20} color="var(--accent-purple)" />
              <h2 style={{ fontFamily: "var(--font-heading)", fontSize: "1.25rem", fontWeight: 600 }}>
                Query Specifications
              </h2>
            </div>
            <button
              type="button"
              onClick={() => setShowWeights(!showWeights)}
              style={{
                background: "transparent",
                border: "none",
                color: "var(--accent-cyan)",
                fontSize: "0.8rem",
                fontWeight: 600,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: "0.3rem",
              }}
            >
              <Sliders size={14} />
              <span>{showWeights ? "Hide Weights" : "Adjust Weights"}</span>
            </button>
          </div>

          <form onSubmit={handleSearch} style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
              <div>
                <label style={{ display: "block", fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "0.35rem" }}>Category</label>
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
                <label style={{ display: "block", fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "0.35rem" }}>Make / Brand</label>
                <input
                  type="text"
                  className="input-field"
                  value={formData.brand}
                  onChange={(e) => setFormData({ ...formData, brand: e.target.value })}
                  required
                />
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
              <div>
                <label style={{ display: "block", fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "0.35rem" }}>Model</label>
                <input
                  type="text"
                  className="input-field"
                  value={formData.model}
                  onChange={(e) => setFormData({ ...formData, model: e.target.value })}
                  required
                />
              </div>

              <div>
                <label style={{ display: "block", fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "0.35rem" }}>Manufacture Year</label>
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
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
              <div>
                <label style={{ display: "block", fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "0.35rem" }}>Mileage (km)</label>
                <input
                  type="number"
                  min="0"
                  className="input-field"
                  value={formData.mileage}
                  onChange={(e) => setFormData({ ...formData, mileage: parseInt(e.target.value, 10) || 0 })}
                  required
                />
              </div>

              <div>
                <label style={{ display: "block", fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "0.35rem" }}>Engine CC</label>
                <input
                  type="number"
                  min="0"
                  className="input-field"
                  value={formData.engine_cc}
                  onChange={(e) => setFormData({ ...formData, engine_cc: parseInt(e.target.value, 10) || 0 })}
                  required
                />
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
              <div>
                <label style={{ display: "block", fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "0.35rem" }}>Fuel Type</label>
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

              <div>
                <label style={{ display: "block", fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "0.35rem" }}>Transmission</label>
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
            </div>

            {/* Custom Weight Tuning (Phase 10.4 Feature) */}
            {showWeights && (
              <div style={{
                background: "rgba(10, 15, 29, 0.75)",
                padding: "1.25rem",
                borderRadius: "var(--radius-md)",
                border: "1px solid var(--border-subtle)",
                marginTop: "0.5rem",
                display: "flex",
                flexDirection: "column",
                gap: "0.75rem",
              }}>
                <div style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--accent-purple)", marginBottom: "0.25rem" }}>
                  Specification Similarity Weights
                </div>
                {[
                  { key: "brand", label: "Brand Weight", val: weights.brand },
                  { key: "model", label: "Model Weight", val: weights.model },
                  { key: "year", label: "Year Decay Weight", val: weights.year },
                  { key: "mileage", label: "Mileage Distance Weight", val: weights.mileage },
                  { key: "engine_cc", label: "Engine CC Weight", val: weights.engine_cc },
                  { key: "transmission", label: "Transmission Weight", val: weights.transmission },
                ].map((wt) => (
                  <div key={wt.key} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "1rem" }}>
                    <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>{wt.label}</span>
                    <input
                      type="range"
                      min="0.0"
                      max="0.5"
                      step="0.01"
                      value={wt.val}
                      onChange={(e) => setWeights({ ...weights, [wt.key]: parseFloat(e.target.value) })}
                      style={{ width: "120px", accentColor: "var(--accent-cyan)" }}
                    />
                    <span style={{ fontSize: "0.75rem", color: "var(--accent-cyan)", width: "35px", textAlign: "right" }}>
                      {wt.val.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="btn btn-primary"
              style={{ width: "100%", marginTop: "0.75rem", padding: "0.85rem" }}
            >
              {loading ? (
                <>
                  <RefreshCw size={18} className="status-dot-pulse" />
                  <span>Searching Similar Listings...</span>
                </>
              ) : (
                <>
                  <Search size={18} />
                  <span>Execute Comparable Search</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Results Display */}
        <div>
          {error && (
            <div className="glass-panel" style={{ padding: "1.25rem", border: "1px solid var(--accent-rose)", marginBottom: "1.5rem" }}>
              <div style={{ color: "var(--accent-rose)", fontWeight: 600 }}>Search Error</div>
              <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginTop: "0.25rem" }}>{error}</p>
            </div>
          )}

          {!results && !loading && (
            <div className="glass-panel" style={{ padding: "3.5rem 2rem", textAlign: "center" }}>
              <div style={{
                width: "56px",
                height: "56px",
                borderRadius: "16px",
                background: "rgba(168, 85, 247, 0.1)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                margin: "0 auto 1.25rem",
              }}>
                <Search size={28} color="var(--accent-purple)" />
              </div>
              <h3 style={{ fontFamily: "var(--font-heading)", fontSize: "1.25rem", fontWeight: 600 }}>
                Awaiting Search Query
              </h3>
              <p style={{ color: "var(--text-muted)", fontSize: "0.875rem", maxWidth: "420px", margin: "0.5rem auto 1.5rem" }}>
                Select query attributes or a quick preset and click Execute Comparable Search to find closest matches based on weighted feature distances.
              </p>
              <button
                type="button"
                onClick={() => handleSearch()}
                className="btn btn-secondary"
                style={{ fontSize: "0.85rem" }}
              >
                Search Default Toyota Premio
              </button>
            </div>
          )}

          {results && (
            <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
              {/* Market Summary Card */}
              {marketSummary && (
                <div className="glass-panel" style={{ padding: "1.25rem 1.5rem" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
                    <span style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--text-primary)" }}>
                      Comparable Cohort Asking Price Summary
                    </span>
                    <span className="badge badge-purple">{results.length} Matching Peers</span>
                  </div>

                  <div style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr 1fr 1fr",
                    gap: "0.75rem",
                    textAlign: "center",
                    padding: "0.75rem",
                    background: "rgba(10, 15, 29, 0.6)",
                    borderRadius: "var(--radius-md)",
                  }}>
                    <div>
                      <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Minimum</div>
                      <div style={{ fontSize: "0.95rem", fontWeight: 700, color: "var(--text-secondary)", marginTop: "0.15rem" }}>
                        {formatLKR(marketSummary.min)}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.72rem", color: "var(--accent-cyan)", fontWeight: 600 }}>Median</div>
                      <div style={{ fontSize: "0.95rem", fontWeight: 700, color: "var(--accent-cyan)", marginTop: "0.15rem" }}>
                        {formatLKR(marketSummary.median)}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Mean</div>
                      <div style={{ fontSize: "0.95rem", fontWeight: 700, color: "var(--text-secondary)", marginTop: "0.15rem" }}>
                        {formatLKR(marketSummary.mean)}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Maximum</div>
                      <div style={{ fontSize: "0.95rem", fontWeight: 700, color: "var(--text-secondary)", marginTop: "0.15rem" }}>
                        {formatLKR(marketSummary.max)}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Matched Listings Cards */}
              <div style={{ display: "flex", flexDirection: "column", gap: "0.85rem" }}>
                {results.map((item, idx) => (
                  <div key={idx} className="glass-panel glass-panel-hoverable" style={{ padding: "1.25rem 1.5rem" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "0.5rem" }}>
                      <div>
                        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
                          <span style={{ fontFamily: "var(--font-heading)", fontSize: "1.05rem", fontWeight: 700, color: "#ffffff" }}>
                            {item.manufacture_year} {item.brand} {item.model}
                          </span>
                          <span className="badge badge-emerald">
                            {item.similarity_percentage}% Match
                          </span>
                        </div>
                        <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: "0.35rem" }}>
                          {item.mileage ? `${item.mileage.toLocaleString()} km` : "Odo N/A"} • {item.engine_cc} cc • {item.fuel_type} • {item.transmission} • {item.district}
                        </div>
                      </div>

                      <div style={{ textAlign: "right" }}>
                        <div style={{ fontFamily: "var(--font-heading)", fontSize: "1.25rem", fontWeight: 700, color: "var(--accent-cyan)" }}>
                          {formatLKR(item.asking_price)}
                        </div>
                        <div style={{ fontSize: "0.72rem", color: "var(--text-dim)", marginTop: "0.15rem" }}>
                          Riyasewana ID: {item.listing_id}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
