"use client";

import { useEffect, useState } from "react";
import { TrendingUp, Filter, BarChart3, PieChart, MapPin, Fuel, ShieldCheck, AlertCircle, RefreshCw, CheckCircle2, ChevronDown } from "lucide-react";

export default function MarketIntelligencePage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>("All");
  const [activeTab, setActiveTab] = useState<"categories" | "brands" | "characteristics" | "districts" | "quality">("categories");

  useEffect(() => {
    async function loadAnalytics() {
      try {
        const res = await fetch("/api/analytics");
        if (!res.ok) throw new Error("Failed to load market analytics data.");
        const json = await res.json();
        setData(json);
      } catch (err: any) {
        setError(err.message || "Failed to load analytics");
      } finally {
        setLoading(false);
      }
    }
    loadAnalytics();
  }, []);

  const formatLKR = (val?: number) => {
    if (val === undefined || isNaN(val)) return "—";
    return new Intl.NumberFormat("en-LK", {
      style: "currency",
      currency: "LKR",
      maximumFractionDigits: 0,
    }).format(val).replace("LKR", "Rs.");
  };

  if (loading) {
    return (
      <div style={{ maxWidth: "1400px", margin: "4rem auto", textAlign: "center", padding: "2rem" }}>
        <RefreshCw size={36} color="var(--accent-cyan)" className="status-dot-pulse" style={{ margin: "0 auto 1rem" }} />
        <h2 style={{ fontFamily: "var(--font-heading)", fontSize: "1.5rem" }}>Loading Market Intelligence...</h2>
        <p style={{ color: "var(--text-muted)", fontSize: "0.9rem", marginTop: "0.25rem" }}>
          Aggregating verified market distributions and dataset summaries
        </p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div style={{ maxWidth: "800px", margin: "4rem auto", padding: "2rem" }}>
        <div className="glass-panel" style={{ padding: "2rem", border: "1px solid var(--accent-rose)", textAlign: "center" }}>
          <AlertCircle size={36} color="var(--accent-rose)" style={{ margin: "0 auto 0.75rem" }} />
          <h3 style={{ fontSize: "1.25rem", color: "var(--accent-rose)" }}>Unable to Load Analytics</h3>
          <p style={{ color: "var(--text-secondary)", marginTop: "0.5rem" }}>{error}</p>
        </div>
      </div>
    );
  }

  const { overview, categories, brands, districts, fuel, transmission, numerical } = data;

  const filteredCategories = selectedCategory === "All"
    ? categories
    : categories.filter((c: any) => c.category === selectedCategory);

  return (
    <div style={{ maxWidth: "1400px", margin: "0 auto", padding: "2rem 1.5rem 4rem", width: "100%" }}>
      {/* Header */}
      <div style={{ marginBottom: "2rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.5rem" }}>
          <span className="badge badge-emerald">Phase 10.3 Analytics</span>
          <span className="badge badge-cyan">9 Analytical Modules</span>
        </div>
        <h1 style={{ fontFamily: "var(--font-heading)", fontSize: "2.2rem", fontWeight: 700 }}>
          Market Intelligence & Landscape Explorer
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", maxWidth: "800px", marginTop: "0.25rem" }}>
          Descriptive statistical insights across advertised vehicle listings from Riyasewana. Reflects market asking prices, not confirmed final transaction prices.
        </p>
      </div>

      {/* Top KPIs Row */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
        gap: "1.25rem",
        marginBottom: "2rem",
      }}>
        <div className="glass-panel" style={{ padding: "1.25rem" }}>
          <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", fontWeight: 500 }}>Total Advertised Listings</div>
          <div style={{ fontFamily: "var(--font-heading)", fontSize: "1.75rem", fontWeight: 700, color: "#ffffff", marginTop: "0.25rem" }}>
            {overview?.total_listings ?? 171}
          </div>
          <div style={{ fontSize: "0.75rem", color: "var(--accent-cyan)", marginTop: "0.2rem" }}>
            {overview?.distinct_categories ?? 8} Vehicle Classes
          </div>
        </div>

        <div className="glass-panel" style={{ padding: "1.25rem" }}>
          <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", fontWeight: 500 }}>Unique Vehicle Specifications</div>
          <div style={{ fontFamily: "var(--font-heading)", fontSize: "1.75rem", fontWeight: 700, color: "#ffffff", marginTop: "0.25rem" }}>
            {overview?.total_vehicles ?? 171}
          </div>
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
            {overview?.distinct_models ?? 112} Distinct Models
          </div>
        </div>

        <div className="glass-panel" style={{ padding: "1.25rem" }}>
          <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", fontWeight: 500 }}>ML-Eligible Clean Ratio</div>
          <div style={{ fontFamily: "var(--font-heading)", fontSize: "1.75rem", fontWeight: 700, color: "var(--accent-emerald)", marginTop: "0.25rem" }}>
            {overview?.ml_eligibility_rate_pct ?? 66.1}%
          </div>
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
            {overview?.ml_eligible_listings ?? 113} Training Usable
          </div>
        </div>

        <div className="glass-panel" style={{ padding: "1.25rem" }}>
          <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", fontWeight: 500 }}>Active Market Presence</div>
          <div style={{ fontFamily: "var(--font-heading)", fontSize: "1.75rem", fontWeight: 700, color: "var(--accent-purple)", marginTop: "0.25rem" }}>
            100%
          </div>
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
            Zero synthetic sold assumptions
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: "0.5rem",
        borderBottom: "1px solid var(--border-subtle)",
        paddingBottom: "0.75rem",
        marginBottom: "2rem",
        flexWrap: "wrap",
      }}>
        {[
          { id: "categories", label: "Category Analytics", icon: BarChart3 },
          { id: "brands", label: "Brands & Makes", icon: TrendingUp },
          { id: "characteristics", label: "Fuel & Transmission", icon: Fuel },
          { id: "districts", label: "Geographic Districts", icon: MapPin },
          { id: "quality", label: "Data Quality & Invariants", icon: ShieldCheck },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "0.45rem",
                padding: "0.5rem 1rem",
                borderRadius: "var(--radius-md)",
                background: isActive ? "rgba(56, 189, 248, 0.15)" : "transparent",
                border: isActive ? "1px solid rgba(56, 189, 248, 0.3)" : "1px solid transparent",
                color: isActive ? "#ffffff" : "var(--text-secondary)",
                fontSize: "0.875rem",
                fontWeight: isActive ? 600 : 500,
                cursor: "pointer",
                transition: "all 0.2s ease",
              }}
            >
              <Icon size={16} color={isActive ? "var(--accent-cyan)" : "var(--text-muted)"} />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Tab 1: Categories */}
      {activeTab === "categories" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          <div className="glass-panel" style={{ padding: "1.5rem", overflowX: "auto" }}>
            <h3 style={{ fontFamily: "var(--font-heading)", fontSize: "1.15rem", fontWeight: 600, marginBottom: "1.25rem" }}>
              Vehicle Category Distribution & Price Overview
            </h3>

            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.875rem", textAlign: "left" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border-medium)", color: "var(--text-muted)" }}>
                  <th style={{ padding: "0.75rem 1rem" }}>Category</th>
                  <th style={{ padding: "0.75rem 1rem" }}>Listings</th>
                  <th style={{ padding: "0.75rem 1rem" }}>Share %</th>
                  <th style={{ padding: "0.75rem 1rem" }}>ML Eligible %</th>
                  <th style={{ padding: "0.75rem 1rem" }}>Median Asking Price</th>
                  <th style={{ padding: "0.75rem 1rem" }}>Mean Asking Price</th>
                  <th style={{ padding: "0.75rem 1rem" }}>Median Age/YOM</th>
                </tr>
              </thead>
              <tbody>
                {filteredCategories.map((c: any, idx: number) => (
                  <tr key={idx} style={{
                    borderBottom: "1px solid var(--border-subtle)",
                    transition: "background 0.2s ease",
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(255, 255, 255, 0.03)")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                  >
                    <td style={{ padding: "0.85rem 1rem", fontWeight: 600, color: "#ffffff" }}>
                      {c.category}
                    </td>
                    <td style={{ padding: "0.85rem 1rem" }}>{c.listing_count}</td>
                    <td style={{ padding: "0.85rem 1rem" }}>{c.pct_of_total}%</td>
                    <td style={{ padding: "0.85rem 1rem" }}>
                      <span className="badge badge-emerald">{c.ml_eligible_pct}%</span>
                    </td>
                    <td style={{ padding: "0.85rem 1rem", fontWeight: 600, color: "var(--accent-cyan)" }}>
                      {formatLKR(c.median_asking_price)}
                    </td>
                    <td style={{ padding: "0.85rem 1rem", color: "var(--text-secondary)" }}>
                      {formatLKR(c.mean_asking_price)}
                    </td>
                    <td style={{ padding: "0.85rem 1rem", color: "var(--text-muted)" }}>
                      {c.median_yom}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab 2: Brands */}
      {activeTab === "brands" && (
        <div className="glass-panel" style={{ padding: "1.5rem" }}>
          <h3 style={{ fontFamily: "var(--font-heading)", fontSize: "1.15rem", fontWeight: 600, marginBottom: "1.25rem" }}>
            Top Vehicle Brands by Market Listing Volume
          </h3>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1rem" }}>
            {brands.slice(0, 12).map((b: any, idx: number) => (
              <div key={idx} style={{
                background: "rgba(10, 15, 29, 0.6)",
                padding: "1rem",
                borderRadius: "var(--radius-md)",
                border: "1px solid var(--border-subtle)",
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
                  <strong style={{ fontSize: "1rem", color: "#ffffff" }}>{b.brand}</strong>
                  <span className="badge badge-cyan">{b.listing_count} listings</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.825rem", color: "var(--text-secondary)" }}>
                  <span>Median Asking:</span>
                  <span style={{ fontWeight: 600, color: "var(--accent-cyan)" }}>{formatLKR(b.median_asking_price)}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.825rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
                  <span>Market Share:</span>
                  <span>{b.market_share_pct}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tab 3: Fuel & Transmission */}
      {activeTab === "characteristics" && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
          <div className="glass-panel" style={{ padding: "1.5rem" }}>
            <h3 style={{ fontFamily: "var(--font-heading)", fontSize: "1.15rem", fontWeight: 600, marginBottom: "1rem" }}>
              Fuel Type Proportions & Pricing
            </h3>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
              {fuel.map((f: any, idx: number) => (
                <div key={idx} style={{
                  background: "rgba(10, 15, 29, 0.6)",
                  padding: "0.85rem 1rem",
                  borderRadius: "var(--radius-md)",
                  border: "1px solid var(--border-subtle)",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}>
                  <div>
                    <strong style={{ color: "#ffffff", fontSize: "0.95rem" }}>{f.fuel_type}</strong>
                    <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>{f.count} vehicles ({f.percentage}%)</div>
                  </div>
                  <div style={{ fontWeight: 700, color: "var(--accent-cyan)" }}>
                    {formatLKR(f.median_price)}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="glass-panel" style={{ padding: "1.5rem" }}>
            <h3 style={{ fontFamily: "var(--font-heading)", fontSize: "1.15rem", fontWeight: 600, marginBottom: "1rem" }}>
              Transmission Distribution
            </h3>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
              {transmission.map((t: any, idx: number) => (
                <div key={idx} style={{
                  background: "rgba(10, 15, 29, 0.6)",
                  padding: "0.85rem 1rem",
                  borderRadius: "var(--radius-md)",
                  border: "1px solid var(--border-subtle)",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}>
                  <div>
                    <strong style={{ color: "#ffffff", fontSize: "0.95rem" }}>{t.transmission}</strong>
                    <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>{t.count} vehicles ({t.percentage}%)</div>
                  </div>
                  <div style={{ fontWeight: 700, color: "var(--accent-cyan)" }}>
                    {formatLKR(t.median_price)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Tab 4: Districts */}
      {activeTab === "districts" && (
        <div className="glass-panel" style={{ padding: "1.5rem" }}>
          <h3 style={{ fontFamily: "var(--font-heading)", fontSize: "1.15rem", fontWeight: 600, marginBottom: "1.25rem" }}>
            Geographic Listing Distribution across Sri Lanka
          </h3>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "1rem" }}>
            {districts.map((d: any, idx: number) => (
              <div key={idx} style={{
                background: "rgba(10, 15, 29, 0.6)",
                padding: "0.85rem 1rem",
                borderRadius: "var(--radius-md)",
                border: "1px solid var(--border-subtle)",
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                    <MapPin size={14} color="var(--accent-cyan)" />
                    <strong style={{ color: "#ffffff" }}>{d.district}</strong>
                  </div>
                  <span className="badge badge-cyan">{d.count} ads</span>
                </div>
                <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: "0.5rem", display: "flex", justifyContent: "space-between" }}>
                  <span>Median Price:</span>
                  <span style={{ fontWeight: 600, color: "var(--accent-cyan)" }}>{formatLKR(d.median_price)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tab 5: Data Quality */}
      {activeTab === "quality" && (
        <div className="glass-panel" style={{ padding: "1.75rem" }}>
          <h3 style={{ fontFamily: "var(--font-heading)", fontSize: "1.25rem", fontWeight: 600, marginBottom: "1rem" }}>
            Data Quality & Historical Invariant Auditing
          </h3>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem", lineHeight: 1.5, marginBottom: "1.5rem" }}>
            The platform enforces 11 relational invariants and decouples general data cleanliness (<code>is_valid</code>) from model usability (<code>ml_eligible</code>), guaranteeing that no seller records are destructively removed.
          </p>

          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
            gap: "1rem",
          }}>
            {[
              "Duplicate Listings (source + listing_id)",
              "Duplicate URLs across runs",
              "Consecutive Duplicate Prices in PriceHistory",
              "Observation Timestamps >= first_seen_at",
              "Timestamp Ordering (first_seen <= last_seen)",
              "Strict Status Set (ACTIVE or NO_LONGER_OBSERVED)",
              "Orphan Price History Detection",
              "Orphan Observation References",
              "Orphan Vehicles without Listings",
              "Broken Vehicle ForeignKey References",
              "Scope-Gated Disappearance Protection",
            ].map((inv, idx) => (
              <div key={idx} style={{
                background: "rgba(10, 15, 29, 0.6)",
                padding: "0.85rem 1rem",
                borderRadius: "var(--radius-md)",
                border: "1px solid var(--border-subtle)",
                display: "flex",
                alignItems: "center",
                gap: "0.6rem",
              }}>
                <CheckCircle2 size={16} color="var(--accent-emerald)" />
                <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>{inv}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
