import Link from "next/link";
import { Car, TrendingUp, Search, ShieldCheck, Database, Cpu, ArrowRight, CheckCircle2, AlertCircle, Sparkles, Layers, Sliders } from "lucide-react";

export default function OverviewPage() {
  const metrics = [
    { label: "Tracked Market Listings", value: "171", change: "+14 this week", icon: Database, color: "var(--accent-cyan)" },
    { label: "ML-Eligible Training Records", value: "113", change: "66.1% Clean Ratio", icon: ShieldCheck, color: "var(--accent-emerald)" },
    { label: "Canonical Categories", value: "8", change: "Cars, SUVs, Vans, Bikes...", icon: Layers, color: "var(--accent-purple)" },
    { label: "Distinct Brands & Models", value: "33 / 112", change: "Toyota, Honda, Nissan...", icon: Car, color: "var(--accent-amber)" },
  ];

  const features = [
    {
      title: "Explainable Machine Learning Valuation",
      desc: "Category-specific Random Forest regression trained on log1p asking prices, featuring exact additive Tree SHAP factor attributions calculated directly in Sri Lankan Rupees.",
      icon: Cpu,
      color: "var(--accent-cyan)",
      link: "/valuation",
      badge: "Tree SHAP Explainability",
    },
    {
      title: "Indicative 300-Tree Prediction Range",
      desc: "Instead of opaque single-number outputs, the platform derives an empirical dispersion interval across all 300 decision trees in the ensemble (10th to 90th percentiles).",
      icon: Sliders,
      color: "var(--accent-purple)",
      link: "/valuation",
      badge: "Ensemble Dispersion",
    },
    {
      title: "Descriptive Market Intelligence",
      desc: "Nine comprehensive analytical modules exploring price distributions, non-causal bivariate correlations, vehicle characteristics, district differences, and depth-gated trends.",
      icon: TrendingUp,
      color: "var(--accent-emerald)",
      link: "/market-intelligence",
      badge: "9 Analytical Views",
    },
    {
      title: "Multi-Attribute Comparable Matching",
      desc: "Weighted specification distance scoring across Make, Model, Age, Mileage, Engine CC, Fuel, Transmission, District, and Condition to locate true peer vehicles.",
      icon: Search,
      color: "var(--accent-amber)",
      link: "/comparables",
      badge: "Similarity Scoring",
    },
  ];

  const architectureSteps = [
    { step: "01", name: "Data Acquisition", desc: "Sequential polite crawler (HTTPX / BS4) with 1.5s delay and macOS launchd / Cron scheduling" },
    { step: "02", name: "Quality & Invariants", desc: "11 dataset consistency invariants audit, IQR outlier detection, and decoupled ml_eligible logic" },
    { step: "03", name: "Relational Persistence", desc: "PostgreSQL schema tracking vehicles, listings, price changes, and point-in-time observations" },
    { step: "04", name: "ML & Explainability", desc: "TransformedTargetRegressor, Random Forest 300 estimators, and Tree SHAP feature attribution" },
    { step: "05", name: "Workflow Layer", desc: "VehicleValuationWorkflow coordinating Point Estimate, Range, SHAP, Comps, and SHA-256 fingerprint" },
    { step: "06", name: "Delivery Interfaces", desc: "Production FastAPI backend (localhost:8000) and Next.js modern responsive dashboard" },
  ];

  return (
    <div style={{ maxWidth: "1400px", margin: "0 auto", padding: "2.5rem 1.5rem 4rem", width: "100%" }}>
      {/* Hero Section */}
      <section style={{ textAlign: "center", marginBottom: "3.5rem" }}>
        <div style={{ display: "inline-flex", alignItems: "center", gap: "0.5rem", marginBottom: "1.25rem" }}>
          <span className="badge badge-cyan">
            <Sparkles size={12} /> Next.js 16 • React 19 • App Router
          </span>
          <span className="badge badge-purple">
            FastAPI Live Connected
          </span>
        </div>
        
        <h1 style={{
          fontFamily: "var(--font-heading)",
          fontSize: "clamp(2.2rem, 5vw, 3.8rem)",
          fontWeight: 800,
          lineHeight: 1.15,
          letterSpacing: "-0.03em",
          maxWidth: "1000px",
          margin: "0 auto 1.25rem",
          background: "linear-gradient(135deg, #ffffff 30%, #94a3b8 100%)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
        }}>
          Sri Lankan Vehicle Market Intelligence & Explainable AI Valuation
        </h1>

        <p style={{
          fontSize: "1.125rem",
          color: "var(--text-secondary)",
          maxWidth: "760px",
          margin: "0 auto 2.25rem",
          lineHeight: 1.6,
        }}>
          An end-to-end platform collecting public listings from Riyasewana, auditing relational data quality, 
          detecting market anomalies, and estimating advertised asking values with model-based uncertainty and Tree SHAP factor explanations.
        </p>

        <div style={{ display: "flex", justifyContent: "center", gap: "1rem", flexWrap: "wrap" }}>
          <Link href="/valuation" className="btn btn-primary" style={{ padding: "0.8rem 1.75rem", fontSize: "1rem" }}>
            <Car size={18} />
            <span>Launch Vehicle Valuation</span>
            <ArrowRight size={16} />
          </Link>
          <Link href="/market-intelligence" className="btn btn-secondary" style={{ padding: "0.8rem 1.5rem", fontSize: "1rem" }}>
            <TrendingUp size={18} />
            <span>Market Intelligence</span>
          </Link>
          <Link href="/comparables" className="btn btn-secondary" style={{ padding: "0.8rem 1.5rem", fontSize: "1rem" }}>
            <Search size={18} />
            <span>Comparable Search</span>
          </Link>
        </div>
      </section>

      {/* Metrics Row */}
      <section style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
        gap: "1.25rem",
        marginBottom: "3.5rem",
      }}>
        {metrics.map((m, idx) => {
          const Icon = m.icon;
          return (
            <div key={idx} className="glass-panel glass-panel-hoverable" style={{ padding: "1.5rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
                <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)", fontWeight: 500 }}>{m.label}</span>
                <div style={{
                  width: "36px",
                  height: "36px",
                  borderRadius: "10px",
                  background: `rgba(255, 255, 255, 0.05)`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  border: "1px solid var(--border-subtle)",
                }}>
                  <Icon size={18} color={m.color} />
                </div>
              </div>
              <div style={{
                fontFamily: "var(--font-heading)",
                fontSize: "2rem",
                fontWeight: 700,
                color: "#ffffff",
                lineHeight: 1.1,
                marginBottom: "0.4rem",
              }}>
                {m.value}
              </div>
              <div style={{ fontSize: "0.78rem", color: m.color, fontWeight: 500 }}>
                {m.change}
              </div>
            </div>
          );
        })}
      </section>

      {/* Feature Highlights Grid */}
      <section style={{ marginBottom: "3.5rem" }}>
        <div style={{ marginBottom: "1.75rem", display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: "1rem" }}>
          <div>
            <div style={{ fontSize: "0.85rem", color: "var(--accent-cyan)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Core Capabilities
            </div>
            <h2 style={{ fontFamily: "var(--font-heading)", fontSize: "1.85rem", fontWeight: 700, marginTop: "0.25rem" }}>
              Engineered for Rigor, Transparency & Auditability
            </h2>
          </div>
          <span style={{ fontSize: "0.875rem", color: "var(--text-muted)" }}>
            Four cohesive modules built on modern web standards
          </span>
        </div>

        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
          gap: "1.5rem",
        }}>
          {features.map((f, idx) => {
            const Icon = f.icon;
            return (
              <div key={idx} className="glass-panel glass-panel-hoverable" style={{ padding: "1.75rem", display: "flex", flexDirection: "column" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1rem" }}>
                  <div style={{
                    width: "44px",
                    height: "44px",
                    borderRadius: "12px",
                    background: `rgba(255, 255, 255, 0.05)`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    border: "1px solid var(--border-subtle)",
                  }}>
                    <Icon size={22} color={f.color} />
                  </div>
                  <span className="badge" style={{ background: "rgba(255, 255, 255, 0.06)", color: "var(--text-secondary)", border: "1px solid var(--border-subtle)" }}>
                    {f.badge}
                  </span>
                </div>

                <h3 style={{ fontFamily: "var(--font-heading)", fontSize: "1.25rem", fontWeight: 600, marginBottom: "0.6rem" }}>
                  {f.title}
                </h3>
                
                <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.55, marginBottom: "1.5rem", flexGrow: 1 }}>
                  {f.desc}
                </p>

                <Link href={f.link} style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "0.4rem",
                  fontSize: "0.875rem",
                  fontWeight: 600,
                  color: f.color,
                  marginTop: "auto",
                }}>
                  <span>Open Module</span>
                  <ArrowRight size={14} />
                </Link>
              </div>
            );
          })}
        </div>
      </section>

      {/* Architecture Pipeline Flow */}
      <section className="glass-panel" style={{ padding: "2.25rem", marginBottom: "2rem" }}>
        <div style={{ textAlign: "center", maxWidth: "700px", margin: "0 auto 2.25rem" }}>
          <div style={{ fontSize: "0.8rem", color: "var(--accent-purple)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>
            End-To-End Architecture
          </div>
          <h2 style={{ fontFamily: "var(--font-heading)", fontSize: "1.75rem", fontWeight: 700, marginTop: "0.3rem" }}>
            The Complete Data-To-Valuation Pipeline
          </h2>
          <p style={{ fontSize: "0.875rem", color: "var(--text-muted)", marginTop: "0.4rem" }}>
            Strict separation of concerns, zero data leakage, and deterministic audit trails across every layer.
          </p>
        </div>

        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: "1.25rem",
        }}>
          {architectureSteps.map((st, idx) => (
            <div key={idx} style={{
              background: "rgba(10, 15, 29, 0.6)",
              borderRadius: "var(--radius-md)",
              border: "1px solid var(--border-subtle)",
              padding: "1.25rem",
              position: "relative",
              overflow: "hidden",
            }}>
              <div style={{
                position: "absolute",
                top: "0.75rem",
                right: "0.75rem",
                fontFamily: "var(--font-heading)",
                fontSize: "1.75rem",
                fontWeight: 800,
                color: "rgba(255, 255, 255, 0.05)",
                lineHeight: 1,
              }}>
                {st.step}
              </div>
              <div style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--accent-cyan)", marginBottom: "0.3rem" }}>
                PHASE {st.step}
              </div>
              <div style={{ fontFamily: "var(--font-heading)", fontSize: "1.05rem", fontWeight: 600, color: "#ffffff", marginBottom: "0.4rem" }}>
                {st.name}
              </div>
              <div style={{ fontSize: "0.825rem", color: "var(--text-secondary)", lineHeight: 1.45 }}>
                {st.desc}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
