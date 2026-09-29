import { AlertCircle, ShieldAlert } from "lucide-react";

export default function Footer() {
  return (
    <footer style={{
      marginTop: "auto",
      borderTop: "1px solid var(--border-subtle)",
      background: "rgba(7, 11, 20, 0.9)",
      padding: "2.5rem 1.5rem 2rem",
      fontSize: "0.8rem",
      color: "var(--text-muted)",
    }}>
      <div style={{ maxWidth: "1400px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "1.5rem" }}>
        {/* Methodological Notice */}
        <div style={{
          display: "flex",
          alignItems: "flex-start",
          gap: "0.75rem",
          padding: "1rem 1.25rem",
          borderRadius: "var(--radius-md)",
          background: "rgba(245, 158, 11, 0.05)",
          border: "1px solid rgba(245, 158, 11, 0.2)",
        }}>
          <ShieldAlert size={18} color="var(--accent-amber)" style={{ flexShrink: 0, marginTop: "2px" }} />
          <div style={{ lineHeight: 1.5 }}>
            <strong style={{ color: "var(--accent-amber)" }}>Essential Market & Methodological Disclaimers:</strong>
            <p style={{ marginTop: "0.25rem", color: "var(--text-secondary)" }}>
              1. <strong>Advertised Asking Prices Only</strong>: Predictions and analytics estimate observed seller advertised asking prices on Riyasewana. They do <em>not</em> represent negotiated, settlement, or verified sale contract prices.
            </p>
            <p style={{ marginTop: "0.25rem", color: "var(--text-secondary)" }}>
              2. <strong>Unobserved Physical Factors</strong>: Actual physical vehicle condition, mechanical health, accident history, battery degradation, and title paperwork cannot be captured from online listings.
            </p>
            <p style={{ marginTop: "0.25rem", color: "var(--text-secondary)" }}>
              3. <strong>Experimental Research Benchmark</strong>: The underlying model pipeline is an experimental research benchmark. It does not constitute a legally or financially certified appraisal.
            </p>
          </div>
        </div>

        <div style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "1rem",
          paddingTop: "1rem",
          borderTop: "1px solid rgba(255, 255, 255, 0.05)",
        }}>
          <div>
            © {new Date().getFullYear()} AutoValuate LK — Sri Lankan Vehicle Market Intelligence & Valuation Platform.
          </div>
          <div style={{ display: "flex", gap: "1.25rem" }}>
            <span>Powered by Scikit-Learn RandomForest + Tree SHAP</span>
            <span>•</span>
            <span>FastAPI & Next.js</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
