# 🚗 AutoValuate LK — Next.js Dashboard Frontend

A modern, responsive, dark-mode web application for the **Sri Lankan Vehicle Market Intelligence & Explainable AI Valuation Platform**.

Built with **Next.js 16 (App Router)**, **React 19**, **TypeScript**, and a custom **Vanilla CSS** design system.

---

## 🌟 Key Capabilities

1. 🏠 **Overview (`/`)**:
   - System introduction, metrics cards (171 listings, 8 categories, 33 brands, 112 models).
   - Live backend monitor (FastAPI liveness & RandomForest model readiness).
   - 6-phase end-to-end architecture pipeline interactive overview.

2. 🔍 **Vehicle Valuation (`/valuation`)**:
   - Interactive specification form with quick presets (Toyota Premio, Honda Vezel, Toyota Prado, Suzuki Alto).
   - **Hero Estimated Asking Price**: Formatted in Sri Lankan Rupees (Rs. / LKR).
   - **Indicative 300-Tree Prediction Range**: Empirical 10th to 90th percentile decision-tree dispersion with lower bound, point estimate, upper bound, and spread.
   - **Tree SHAP Factor Attribution**: Bidirectional horizontal contribution bars showing positive (emerald) and negative (rose) impacts in LKR with human-readable rationale.
   - **Top Matching Comparable Listings**: Peer listings from verified dataset with similarity percentages (e.g. 96% Match, 88% Match).

3. 📈 **Market Intelligence (`/market-intelligence`)**:
   - 9 descriptive analytical modules covering public listings from Riyasewana.
   - Category distribution, Top brands by volume and price spreads, Fuel type breakdown, Transmission comparison, and Geographic district mapping across Sri Lanka.
   - 11 dataset consistency invariants audit and strict 60-day observation depth warning.

4. 🔎 **Comparable Vehicles (`/comparables`)**:
   - Dedicated peer vehicle retrieval engine based on multi-factor weighted specification distance.
   - Customizable criteria weights (Brand, Model, Year, Mileage, Engine CC, Transmission, etc.).
   - Cohort asking price distribution summary (Min, Median, Mean, Max).

---

## 🛠️ Technology Stack & Design System

- **Framework**: Next.js 16 (App Router) + React 19 + TypeScript
- **Styling**: Vanilla CSS with custom CSS variables, glassmorphism (`backdrop-filter: blur(16px)`), and radial background gradient meshes.
- **Icons**: Lucide React
- **API Proxy**: Rewrites `/api/py/:path*` directly to FastAPI backend (`http://127.0.0.1:8000/:path*`).

---

## 🚀 Running Locally

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server on port 3000
npm run dev -- -p 3000

# Or build production bundle
npm run build
npm start
```

- Web UI: `http://localhost:3000`
- Requires FastAPI backend running on `http://localhost:8000`.
