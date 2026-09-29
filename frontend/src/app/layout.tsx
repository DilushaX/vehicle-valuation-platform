import type { Metadata } from "next";
import "./globals.css";
import Header from "@/components/Header";
import Footer from "@/components/Footer";

export const metadata: Metadata = {
  title: "AutoValuate LK — Sri Lankan Vehicle Intelligence & AI Valuation Platform",
  description:
    "Data-driven market intelligence, explainable ML asking price valuation with Tree SHAP, and multi-factor comparable matching for Sri Lankan used vehicles.",
  keywords: [
    "Sri Lanka vehicle valuation",
    "Riyasewana car prices",
    "Used vehicle prices Sri Lanka",
    "Machine learning car valuation",
    "Tree SHAP explainable AI",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <meta name="color-scheme" content="dark" />
      </head>
      <body>
        <Header />
        <main style={{ minHeight: "calc(100vh - 180px)", display: "flex", flexDirection: "column" }}>
          {children}
        </main>
        <Footer />
      </body>
    </html>
  );
}
