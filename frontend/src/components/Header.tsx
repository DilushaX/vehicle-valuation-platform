"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Car, TrendingUp, Search, Activity, CheckCircle2, AlertTriangle, Shield } from "lucide-react";

export default function Header() {
  const pathname = usePathname();
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");
  const [modelReady, setModelReady] = useState<boolean>(false);

  useEffect(() => {
    let mounted = true;
    async function checkHealth() {
      try {
        const [healthRes, readyRes] = await Promise.allSettled([
          fetch("/api/py/health", { cache: "no-store" }),
          fetch("/api/py/ready", { cache: "no-store" }),
        ]);

        if (!mounted) return;

        if (healthRes.status === "fulfilled" && healthRes.value.ok) {
          setApiStatus("online");
        } else {
          setApiStatus("offline");
        }

        if (readyRes.status === "fulfilled" && readyRes.value.ok) {
          setModelReady(true);
        } else {
          setModelReady(false);
        }
      } catch {
        if (mounted) {
          setApiStatus("offline");
          setModelReady(false);
        }
      }
    }

    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const navLinks = [
    { href: "/", label: "Overview", icon: Activity },
    { href: "/valuation", label: "Vehicle Valuation", icon: Car },
    { href: "/market-intelligence", label: "Market Intelligence", icon: TrendingUp },
    { href: "/comparables", label: "Comparable Vehicles", icon: Search },
  ];

  return (
    <header style={{
      position: "sticky",
      top: 0,
      zIndex: 50,
      background: "rgba(7, 11, 20, 0.8)",
      backdropFilter: "blur(20px)",
      WebkitBackdropFilter: "blur(20px)",
      borderBottom: "1px solid var(--border-subtle)",
    }}>
      <div style={{
        maxWidth: "1400px",
        margin: "0 auto",
        padding: "0.85rem 1.5rem",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        flexWrap: "wrap",
        gap: "1rem",
      }}>
        {/* Brand */}
        <Link href="/" style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <div style={{
            width: "38px",
            height: "38px",
            borderRadius: "10px",
            background: "linear-gradient(135deg, #0284c7 0%, #7c3aed 100%)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 0 16px rgba(56, 189, 248, 0.35)",
          }}>
            <Car size={22} color="#ffffff" />
          </div>
          <div>
            <div style={{
              fontFamily: "var(--font-heading)",
              fontSize: "1.15rem",
              fontWeight: 700,
              letterSpacing: "-0.02em",
              background: "linear-gradient(to right, #ffffff, #94a3b8)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}>
              AutoValuate <span style={{ color: "var(--accent-cyan)", WebkitTextFillColor: "var(--accent-cyan)", fontSize: "0.8rem", fontWeight: 600 }}>LK</span>
            </div>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", letterSpacing: "0.02em" }}>
              Sri Lankan Vehicle Intelligence & AI Valuation
            </div>
          </div>
        </Link>

        {/* Navigation Tabs */}
        <nav style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
          {navLinks.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.45rem",
                  padding: "0.45rem 0.85rem",
                  borderRadius: "var(--radius-md)",
                  fontSize: "0.875rem",
                  fontWeight: isActive ? 600 : 500,
                  color: isActive ? "#ffffff" : "var(--text-secondary)",
                  background: isActive ? "rgba(56, 189, 248, 0.12)" : "transparent",
                  border: isActive ? "1px solid rgba(56, 189, 248, 0.3)" : "1px solid transparent",
                  transition: "all 0.2s ease",
                }}
              >
                <Icon size={16} color={isActive ? "var(--accent-cyan)" : "var(--text-muted)"} />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Live Service Monitor */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
          <div style={{
            display: "flex",
            alignItems: "center",
            gap: "0.45rem",
            padding: "0.3rem 0.75rem",
            background: "rgba(15, 23, 42, 0.8)",
            borderRadius: "var(--radius-full)",
            border: "1px solid var(--border-subtle)",
            fontSize: "0.75rem",
          }}>
            <div
              className={apiStatus === "online" ? "status-dot-pulse" : ""}
              style={{
                width: "8px",
                height: "8px",
                borderRadius: "50%",
                background: apiStatus === "online" ? "var(--accent-emerald)" : apiStatus === "checking" ? "var(--accent-amber)" : "var(--accent-rose)",
                boxShadow: apiStatus === "online" ? "0 0 8px var(--accent-emerald)" : "none",
              }}
            />
            <span style={{ color: "var(--text-secondary)" }}>API</span>
            <span style={{
              fontWeight: 600,
              color: apiStatus === "online" ? "var(--accent-emerald)" : "var(--text-muted)",
            }}>
              {apiStatus === "online" ? "Online" : apiStatus === "checking" ? "Checking" : "Offline"}
            </span>
          </div>

          <div style={{
            display: "flex",
            alignItems: "center",
            gap: "0.45rem",
            padding: "0.3rem 0.75rem",
            background: "rgba(15, 23, 42, 0.8)",
            borderRadius: "var(--radius-full)",
            border: "1px solid var(--border-subtle)",
            fontSize: "0.75rem",
          }}>
            <Shield size={12} color={modelReady ? "var(--accent-purple)" : "var(--text-muted)"} />
            <span style={{ color: "var(--text-secondary)" }}>Model</span>
            <span style={{
              fontWeight: 600,
              color: modelReady ? "var(--accent-purple)" : "var(--text-muted)",
            }}>
              {modelReady ? "RandomForest Ready" : "Unloaded"}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
