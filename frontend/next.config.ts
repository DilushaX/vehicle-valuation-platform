import type { NextConfig } from "next";

const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL || "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/py/:path*",
        destination: `${FASTAPI_BASE_URL}/:path*`,
      },
    ];
  },
};

export default nextConfig;
