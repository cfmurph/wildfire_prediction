import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static files only: the API service serves them from its own origin (see api/main.py)
  output: "export",
  env: {
    // Same origin in production; the local API during `next dev`
    NEXT_PUBLIC_API_URL:
      process.env.NEXT_PUBLIC_API_URL ?? (process.env.NODE_ENV === "production" ? "" : "http://localhost:8000"),
  },
};

export default nextConfig;
