import type { NextConfig } from "next";
import { securityHeaders } from "./lib/securityHeaders.ts";

const nextConfig: NextConfig = {
  agentRules: false,
  devIndicators: false,
  poweredByHeader: false,
  // Next otherwise aliases 127.0.0.1/[::1] to localhost in Request.url.
  skipProxyUrlNormalize: true,
  async headers() {
    return [
      { source: "/:path*", headers: securityHeaders(process.env.NODE_ENV === "production") },
      { source: "/api/:path*", headers: [{ key: "Cache-Control", value: "no-store" }] },
      // Next applies configured cache headers to dynamic handlers as well.
      { source: "/api/chat/stream", headers: [{ key: "Cache-Control", value: "no-store, no-transform" }] },
    ];
  },
};

export default nextConfig;
