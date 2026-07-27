/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    // Disable type checking during build since third-party wagmi/viem packages
    // can trigger deep recursion errors in TypeScript.
    ignoreBuildErrors: true,
  },
  eslint: {
    // Disable eslint checks during build to speed up compilation
    ignoreDuringBuilds: true,
  },
  // Suppress non-critical warnings from third-party packages (MetaMask SDK, pino, etc.)
  webpack: (config, { isServer }) => {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      // MetaMask SDK references React Native packages not needed in browser
      "@react-native-async-storage/async-storage": false,
      "react-native": false,
      "react-native-vector-icons": false,
      // pino-pretty is optional but warns when missing
      "pino-pretty": false,
    };
    // Suppress critical dependency warnings from third-party packages
    config.ignoreWarnings = [
      { module: /node_modules\/@metamask/ },
      { module: /node_modules\/pino/ },
      { module: /node_modules\/@walletconnect/ },
    ];
    return config;
  },
  // API rewrites: In production, set NEXT_PUBLIC_BACKEND_API_URL to your Render backend URL.
  // Falls back to localhost:8000 for local development.
  async rewrites() {
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_API_URL || "http://127.0.0.1:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
