/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow images from localhost (FastAPI)
  images: {
    remotePatterns: [],
  },
  // Env variables available in browser
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
};

module.exports = nextConfig;
