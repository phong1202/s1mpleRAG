/** @type {import('next').NextConfig} */
const nextConfig = {
  // Dev assets are only served to `localhost` by default. A browser on the
  // Windows host may reach the WSL dev server as 127.0.0.1 instead.
  allowedDevOrigins: ['127.0.0.1'],
  // The browser talks to the FastAPI backend through this same origin, so
  // the API needs no CORS grant for it and its address stays server-side.
  async rewrites() {
    const api = process.env.API_URL ?? 'http://localhost:8000'
    return [{ source: '/backend/:path*', destination: `${api}/:path*` }]
  },
  experimental: {
    // How long the rewrite above waits for the backend, 30 s by default.
    // /query is an embedding plus a reasoning-model completion in a single
    // response, and a provider backing off a rate limit can take longer
    // than that; cut off, it would read as the backend being down.
    proxyTimeout: 120_000,
  },
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          { key: 'X-Content-Type-Options', value: 'nosniff' },
          { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
          { key: 'Strict-Transport-Security', value: 'max-age=63072000' },
          { key: 'Permissions-Policy', value: 'camera=(), microphone=(), geolocation=()' },
        ],
      },
    ]
  },
}

export default nextConfig
