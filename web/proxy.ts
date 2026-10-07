import { NextResponse, type NextRequest } from 'next/server'

// A strict CSP matters here because the viewer runs pdf.js in this page on
// PDFs anyone can upload. A nonce + 'strict-dynamic' lets only Next's own
// scripts (and what they load) run; 'wasm-unsafe-eval' allows compiling
// pdf.js's image decoders without allowing JS eval. No
// upgrade-insecure-requests: storage is plain http in local setups.
// style-src stays 'unsafe-inline': the pdf.js text layer and the resizable
// panels position everything with inline styles, and injected CSS cannot
// run code the way an injected script can.
export function proxy(request: NextRequest) {
  const nonce = Buffer.from(crypto.randomUUID()).toString('base64')
  const isDev = process.env.NODE_ENV === 'development'

  const csp = `
    default-src 'self';
    script-src 'self' 'nonce-${nonce}' 'strict-dynamic' 'wasm-unsafe-eval'${isDev ? " 'unsafe-eval'" : ''};
    style-src 'self' 'unsafe-inline';
    img-src 'self' blob: data:;
    font-src 'self' data:;
    connect-src 'self';
    worker-src 'self' blob:;
    frame-src 'none';
    object-src 'none';
    base-uri 'self';
    form-action 'self';
    frame-ancestors 'none';
  `
    .replace(/\s{2,}/g, ' ')
    .trim()

  const requestHeaders = new Headers(request.headers)
  requestHeaders.set('x-nonce', nonce)
  requestHeaders.set('Content-Security-Policy', csp)

  const response = NextResponse.next({ request: { headers: requestHeaders } })
  response.headers.set('Content-Security-Policy', csp)
  return response
}

export const config = {
  matcher: [
    {
      // Pages only: not static assets, pdf.js data files, or API routes.
      source: '/((?!api|backend|pdfjs|_next/static|_next/image|favicon.ico|icon.svg).*)',
      missing: [
        { type: 'header', key: 'next-router-prefetch' },
        { type: 'header', key: 'purpose', value: 'prefetch' },
      ],
    },
  ],
}
