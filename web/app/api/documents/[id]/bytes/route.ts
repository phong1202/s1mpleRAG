// The PDF viewer's bytes, relayed through this server instead of fetched
// by the browser straight from the presigned URL.
//
// Why: download managers (IDM's browser extension, common here) watch every
// response the browser receives, background fetches included, and snatch
// anything that looks like a PDF -- the application/pdf type, a filename in
// Content-Disposition, or a .pdf path. The viewer then never gets its bytes.
// This response carries none of the three, and the GET URL for reading a
// document never reaches the browser. (Uploads still PUT from the browser
// to a presigned URL; that response is empty, so nothing there to snatch.)

const API_URL = process.env.API_URL ?? 'http://localhost:8000'
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

// Deliberately not application/pdf, and not application/octet-stream
// either, which download managers map to a generic binary download. Only
// the viewer's fetch reads this, so any unambiguous private type will do.
const CONTENT_TYPE = 'application/x.s1mplerag-document'

function envelope(status: number, message: string) {
  return Response.json({ code: status, message, data: null }, { status })
}

export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  // Interpolated into the API's URL below: anything but a UUID is refused
  // rather than escaped, so no path can be smuggled through.
  if (!UUID.test(id)) return envelope(404, 'Document not found')

  let api: Response
  try {
    api = await fetch(`${API_URL}/documents/${id}/file-url`, { cache: 'no-store' })
  } catch {
    return envelope(503, 'Backend unreachable')
  }
  if (!api.ok) {
    // The API's own envelope (404 and the like), passed through as is.
    return new Response(api.body, { status: api.status, headers: { 'Content-Type': 'application/json' } })
  }
  const { data } = (await api.json()) as { data: { url: string } }

  let file: Response
  try {
    file = await fetch(data.url, { cache: 'no-store' })
  } catch {
    return envelope(502, 'Storage unreachable')
  }
  if (!file.ok || !file.body) return envelope(502, `Storage answered HTTP ${file.status}`)

  const headers = new Headers({
    'Content-Type': CONTENT_TYPE,
    'Cache-Control': 'private, no-store',
    'X-Content-Type-Options': 'nosniff',
  })
  const length = file.headers.get('content-length')
  if (length) headers.set('Content-Length', length)
  // Streamed, not buffered: a 50 MB scan never sits whole in this server.
  return new Response(file.body, { headers })
}
