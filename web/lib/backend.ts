import type { DocumentSummary } from '@/lib/types'

// Every backend call goes through the `/backend/*` rewrite in next.config.mjs,
// so the browser never needs the API's address or a CORS grant from it. The
// one exception is the file PUT, which goes straight to the presigned MinIO
// URL: the signature is bound to that host.
const BASE = '/backend'

// Mirrors MAX_FILE_SIZE_MB on the backend (its default). The backend is the
// one that enforces it; this only saves hashing and uploading a file that
// would be refused at register time anyway.
export const MAX_FILE_MB = 50
export const ACCEPT = '.pdf'

export type ErrorKey =
  | 'unsupported_type'
  | 'file_too_large'
  | 'already_ingested'
  | 'backend_unavailable'
  | 'storage_rejected'
  | 'storage_unavailable'
  | 'document_not_found'

// `key` picks a translated message when there is one; `message` is the
// backend's own text, shown when there is not.
export class BackendError extends Error {
  constructor(
    readonly key: ErrorKey | null,
    message: string,
  ) {
    super(message)
  }
}

type Envelope<T> = { code: number; message: string; data: T | null }

type Paginated<T> = { items: T[]; total: number; limit: number; offset: number }

type UploadTarget = { upload_url: string; object_key: string; expires_in: number }

const KEY_BY_STATUS: Record<number, ErrorKey> = {
  404: 'document_not_found',
  409: 'already_ingested',
  413: 'file_too_large',
}

// The bytes route adds its own two: the API (503) or storage (502) down.
const BYTES_KEY_BY_STATUS: Record<number, ErrorKey> = {
  ...KEY_BY_STATUS,
  502: 'storage_unavailable',
  503: 'backend_unavailable',
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, init)
  } catch (error) {
    // A caller that cancelled wants the abort back, not an outage.
    if (init?.signal?.aborted) throw error
    throw new BackendError('backend_unavailable', 'Backend unreachable')
  }

  // A backend that is down answers through the rewrite with a plain-text
  // 500, not the envelope -- that is the only case the parse fails.
  const body = (await res.json().catch(() => null)) as Envelope<T> | null
  if (!body) throw new BackendError('backend_unavailable', `HTTP ${res.status}`)
  if (!res.ok) throw new BackendError(KEY_BY_STATUS[res.status] ?? null, body.message)
  return body.data as T
}

function json(method: string, payload: unknown): RequestInit {
  return {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }
}

export const DOCUMENTS_KEY = '/documents?limit=100'

export async function fetchDocuments(path: string): Promise<DocumentSummary[]> {
  const page = await call<Paginated<DocumentSummary>>(path)
  return page.items
}

export function deleteDocument(id: string) {
  return call<null>(`/documents/${encodeURIComponent(id)}`, { method: 'DELETE' })
}

// The PDF's bytes, for the in-page viewer. Relayed by this app's own
// server (app/api/documents/[id]/bytes) rather than fetched from the
// presigned URL here: see that route for why.
export async function downloadDocument(id: string, signal?: AbortSignal): Promise<Uint8Array> {
  let res: Response
  try {
    res = await fetch(`/api/documents/${encodeURIComponent(id)}/bytes`, { signal })
  } catch (error) {
    if (signal?.aborted) throw error
    throw new BackendError('backend_unavailable', 'Backend unreachable')
  }
  if (!res.ok) {
    const body = (await res.json().catch(() => null)) as Envelope<null> | null
    throw new BackendError(BYTES_KEY_BY_STATUS[res.status] ?? null, body?.message ?? `HTTP ${res.status}`)
  }
  return new Uint8Array(await res.arrayBuffer())
}

async function sha256(file: File) {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', await file.arrayBuffer()))
  const hex = Array.from(digest, (b) => b.toString(16).padStart(2, '0')).join('')
  const base64 = btoa(String.fromCharCode(...digest))
  return { hex, base64 }
}

// XHR rather than fetch: fetch still has no upload progress events, and a
// 50 MB scan is long enough on a slow link that a bar is worth having.
function putObject(url: string, file: File, checksum: string, onProgress: (ratio: number) => void) {
  return new Promise<void>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('PUT', url)
    // Bound into the presigned signature: MinIO refuses the PUT without it,
    // and refuses bytes that do not hash to it.
    xhr.setRequestHeader('x-amz-checksum-sha256', checksum)
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total)
    }
    xhr.onload = () =>
      xhr.status >= 200 && xhr.status < 300
        ? resolve()
        : reject(new BackendError('storage_rejected', `Storage answered HTTP ${xhr.status}`))
    xhr.onerror = () => reject(new BackendError('storage_unavailable', 'Storage unreachable'))
    xhr.send(file)
  })
}

export type UploadPhase =
  | { phase: 'hashing' }
  | { phase: 'uploading'; progress: number }
  | { phase: 'registering' }

// The backend's three-step ingest: ask for a URL keyed by the file's hash,
// PUT the bytes there, then register the object so the worker picks it up.
export async function uploadDocument(file: File, onPhase: (phase: UploadPhase) => void) {
  if (!file.name.toLowerCase().endsWith(ACCEPT)) {
    throw new BackendError('unsupported_type', 'Only PDF files are supported')
  }
  if (file.size > MAX_FILE_MB * 1024 * 1024) {
    throw new BackendError('file_too_large', `Larger than ${MAX_FILE_MB} MB`)
  }

  onPhase({ phase: 'hashing' })
  const hash = await sha256(file)
  const filename = file.name.slice(0, 255)

  const target = await call<UploadTarget>(
    '/documents/upload-url',
    json('POST', { filename, sha256: hash.hex }),
  )

  onPhase({ phase: 'uploading', progress: 0 })
  await putObject(target.upload_url, file, hash.base64, (progress) =>
    onPhase({ phase: 'uploading', progress }),
  )

  onPhase({ phase: 'registering' })
  return call<{ document_id: string; status: string }>(
    '/documents',
    json('POST', {
      object_key: target.object_key,
      filename,
      sha256: hash.hex,
      size_bytes: file.size,
    }),
  )
}
