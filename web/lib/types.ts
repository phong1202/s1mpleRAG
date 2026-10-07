export type Locale = 'en' | 'vi'

// The closed set the backend's ck_documents_status constraint enforces.
export type DocumentStatus =
  | 'QUEUED'
  | 'PARSING'
  | 'STRUCTURING'
  | 'ENRICHING'
  | 'EMBEDDING'
  | 'PERSISTING'
  | 'COMPLETED'
  | 'RETRYING'
  | 'DEAD_LETTER'

// Progress of the running stage, overlaid by the API from Redis (backend
// plan, phase 1d task 31). `stalled` means no report for a while.
export type DocumentProgress = {
  stage: string
  done: number | null
  total: number | null
  updated_at: string | null
  stalled: boolean
}

// GET /documents item, as the backend's DocumentStatus schema serialises it.
// Everything after last_error is phase 1d task 30/31: optional so the UI
// works against a backend that does not send them yet.
export type DocumentSummary = {
  id: string
  filename: string
  status: DocumentStatus
  stage: string | null
  attempts: number
  failed_stage: string | null
  last_error: string | null
  title?: string | null
  page_count?: number | null
  size_bytes?: number | null
  language?: string | null
  created_at?: string
  updated_at?: string
  completed_at?: string | null
  progress?: DocumentProgress | null
}

// [x0, y0, x1, y1] as fractions (0..1) of the page, origin top-left, so a
// rect means the same region at any zoom level.
export type Rect = [number, number, number, number]

// Where a chunk sits in its PDF. A chunk can span blocks and pages, hence a
// list. Provisional, like SourceReference: proposed to the backend, which
// does not produce it yet.
export type SourceLocation = { page: number; rect?: Rect }

// What the chat UI renders for a cited passage. Provisional: the backend has
// no chat endpoint yet, so this is the UI's need, not a wire format.
export type SourceReference = {
  documentId: string
  documentName: string
  index: number
  text: string
  score: number
  locations?: SourceLocation[]
}

// What the PDF viewer should bring into view: a page, optionally narrowed to
// the regions to highlight on it.
export type ViewerTarget = { page: number; rects?: Rect[] }

export type Message = {
  id: string
  role: 'user' | 'assistant'
  text: string
  sources?: SourceReference[]
}
