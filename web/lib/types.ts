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

export type ProgressUnit = 'pages' | 'chunks'

// Progress of the running stage, overlaid by the API from Redis (backend
// plan, phase 1d task 31). `stalled` means no report for a while.
export type DocumentProgress = {
  stage: string
  // What done/total count. Not sent by the API yet; see lib/documents.ts.
  unit?: ProgressUnit
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

// What the PDF viewer should bring into view: a page, optionally narrowed to
// the regions to highlight on it. Citations only carry a page so far: the
// backend does not produce regions yet (docs/system-design.md, 5.3).
export type ViewerTarget = { page: number; rects?: Rect[] }

// POST /query citation, as the backend's CitationOut serialises it: the
// document and web kinds flattened into one shape. `ordinal` is the n of the
// [n] markers in the answer, so it can skip numbers. `quote` is the first
// 280 characters of the passage that matched the question, straight from
// OCR: text, never HTML.
export type Citation = {
  ordinal: number
  kind: 'document' | 'web'
  document_id: string | null
  filename: string | null
  page_number: number | null
  quote: string | null
  url: string | null
  title: string | null
}

// POST /query result. The citations are the sources the answer actually
// cites, in order of first use; an answer the corpus cannot support, or a
// reply to small talk, has none. `conversation_id` is always set: the one the
// question was sent with, or a new one when it was sent without.
export type QueryAnswer = {
  answer: string
  citations: Citation[]
  latency_ms: number
  conversation_id: string
}

export type Message = {
  id: string
  role: 'user' | 'assistant'
  text: string
  citations?: Citation[]
}

// GET /me. Until login exists the backend treats every request as its seeded
// admin (docs/be-conversations-handoff.md).
export type CurrentUser = {
  id: string
  username: string
  display_name: string
  is_admin: boolean
}

// GET /conversations item: the current user's conversations, most recently
// updated first. `title` is the first question.
export type ConversationSummary = {
  id: string
  title: string | null
  created_at: string
  updated_at: string
  message_count: number
}

// One turn as the backend stored it. `citations` is set on answers only,
// [] when the answer cited nothing, and null on answers stored before the
// backend kept them.
export type StoredMessage = {
  id: number
  role: 'user' | 'assistant'
  text: string
  citations: Citation[] | null
  created_at: string
}

// GET /conversations/{id}: the turns oldest first.
export type ConversationDetail = {
  id: string
  title: string | null
  created_at: string
  updated_at: string
  messages: StoredMessage[]
}
