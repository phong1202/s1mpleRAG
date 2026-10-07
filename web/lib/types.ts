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

// GET /documents item, as the backend's DocumentStatus schema serialises it.
export type DocumentSummary = {
  id: string
  filename: string
  status: DocumentStatus
  stage: string | null
  attempts: number
  failed_stage: string | null
  last_error: string | null
}

// What the chat UI renders for a cited passage. Provisional: the backend has
// no chat endpoint yet, so this is the UI's need, not a wire format.
export type SourceReference = {
  documentId: string
  documentName: string
  index: number
  text: string
  score: number
}

export type Message = {
  id: string
  role: 'user' | 'assistant'
  text: string
  sources?: SourceReference[]
}
