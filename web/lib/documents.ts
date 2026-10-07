import type { DocumentSummary, ProgressUnit } from '@/lib/types'

// The extracted title reads better than `nd-168-2024_xu-phat(1).pdf`; the
// filename stays the fallback, and the tooltip.
export function displayName(doc: DocumentSummary) {
  return doc.title?.trim() || doc.filename
}

export function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

// OCR counts pages; enrich and embed work through the document's chunks.
// Only a fallback for an API that does not say which.
const UNIT_BY_STAGE: Record<string, ProgressUnit> = {
  PARSING: 'pages',
  ENRICHING: 'chunks',
  EMBEDDING: 'chunks',
}

// Fraction done of the running stage, or null when there is nothing honest
// to draw: no report, an unknown total, or a report about another stage
// than the one the document is in.
export function stageProgress(doc: DocumentSummary) {
  const p = doc.progress
  if (!p || p.stage !== doc.status || !p.total || p.done == null) return null
  const unit = p.unit ?? UNIT_BY_STAGE[p.stage] ?? null
  return { done: p.done, total: p.total, unit, ratio: Math.min(1, p.done / p.total) }
}

export function isStalled(doc: DocumentSummary) {
  return Boolean(doc.progress?.stalled && doc.progress.stage === doc.status)
}
