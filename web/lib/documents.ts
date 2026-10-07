import type { DocumentSummary } from '@/lib/types'

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

// Fraction done of the running stage, or null when there is nothing honest
// to draw: no report, an unknown total, or a report about another stage
// (the backend only reports OCR so far; enrich/embed have none).
export function stageProgress(doc: DocumentSummary) {
  const p = doc.progress
  if (!p || p.stage !== doc.status || !p.total || p.done == null) return null
  return { done: p.done, total: p.total, ratio: Math.min(1, p.done / p.total) }
}

export function isStalled(doc: DocumentSummary) {
  return Boolean(doc.progress?.stalled && doc.progress.stage === doc.status)
}
