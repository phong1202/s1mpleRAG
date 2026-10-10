'use client'

import useSWR from 'swr'
import { DOCUMENTS_KEY, fetchDocuments } from '@/lib/backend'
import type { DocumentSummary } from '@/lib/types'

const SETTLED = new Set<DocumentSummary['status']>(['COMPLETED', 'DEAD_LETTER'])
const POLL_MS = 3000

export function useDocuments() {
  const { data, error, isLoading, mutate } = useSWR(DOCUMENTS_KEY, fetchDocuments, {
    // Poll only while the worker still has something in flight; a settled
    // list does not change on its own.
    refreshInterval: (docs) => (docs?.some((doc) => !SETTLED.has(doc.status)) ? POLL_MS : 0),
  })
  return { documents: data ?? [], error, isLoading, mutate }
}
