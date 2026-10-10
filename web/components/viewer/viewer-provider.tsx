'use client'

import { createContext, useCallback, useContext, useMemo, useState } from 'react'
import type { ViewerTarget } from '@/lib/types'

// `seq` makes every open() a new request even when the target is unchanged:
// clicking the same citation twice has to scroll back to it the second time.
type Request = { target: ViewerTarget; seq: number }

type ViewerContextValue = {
  docId: string | null
  request: Request | null
  // One document at a time: opening another replaces the current one.
  open: (docId: string, target?: ViewerTarget) => void
  close: () => void
}

const ViewerContext = createContext<ViewerContextValue | null>(null)

export function ViewerProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<{ docId: string | null; request: Request | null }>({
    docId: null,
    request: null,
  })

  const open = useCallback((docId: string, target?: ViewerTarget) => {
    setState((prev) => ({
      docId,
      request: target ? { target, seq: (prev.request?.seq ?? 0) + 1 } : null,
    }))
  }, [])

  const close = useCallback(() => setState({ docId: null, request: null }), [])

  const value = useMemo(() => ({ ...state, open, close }), [state, open, close])
  return <ViewerContext.Provider value={value}>{children}</ViewerContext.Provider>
}

export function useViewer() {
  const context = useContext(ViewerContext)
  if (!context) throw new Error('useViewer must be used within ViewerProvider')
  return context
}
