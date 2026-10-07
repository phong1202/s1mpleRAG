'use client'

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, X } from 'lucide-react'
import { useLocale } from './locale-provider'

type Toast = { id: number; message: string }

const DURATION_MS = 6000
// Older toasts give way rather than stacking up the screen when a whole
// batch of uploads fails at once.
const MAX_VISIBLE = 4

const ToastContext = createContext<((message: string) => void) | null>(null)

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const { t } = useLocale()
  const [toasts, setToasts] = useState<Toast[]>([])
  const nextId = useRef(0)

  // Newest first: the stack grows down from the header, so the latest is
  // the one nearest the top of the eye line.
  const notify = useCallback((message: string) => {
    const id = ++nextId.current
    setToasts((prev) => [{ id, message }, ...prev].slice(0, MAX_VISIBLE))
  }, [])

  const dismiss = useCallback((id: number) => setToasts((prev) => prev.filter((t) => t.id !== id)), [])

  return (
    <ToastContext.Provider value={notify}>
      {children}
      <ol
        aria-label={t.notifications}
        // Top right, just under the header: where the eye goes, unlike the
        // bottom-left corner, which is easy to miss entirely.
        className="pointer-events-none fixed inset-x-4 top-16 z-[60] flex flex-col gap-2 md:left-auto md:w-96"
      >
        {toasts.map((toast) => (
          <ToastItem key={toast.id} toast={toast} onDismiss={dismiss} />
        ))}
      </ol>
    </ToastContext.Provider>
  )
}

function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss: (id: number) => void }) {
  const { t } = useLocale()
  const [paused, setPaused] = useState(false)

  // Restarts from the full duration after a pause: someone who stopped to
  // read it gets the whole time again, not a sliver.
  useEffect(() => {
    if (paused) return
    const timer = setTimeout(() => onDismiss(toast.id), DURATION_MS)
    return () => clearTimeout(timer)
  }, [paused, toast.id, onDismiss])

  return (
    <li
      role="alert"
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      onFocus={() => setPaused(true)}
      onBlur={() => setPaused(false)}
      className="pointer-events-auto flex items-start gap-2.5 rounded-xl bg-red-600 p-3 text-sm font-medium text-white shadow-lg shadow-red-600/25 animate-in fade-in slide-in-from-top-2"
    >
      <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
      <p className="min-w-0 flex-1 break-words text-pretty">{toast.message}</p>
      <button
        type="button"
        onClick={() => onDismiss(toast.id)}
        className="-m-1 rounded-md p-1 text-white/80 transition-colors hover:bg-white/15 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/70"
        aria-label={t.dismiss}
      >
        <X className="size-3.5" aria-hidden="true" />
      </button>
    </li>
  )
}

export function useToast() {
  const notify = useContext(ToastContext)
  if (!notify) throw new Error('useToast must be used within ToastProvider')
  return useMemo(() => ({ notify }), [notify])
}
