'use client'

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, X } from 'lucide-react'
import { useLocale } from './locale-provider'

// `key` lets a caller keep updating one toast instead of adding another:
// a batch upload reports each failure reason once, with a growing list.
type Toast = { id: number; key?: string; message: string }
type Notify = (message: string, key?: string) => void

const DURATION_MS = 6000

const ToastContext = createContext<Notify | null>(null)

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const { t } = useLocale()
  const [toasts, setToasts] = useState<Toast[]>([])
  const nextId = useRef(0)

  // Newest first: the stack grows down from the header, so the latest is
  // the one nearest the top of the eye line. Nothing is ever dropped to
  // make room -- a toast only leaves when dismissed or timed out -- since a
  // dropped error is a failure nobody hears about.
  const notify = useCallback<Notify>((message, key) => {
    setToasts((prev) => {
      const existing = key ? prev.find((toast) => toast.key === key) : undefined
      // Same id when updating, so the element stays put and its timer
      // restarts on the new message rather than a fresh toast appearing.
      const toast = { id: existing?.id ?? ++nextId.current, key, message }
      return [toast, ...prev.filter((t) => t !== existing)]
    })
  }, [])

  const dismiss = useCallback((id: number) => setToasts((prev) => prev.filter((t) => t.id !== id)), [])

  return (
    <ToastContext.Provider value={notify}>
      {children}
      <ol
        aria-label={t.notifications}
        // Top right, just under the header: where the eye goes, unlike the
        // bottom-left corner, which is easy to miss entirely.
        className="pointer-events-none fixed inset-x-4 top-16 z-[60] flex max-h-[calc(100dvh-5rem)] flex-col gap-2 overflow-y-auto md:left-auto md:w-96"
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

  // Restarts from the full duration after a pause, and whenever the message
  // changes: someone who stopped to read it, or a toast that just gained a
  // line, gets the whole time again, not a sliver.
  useEffect(() => {
    if (paused) return
    const timer = setTimeout(() => onDismiss(toast.id), DURATION_MS)
    return () => clearTimeout(timer)
  }, [paused, toast.id, toast.message, onDismiss])

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
