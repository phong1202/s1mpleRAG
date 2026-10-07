'use client'

import { useEffect, useId, useRef } from 'react'

// The native <dialog>, opened with showModal(): focus is kept inside, Esc
// cancels, the page behind is inert, and assistive tech announces it -- all
// without a library.
export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel,
  cancelLabel,
  onConfirm,
  onCancel,
}: {
  open: boolean
  title: string
  description: React.ReactNode
  confirmLabel: string
  cancelLabel: string
  onConfirm: () => void
  onCancel: () => void
}) {
  const ref = useRef<HTMLDialogElement>(null)
  const titleId = useId()
  const descriptionId = useId()

  useEffect(() => {
    const dialog = ref.current
    if (!dialog) return
    if (open && !dialog.open) dialog.showModal()
    if (!open && dialog.open) dialog.close()
  }, [open])

  return (
    <dialog
      ref={ref}
      aria-labelledby={titleId}
      aria-describedby={descriptionId}
      // Esc: let React state close it, so `open` never disagrees with the DOM.
      onCancel={(e) => {
        e.preventDefault()
        onCancel()
      }}
      // A click whose target is the <dialog> itself landed on the backdrop:
      // the content fills the dialog's box, so a click inside never does.
      onClick={(e) => {
        if (e.target === e.currentTarget) onCancel()
      }}
      className="fixed inset-0 m-auto h-fit w-[calc(100%-2rem)] max-w-sm rounded-xl border bg-background p-0 text-foreground shadow-xl backdrop:bg-black/40"
    >
      <div className="flex flex-col gap-4 p-5">
        <div className="flex flex-col gap-1.5">
          <h2 id={titleId} className="text-base font-semibold">
            {title}
          </h2>
          <div id={descriptionId} className="text-sm text-muted-foreground text-pretty">
            {description}
          </div>
        </div>
        <div className="flex justify-end gap-2">
          {/* Focused on open: Enter on a reflex keeps the document. */}
          <button
            type="button"
            autoFocus
            onClick={onCancel}
            className="rounded-lg border px-3 py-1.5 text-sm transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className="rounded-lg bg-red-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-red-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-600/50 focus-visible:ring-offset-2"
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </dialog>
  )
}
