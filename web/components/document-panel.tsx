'use client'

import { useRef, useState } from 'react'
import { AlertTriangle, CheckCircle2, FileText, Globe, Loader2, Trash2, Upload } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useDocuments } from '@/hooks/use-documents'
import {
  ACCEPT,
  BackendError,
  deleteDocument as deleteRemote,
  uploadDocument,
  type UploadPhase,
} from '@/lib/backend'
import { displayName, formatBytes, isStalled, stageProgress } from '@/lib/documents'
import type { DocumentSummary } from '@/lib/types'
import type { Dictionary } from '@/lib/i18n'
import { useLocale } from './locale-provider'
import { useToast } from './toast-provider'
import { useViewer } from './viewer/viewer-provider'

type PendingUpload = { id: string; name: string } & UploadPhase

function describePhase(upload: PendingUpload, t: Dictionary) {
  if (upload.phase === 'uploading') return `${t.phases.uploading} ${Math.round(upload.progress * 100)}%`
  return t.phases[upload.phase]
}

function errorMessage(error: unknown, t: Dictionary) {
  if (error instanceof BackendError) return (error.key && t.errors[error.key]) || error.message
  return t.errorGeneric
}

function DocumentLabel({ doc, t }: { doc: DocumentSummary; t: Dictionary }) {
  return (
    <>
      <FileText className="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm" title={doc.filename}>
          {displayName(doc)}
        </p>
        <StatusLine doc={doc} t={t} />
        <ProgressBar doc={doc} t={t} />
      </div>
    </>
  )
}

function StatusLine({ doc, t }: { doc: DocumentSummary; t: Dictionary }) {
  const label = t.statuses[doc.status] ?? doc.status
  if (doc.status === 'COMPLETED') {
    const details = [
      doc.page_count ? `${doc.page_count} ${t.pages}` : null,
      doc.size_bytes ? formatBytes(doc.size_bytes) : null,
    ].filter(Boolean)
    return (
      <p className="flex items-center gap-1 text-xs text-muted-foreground">
        <CheckCircle2 className="size-3 shrink-0 text-primary" aria-hidden="true" />
        <span className="truncate">{[label, ...details].join(' · ')}</span>
      </p>
    )
  }
  if (doc.status === 'DEAD_LETTER') {
    return (
      <p className="flex items-center gap-1 text-xs text-destructive" title={doc.last_error ?? undefined}>
        <AlertTriangle className="size-3 shrink-0" aria-hidden="true" />
        <span className="truncate">
          {label}
          {doc.failed_stage && ` · ${doc.failed_stage}`}
        </span>
      </p>
    )
  }
  if (isStalled(doc)) {
    return (
      <p
        className="flex items-center gap-1 text-xs text-amber-700 dark:text-amber-400"
        title={t.stalledHint}
      >
        <AlertTriangle className="size-3 shrink-0" aria-hidden="true" />
        <span className="truncate">
          {trimEllipsis(label)} · {t.stalled}
        </span>
      </p>
    )
  }
  const progress = stageProgress(doc)
  return (
    <p className="flex items-center gap-1 text-xs text-muted-foreground">
      <Loader2 className="size-3 shrink-0 animate-spin" aria-hidden="true" />
      <span className="truncate">
        {progress ? `${trimEllipsis(label)} · ${progress.done}/${progress.total} ${t.pages}` : label}
        {doc.attempts > 1 && ` · ${t.attempt} ${doc.attempts}`}
      </span>
    </p>
  )
}

// "Reading pages…" ends in an ellipsis that reads wrong once a count follows.
function trimEllipsis(label: string) {
  return label.replace(/…$/, '')
}

function ProgressBar({ doc, t }: { doc: DocumentSummary; t: Dictionary }) {
  const progress = stageProgress(doc)
  if (!progress || isStalled(doc)) return null
  return (
    <div
      role="progressbar"
      aria-label={t.statuses[doc.status] ?? doc.status}
      aria-valuemin={0}
      aria-valuemax={progress.total}
      aria-valuenow={progress.done}
      className="mt-1.5 h-1 overflow-hidden rounded-full bg-muted"
    >
      <div
        className="h-full rounded-full bg-primary transition-[width] duration-500"
        style={{ width: `${progress.ratio * 100}%` }}
      />
    </div>
  )
}

export function DocumentPanel() {
  const { t } = useLocale()
  const { documents, error: listError, isLoading, mutate } = useDocuments()
  const viewer = useViewer()
  const inputRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState<PendingUpload[]>([])
  const { notify } = useToast()
  const [dragging, setDragging] = useState(false)

  async function uploadFiles(files: FileList | File[]) {
    const list = Array.from(files).map((file) => ({ id: crypto.randomUUID(), file }))
    if (list.length === 0) return
    setUploading((prev) => [
      ...prev,
      ...list.map(({ id, file }) => ({ id, name: file.name, phase: 'hashing' as const })),
    ])

    await Promise.all(
      list.map(async ({ id, file }) => {
        try {
          await uploadDocument(file, (phase) =>
            setUploading((prev) => prev.map((u) => (u.id === id ? { ...u, ...phase } : u))),
          )
        } catch (error) {
          notify(`${file.name}: ${errorMessage(error, t)}`)
        } finally {
          setUploading((prev) => prev.filter((u) => u.id !== id))
        }
      }),
    )
    await mutate()
  }

  async function deleteDocument(id: string) {
    const without = (current?: DocumentSummary[]) => (current ?? []).filter((doc) => doc.id !== id)
    try {
      await mutate(
        async (current) => {
          await deleteRemote(id)
          return without(current)
        },
        { optimisticData: without, rollbackOnError: true, revalidate: false },
      )
    } catch (error) {
      notify(errorMessage(error, t))
    }
  }

  return (
    <aside className="flex h-full flex-col gap-5 overflow-hidden border-b bg-card/40 p-5 md:border-r md:border-b-0">
      <div className="flex flex-col gap-1">
        <h2 className="text-sm font-semibold">{t.knowledgeBase}</h2>
        <p className="text-xs text-muted-foreground">{t.knowledgeBaseHint}</p>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          uploadFiles(e.dataTransfer.files)
        }}
        className={cn(
          'flex flex-col items-center gap-3 rounded-xl border border-dashed p-5 text-center transition-colors',
          dragging ? 'border-primary bg-primary/5' : 'bg-background/40',
        )}
      >
        <div className="flex size-10 items-center justify-center rounded-full bg-primary/10 text-primary">
          <Upload className="size-4" aria-hidden="true" />
        </div>
        <div className="flex flex-col gap-1">
          <p className="text-sm font-medium">{t.uploadTitle}</p>
          <p className="text-xs text-muted-foreground text-pretty">{t.uploadHint}</p>
        </div>
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="rounded-lg bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground transition-opacity hover:opacity-90"
        >
          {t.uploadCta}
        </button>
        <p className="text-xs text-muted-foreground">{t.dropHere}</p>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          multiple
          className="sr-only"
          aria-label={t.uploadTitle}
          onChange={(e) => {
            if (e.target.files) uploadFiles(e.target.files)
            e.target.value = ''
          }}
        />
      </div>

      <ul className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto" aria-label={t.knowledgeBase}>
        {uploading.map((upload) => (
          <li key={upload.id} className="flex items-center gap-3 rounded-lg border bg-background/40 p-3">
            <Loader2 className="size-4 shrink-0 animate-spin text-primary" aria-hidden="true" />
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm">{upload.name}</p>
              <p className="text-xs text-muted-foreground">{describePhase(upload, t)}</p>
            </div>
          </li>
        ))}

        {listError && (
          <li role="alert" className="px-1 py-4 text-center text-xs text-destructive">
            {errorMessage(listError, t)}
          </li>
        )}

        {!isLoading && !listError && documents.length === 0 && uploading.length === 0 && (
          <li className="px-1 py-4 text-center text-xs text-muted-foreground">{t.noDocuments}</li>
        )}

        {documents.map((doc) => (
          <li
            key={doc.id}
            className={cn(
              'group flex items-center gap-1 rounded-lg border p-1.5 transition-colors',
              viewer.docId === doc.id ? 'border-primary/60 bg-primary/5' : 'bg-background/40',
            )}
          >
            {doc.status === 'COMPLETED' ? (
              <button
                type="button"
                onClick={() => viewer.open(doc.id)}
                aria-pressed={viewer.docId === doc.id}
                aria-label={`${t.viewer.open} ${displayName(doc)}`}
                className="flex min-w-0 flex-1 items-center gap-3 rounded-md p-1.5 text-left transition-colors hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
              >
                <DocumentLabel doc={doc} t={t} />
              </button>
            ) : (
              <div className="flex min-w-0 flex-1 items-center gap-3 p-1.5">
                <DocumentLabel doc={doc} t={t} />
              </div>
            )}
            <button
              type="button"
              onClick={() => deleteDocument(doc.id)}
              className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
              aria-label={`${t.remove} ${displayName(doc)}`}
            >
              <Trash2 className="size-4" aria-hidden="true" />
            </button>
          </li>
        ))}
      </ul>

      <div className="flex items-center gap-2 rounded-lg border border-dashed px-3 py-2 text-xs text-muted-foreground">
        <Globe className="size-3.5" aria-hidden="true" />
        {t.comingSoon}
      </div>
    </aside>
  )
}
