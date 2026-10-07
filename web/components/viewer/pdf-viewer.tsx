'use client'

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Document, Page, pdfjs, type DocumentProps } from 'react-pdf'
import 'react-pdf/dist/Page/TextLayer.css'
import { FileText, Loader2, Maximize2, Minus, Plus, RotateCw, X } from 'lucide-react'
import { BackendError, downloadDocument } from '@/lib/backend'
import type { Rect, ViewerTarget } from '@/lib/types'
import { useLocale } from '../locale-provider'

// Must be set in the module that renders <Document>: react-pdf's own default
// would otherwise win on module evaluation order. Bundled from pdfjs-dist, so
// the worker is served from this origin, never a CDN.
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString()

// Module-level: react-pdf reloads the document whenever `options` changes
// identity. The data files are copied into public/ by scripts/copy-pdfjs-assets.mjs.
const OPTIONS: DocumentProps['options'] = {
  cMapUrl: '/pdfjs/cmaps/',
  standardFontDataUrl: '/pdfjs/standard_fonts/',
  wasmUrl: '/pdfjs/wasm/',
  enableXfa: false,
}

const ZOOMS = [0.5, 0.75, 1, 1.25, 1.5, 2, 3]
const FIT = 1 // zoom 1 = page width fills the viewer
const PAD = 16
const GAP = 12
// Pages rendered beyond the visible area, in viewport heights, so a normal
// scroll never shows a blank page while only a window of canvases exists.
const OVERSCAN = 1

type Size = { w: number; h: number }
type Load =
  | { status: 'loading' }
  | { status: 'ready'; file: { data: Uint8Array } }
  | { status: 'error'; message: string }

export function PdfViewer({
  docId,
  filename,
  request,
  onNavigate,
  onClose,
}: {
  docId: string
  filename: string
  request: { target: ViewerTarget; seq: number } | null
  onNavigate: (target: ViewerTarget) => void
  onClose: () => void
}) {
  const { t } = useLocale()
  const [load, setLoad] = useState<Load>({ status: 'loading' })
  const [attempt, setAttempt] = useState(0)
  const [sizes, setSizes] = useState<Size[] | null>(null)
  const [zoom, setZoom] = useState(FIT)
  const [viewport, setViewport] = useState({ width: 0, height: 0 })
  const [scrollTop, setScrollTop] = useState(0)
  const [highlight, setHighlight] = useState<{ page: number; rects: Rect[] } | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const controller = new AbortController()
    setLoad({ status: 'loading' })
    setSizes(null)
    setHighlight(null)
    downloadDocument(docId, controller.signal)
      .then((data) => setLoad({ status: 'ready', file: { data } }))
      .catch((error) => {
        if (controller.signal.aborted) return
        const message =
          error instanceof BackendError ? (error.key && t.errors[error.key]) || error.message : t.viewer.error
        setLoad({ status: 'error', message })
      })
    return () => controller.abort()
    // t only words the error message; switching language must not refetch.
  }, [docId, attempt])

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const observer = new ResizeObserver(() =>
      setViewport({ width: el.clientWidth, height: el.clientHeight }),
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  const docKey = `${docId}:${attempt}`
  const currentKey = useRef(docKey)
  currentKey.current = docKey

  async function onLoadSuccess(pdf: Parameters<NonNullable<DocumentProps['onLoadSuccess']>>[0]) {
    // Every page's size up front: placeholders need exact heights for the
    // scrollbar to be right and for a target page's offset to be known
    // before it has ever been rendered.
    const key = docKey
    const pages = await Promise.all(
      Array.from({ length: pdf.numPages }, (_, i) =>
        pdf.getPage(i + 1).then((page) => {
          const { width, height } = page.getViewport({ scale: 1 })
          return { w: width, h: height }
        }),
      ),
    )
    // Another document may have been opened while these were measured.
    if (currentKey.current === key) setSizes(pages)
  }

  const pageWidth = Math.max(160, viewport.width - 2 * PAD) * zoom
  const layout = useMemo(() => {
    if (!sizes) return null
    let top = PAD
    const pages = sizes.map(({ w, h }) => {
      const height = (pageWidth * h) / w
      const page = { top, height }
      top += height + GAP
      return page
    })
    return { pages, total: top - GAP + PAD }
  }, [sizes, pageWidth])

  // Keep the reader on the same spot when the layout's height changes --
  // zooming, or dragging the split -- instead of jumping elsewhere.
  const previousTotal = useRef<number | null>(null)
  useLayoutEffect(() => {
    const el = scrollRef.current
    if (!el || !layout) return
    if (previousTotal.current && previousTotal.current !== layout.total) {
      el.scrollTop = (el.scrollTop * layout.total) / previousTotal.current
    }
    previousTotal.current = layout.total
  }, [layout])

  // Bring each new request into view once its page's offset is known.
  const handledSeq = useRef(0)
  useEffect(() => {
    const el = scrollRef.current
    if (!el || !layout || !request || request.seq === handledSeq.current) return
    handledSeq.current = request.seq
    const index = Math.min(Math.max(request.target.page, 1), layout.pages.length) - 1
    const page = layout.pages[index]
    const rects = request.target.rects ?? []
    const firstY = rects.length ? Math.min(...rects.map((r) => r[1])) : 0
    el.scrollTo({ top: Math.max(0, page.top + firstY * page.height - 3 * GAP), behavior: 'smooth' })
    setHighlight(rects.length ? { page: index + 1, rects } : null)
  }, [layout, request])

  const scrollLeftPad = Math.max(PAD, (viewport.width - pageWidth) / 2)
  const dpr = typeof window === 'undefined' ? 1 : Math.min(window.devicePixelRatio || 1, 2)

  let currentPage = 1
  const visible: number[] = []
  if (layout) {
    const from = scrollTop - OVERSCAN * viewport.height
    const to = scrollTop + (1 + OVERSCAN) * viewport.height
    const anchor = scrollTop + viewport.height * 0.25
    layout.pages.forEach((p, i) => {
      if (p.top + p.height >= from && p.top <= to) visible.push(i + 1)
      if (p.top <= anchor) currentPage = i + 1
    })
  }

  return (
    <section
      aria-label={t.viewer.title}
      className="@container flex h-full min-h-0 flex-col bg-muted/30"
      onKeyDown={(e) => {
        if (e.key === 'Escape') onClose()
      }}
    >
      <Toolbar
        filename={filename}
        currentPage={currentPage}
        pageCount={sizes?.length ?? 0}
        zoom={zoom}
        onZoom={setZoom}
        onGoTo={(page) => onNavigate({ page })}
        onClose={onClose}
      />

      <div
        ref={scrollRef}
        onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
        // The rendered pages are untrusted content; the context menu would
        // offer "save image" on a canvas, which the viewer does not.
        onContextMenu={(e) => e.preventDefault()}
        className="relative min-h-0 flex-1 overflow-auto"
      >
        {load.status === 'loading' && <Status icon spin text={t.viewer.loading} />}
        {load.status === 'error' && (
          <Status text={load.message} action={{ label: t.viewer.retry, onClick: () => setAttempt((n) => n + 1) }} />
        )}
        {load.status === 'ready' && (
          <Document
            key={docKey}
            file={load.file}
            options={OPTIONS}
            suspense={false}
            loading={<Status icon spin text={t.viewer.loading} />}
            error={<Status text={t.viewer.error} />}
            onLoadSuccess={onLoadSuccess}
            onLoadError={() => setLoad({ status: 'error', message: t.viewer.error })}
          >
            {layout && (
              <div className="relative" style={{ height: layout.total, width: pageWidth + 2 * scrollLeftPad }}>
                {layout.pages.map((p, i) => {
                  const number = i + 1
                  return (
                    <div
                      key={number}
                      className="absolute overflow-hidden bg-white shadow-sm ring-1 ring-black/5"
                      style={{ top: p.top, left: scrollLeftPad, width: pageWidth, height: p.height }}
                    >
                      {visible.includes(number) && (
                        <Page
                          pageNumber={number}
                          width={pageWidth}
                          devicePixelRatio={dpr}
                          renderTextLayer
                          // Links, forms and scripts in an uploaded PDF are
                          // never rendered: no annotation layer at all.
                          renderAnnotationLayer={false}
                          renderForms={false}
                          loading={null}
                        />
                      )}
                      {highlight?.page === number &&
                        highlight.rects.map(([x0, y0, x1, y1], k) => (
                          <div
                            key={k}
                            aria-hidden="true"
                            className="pointer-events-none absolute rounded-sm bg-yellow-300/35 ring-2 ring-yellow-500/70"
                            style={{
                              left: `${x0 * 100}%`,
                              top: `${y0 * 100}%`,
                              width: `${(x1 - x0) * 100}%`,
                              height: `${(y1 - y0) * 100}%`,
                            }}
                          />
                        ))}
                    </div>
                  )
                })}
              </div>
            )}
          </Document>
        )}
      </div>
    </section>
  )
}

function Toolbar({
  filename,
  currentPage,
  pageCount,
  zoom,
  onZoom,
  onGoTo,
  onClose,
}: {
  filename: string
  currentPage: number
  pageCount: number
  zoom: number
  onZoom: (zoom: number) => void
  onGoTo: (page: number) => void
  onClose: () => void
}) {
  const { t } = useLocale()
  const [draft, setDraft] = useState<string | null>(null)
  const index = ZOOMS.indexOf(zoom)

  const iconButton =
    'flex size-7 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:pointer-events-none disabled:opacity-40'

  return (
    // Below ~28rem (the 30% split on a laptop) the name gets its own row:
    // squeezed between the controls it truncates to a letter or two.
    <div className="flex shrink-0 flex-wrap items-center gap-x-2 gap-y-1 border-b bg-background px-3 py-2">
      <div className="flex min-w-0 basis-full items-center gap-2 @md:basis-0 @md:flex-1">
        <FileText className="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
        <p className="min-w-0 flex-1 truncate text-sm font-medium" title={filename}>
          {filename}
        </p>
      </div>

      {pageCount > 0 && (
        <form
          className="flex items-center gap-1 text-xs text-muted-foreground"
          onSubmit={(e) => {
            e.preventDefault()
            const page = Number(draft)
            if (Number.isInteger(page) && page >= 1 && page <= pageCount) onGoTo(page)
            setDraft(null)
          }}
        >
          <label htmlFor="viewer-page" className="sr-only">
            {t.viewer.page}
          </label>
          <input
            id="viewer-page"
            inputMode="numeric"
            value={draft ?? String(currentPage)}
            onChange={(e) => setDraft(e.target.value.replace(/\D/g, ''))}
            onFocus={(e) => e.target.select()}
            onBlur={() => setDraft(null)}
            className="h-7 w-10 rounded-md border bg-background text-center text-xs tabular-nums text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
          />
          <span className="tabular-nums">/ {pageCount}</span>
        </form>
      )}

      <div className="ml-auto flex items-center gap-0.5 @md:ml-0">
        <button
          type="button"
          className={iconButton}
          onClick={() => onZoom(ZOOMS[index - 1])}
          disabled={index <= 0}
          aria-label={t.viewer.zoomOut}
        >
          <Minus className="size-3.5" aria-hidden="true" />
        </button>
        <span className="w-10 text-center text-xs tabular-nums text-muted-foreground">
          {Math.round(zoom * 100)}%
        </span>
        <button
          type="button"
          className={iconButton}
          onClick={() => onZoom(ZOOMS[index + 1])}
          disabled={index >= ZOOMS.length - 1}
          aria-label={t.viewer.zoomIn}
        >
          <Plus className="size-3.5" aria-hidden="true" />
        </button>
        <button
          type="button"
          className={iconButton}
          onClick={() => onZoom(FIT)}
          disabled={zoom === FIT}
          aria-label={t.viewer.fitWidth}
          title={t.viewer.fitWidth}
        >
          <Maximize2 className="size-3.5" aria-hidden="true" />
        </button>
      </div>

      <button type="button" className={iconButton} onClick={onClose} aria-label={t.viewer.close}>
        <X className="size-4" aria-hidden="true" />
      </button>
    </div>
  )
}

function Status({
  text,
  icon,
  spin,
  action,
}: {
  text: string
  icon?: boolean
  spin?: boolean
  action?: { label: string; onClick: () => void }
}) {
  return (
    <div role="status" className="flex h-full flex-col items-center justify-center gap-3 p-6 text-center">
      {icon && <Loader2 className={spin ? 'size-5 animate-spin text-primary' : 'size-5'} aria-hidden="true" />}
      <p className="text-sm text-muted-foreground">{text}</p>
      {action && (
        <button
          type="button"
          onClick={action.onClick}
          className="flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs transition-colors hover:bg-muted"
        >
          <RotateCw className="size-3" aria-hidden="true" />
          {action.label}
        </button>
      )}
    </div>
  )
}
