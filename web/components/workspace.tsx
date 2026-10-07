'use client'

import { useEffect, useSyncExternalStore } from 'react'
import dynamic from 'next/dynamic'
import { Group, Panel, Separator, useDefaultLayout, type LayoutStorage } from 'react-resizable-panels'
import { useDocuments } from '@/hooks/use-documents'
import { displayName } from '@/lib/documents'
import { ChatPanel } from './chat-panel'
import { DocumentPanel } from './document-panel'
import { useLocale } from './locale-provider'
import { useViewer, ViewerProvider } from './viewer/viewer-provider'

// pdf.js needs browser APIs and ships a ~1 MB worker: load it only on the
// client, and only once a document is actually opened.
const PdfViewer = dynamic(() => import('./viewer/pdf-viewer').then((m) => m.PdfViewer), {
  ssr: false,
})

// The split is remembered per browser. Storage can be missing or throw
// (private windows, blocked site data); the layout then just resets.
const storage: LayoutStorage = {
  getItem: (key) => {
    try {
      return localStorage.getItem(key)
    } catch {
      return null
    }
  },
  setItem: (key, value) => {
    try {
      localStorage.setItem(key, value)
    } catch {}
  },
}

const DESKTOP = '(min-width: 768px)'

function useIsDesktop() {
  return useSyncExternalStore(
    (onChange) => {
      const query = window.matchMedia(DESKTOP)
      query.addEventListener('change', onChange)
      return () => query.removeEventListener('change', onChange)
    },
    () => window.matchMedia(DESKTOP).matches,
    () => true,
  )
}

export function Workspace() {
  return (
    <ViewerProvider>
      <main className="flex min-h-0 flex-1 flex-col md:flex-row">
        <div className="max-h-[45dvh] shrink-0 md:max-h-none md:w-80 lg:w-96">
          <DocumentPanel />
        </div>
        <div className="min-h-0 flex-1">
          <ChatArea />
        </div>
      </main>
    </ViewerProvider>
  )
}

function ChatArea() {
  const { t } = useLocale()
  const { docId, request, open, close } = useViewer()
  const { documents, isLoading } = useDocuments()
  const isDesktop = useIsDesktop()

  const doc = documents.find((d) => d.id === docId)
  const showSplit = Boolean(docId) && isDesktop
  const { defaultLayout, onLayoutChanged } = useDefaultLayout({
    id: 'chat-viewer',
    panelIds: showSplit ? ['chat', 'viewer'] : ['chat'],
    storage,
  })

  // A document that was deleted, or is no longer ready, cannot stay open.
  useEffect(() => {
    if (docId && !isLoading && doc?.status !== 'COMPLETED') close()
  }, [docId, doc?.status, isLoading, close])

  const viewer =
    docId && doc ? (
      <PdfViewer
        key={docId}
        docId={docId}
        filename={displayName(doc)}
        request={request}
        onNavigate={(target) => open(docId, target)}
        onClose={close}
      />
    ) : null

  return (
    <>
      <Group
        orientation="horizontal"
        defaultLayout={defaultLayout}
        onLayoutChanged={onLayoutChanged}
        className="h-full"
      >
        {/* Always the first child, so opening or closing the viewer never
            remounts the chat and loses the conversation. */}
        <Panel id="chat" minSize="50%">
          <ChatPanel />
        </Panel>
        {showSplit && (
          <>
            <Separator
              aria-label={t.viewer.resize}
              className="w-1.5 shrink-0 border-l bg-border/40 outline-none transition-colors hover:bg-primary/30 focus-visible:bg-primary/40"
            />
            <Panel id="viewer" defaultSize="40%" minSize="30%" maxSize="50%">
              {viewer}
            </Panel>
          </>
        )}
      </Group>

      {/* Below md there is no room to split: the viewer covers the screen. */}
      {docId && !isDesktop && <div className="fixed inset-0 z-50 bg-background">{viewer}</div>}
    </>
  )
}
