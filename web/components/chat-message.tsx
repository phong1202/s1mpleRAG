'use client'

import { createContext, useContext, useMemo, useState } from 'react'
import { ChevronDown, FileText, Globe } from 'lucide-react'
import { defaultRehypePlugins, Streamdown, type ExtraProps, type StreamdownProps } from 'streamdown'
import { useDocuments } from '@/hooks/use-documents'
import { displayName } from '@/lib/documents'
import { CITATION_TAG, rehypeCitations } from '@/lib/rehype-citations'
import type { Citation, Message } from '@/lib/types'
import { cn } from '@/lib/utils'
import { useLocale } from './locale-provider'
import { useViewer } from './viewer/viewer-provider'

// The backend cuts a quote at this many characters (app/core/generator.py).
const QUOTE_LIMIT = 280

// A citation as the UI shows it: named after the document's title when the
// document is still in the list, and openable only while it is ready.
type Source = {
  citation: Citation
  name: string
  open: (() => void) | null
}

const SourcesContext = createContext<Map<number, Source>>(new Map())

// The answer is the model's text over passages from PDFs anyone can upload,
// so it is rendered as Markdown and nothing more. Streamdown's default list
// starts with rehype-raw, which parses HTML in the text into real elements;
// without it Streamdown shows such HTML as the text it is. Sanitize and
// harden stay for what Markdown itself can carry (link targets). Citations
// run last, on the tree that is already clean, because sanitize would strip
// their element.
const REHYPE_PLUGINS: StreamdownProps['rehypePlugins'] = [
  defaultRehypePlugins.sanitize,
  defaultRehypePlugins.harden,
  rehypeCitations,
]

// An answer has no reason to embed an image, and one pointing off-site
// would be a request to wherever the PDF's author chose.
const DISALLOWED = ['img']

const COMPONENTS: StreamdownProps['components'] = { [CITATION_TAG]: CitationMarker }

export function ChatMessage({ message }: { message: Message }) {
  const { t } = useLocale()

  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-sm text-primary-foreground whitespace-pre-wrap">
          <span className="sr-only">{t.you}: </span>
          {message.text}
        </div>
      </div>
    )
  }

  return <AssistantMessage text={message.text} citations={message.citations ?? []} />
}

function AssistantMessage({ text, citations }: { text: string; citations: Citation[] }) {
  const { t } = useLocale()
  const { documents } = useDocuments()
  const { open } = useViewer()

  const sources = useMemo(() => {
    const byOrdinal = new Map<number, Source>()
    for (const citation of citations) {
      const { document_id: id, page_number: page } = citation
      const doc = documents.find((d) => d.id === id)
      byOrdinal.set(citation.ordinal, {
        citation,
        name: (doc && displayName(doc)) || citation.filename || citation.title || citation.url || '',
        // Deleted since the answer, or not a document at all: still listed,
        // no longer a way into the viewer.
        open: doc?.status === 'COMPLETED' && id && page ? () => open(id, { page }) : null,
      })
    }
    return byOrdinal
  }, [citations, documents, open])

  return (
    <div className="flex flex-col gap-3">
      <span className="sr-only">{t.assistant}:</span>
      <SourcesContext.Provider value={sources}>
        <div className="prose-chat text-sm leading-relaxed">
          <Streamdown
            mode="static"
            rehypePlugins={REHYPE_PLUGINS}
            disallowedElements={DISALLOWED}
            components={COMPONENTS}
          >
            {text}
          </Streamdown>
        </div>
      </SourcesContext.Provider>
      {sources.size > 0 && <SourceList sources={[...sources.values()]} />}
    </div>
  )
}

function useSourceLabel() {
  const { t } = useLocale()
  return ({ citation, name, open }: Source) => {
    const page = citation.page_number ? `, ${t.viewer.page.toLowerCase()} ${citation.page_number}` : ''
    const gone = citation.kind === 'document' && !open ? ` (${t.sourceUnavailable})` : ''
    return `${t.source} ${citation.ordinal}: ${name}${page}${gone}`
  }
}

// Stands in for a [n] in the answer. `children` is that original text: a
// marker the backend returned no citation for stays exactly as written.
function CitationMarker({ node, children }: ExtraProps & { children?: React.ReactNode }) {
  const source = useContext(SourcesContext).get(Number(node?.properties.ordinal))
  const label = useSourceLabel()
  if (!source) return <>{children}</>

  return (
    <button
      type="button"
      onClick={source.open ?? undefined}
      disabled={!source.open}
      title={label(source)}
      aria-label={label(source)}
      className="mx-0.5 inline-flex h-4 min-w-4 -translate-y-px items-center justify-center rounded bg-primary/15 px-1 align-middle font-mono text-[10px] font-semibold leading-none text-primary transition-colors hover:bg-primary hover:text-primary-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40 disabled:bg-muted disabled:text-muted-foreground"
    >
      {source.citation.ordinal}
    </button>
  )
}

function SourceList({ sources }: { sources: Source[] }) {
  const { t } = useLocale()
  const label = useSourceLabel()
  const [expanded, setExpanded] = useState<number | null>(null)

  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{t.sources}</p>
      <ol className="flex flex-col gap-1.5">
        {sources.map((source) => {
          const { citation, name, open } = source
          const isExpanded = expanded === citation.ordinal
          const quote = citation.quote?.trim()
          const isCut = (citation.quote?.length ?? 0) >= QUOTE_LIMIT
          const Icon = citation.kind === 'web' ? Globe : FileText
          return (
            <li key={citation.ordinal} className="rounded-lg border bg-card/40">
              <div className="flex items-center">
                <button
                  type="button"
                  onClick={open ?? undefined}
                  disabled={!open}
                  title={label(source)}
                  className="flex min-w-0 flex-1 items-center gap-2 rounded-l-lg px-3 py-2 text-left text-xs transition-colors enabled:hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40 disabled:text-muted-foreground"
                >
                  {/* The ordinal, not the row's position: it is the number
                      the answer's [n] uses, and it can skip. */}
                  <span className="flex h-5 min-w-5 shrink-0 items-center justify-center rounded bg-primary/15 px-1 font-mono text-[10px] font-semibold text-primary">
                    {citation.ordinal}
                  </span>
                  <Icon className="size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
                  <span className="min-w-0 flex-1 truncate">{name}</span>
                  {citation.page_number != null && (
                    <span className="shrink-0 text-muted-foreground">
                      {t.viewer.page} {citation.page_number}
                    </span>
                  )}
                </button>
                {quote && (
                  <button
                    type="button"
                    onClick={() => setExpanded(isExpanded ? null : citation.ordinal)}
                    aria-expanded={isExpanded}
                    aria-label={t.excerpt}
                    title={t.excerpt}
                    className="shrink-0 rounded-r-lg p-2 text-muted-foreground transition-colors hover:bg-muted/60 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
                  >
                    <ChevronDown
                      className={cn('size-3.5 transition-transform', isExpanded && 'rotate-180')}
                      aria-hidden="true"
                    />
                  </button>
                )}
              </div>
              {isExpanded && quote && (
                // A text node, whatever the OCR left in it: Markdown and
                // table tags show as characters.
                <p className="border-t px-3 py-2 text-xs leading-relaxed text-muted-foreground whitespace-pre-wrap break-words">
                  {quote}
                  {isCut && '…'}
                </p>
              )}
            </li>
          )
        })}
      </ol>
    </div>
  )
}
