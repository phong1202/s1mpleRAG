'use client'

import { useState } from 'react'
import { ChevronDown, FileText } from 'lucide-react'
import { Streamdown } from 'streamdown'
import { cn } from '@/lib/utils'
import type { Message, SourceReference } from '@/lib/types'
import { useLocale } from './locale-provider'

export function ChatMessage({ message, isStreaming }: { message: Message; isStreaming: boolean }) {
  const { t } = useLocale()
  const isUser = message.role === 'user'
  const { text, sources = [] } = message

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-sm text-primary-foreground whitespace-pre-wrap">
          <span className="sr-only">{t.you}: </span>
          {text}
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3">
      <span className="sr-only">{t.assistant}:</span>
      {text ? (
        <div className="prose-chat text-sm leading-relaxed">
          <Streamdown isAnimating={isStreaming}>{text}</Streamdown>
        </div>
      ) : (
        <p className="flex items-center gap-2 text-sm text-muted-foreground">
          <span className="size-2 animate-pulse rounded-full bg-primary" aria-hidden="true" />
          {t.thinking}
        </p>
      )}
      {sources.length > 0 && <SourceList sources={sources} />}
    </div>
  )
}

function SourceList({ sources }: { sources: SourceReference[] }) {
  const { t } = useLocale()
  const [openIndex, setOpenIndex] = useState<number | null>(null)

  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{t.sources}</p>
      <ol className="flex flex-col gap-1.5">
        {sources.map((source, i) => {
          const open = openIndex === i
          return (
            <li key={`${source.documentId}-${source.index}`} className="rounded-lg border bg-card/40">
              <button
                type="button"
                onClick={() => setOpenIndex(open ? null : i)}
                aria-expanded={open}
                className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs"
              >
                <span className="flex size-5 shrink-0 items-center justify-center rounded bg-primary/15 font-mono text-[10px] font-semibold text-primary">
                  {i + 1}
                </span>
                <FileText className="size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
                <span className="min-w-0 flex-1 truncate">{source.documentName}</span>
                <span className="shrink-0 text-muted-foreground">
                  {t.chunk} {source.index + 1} · {Math.round(source.score * 100)}%
                </span>
                <ChevronDown
                  className={cn('size-3.5 shrink-0 text-muted-foreground transition-transform', open && 'rotate-180')}
                  aria-hidden="true"
                />
              </button>
              {open && (
                <p className="border-t px-3 py-2 text-xs leading-relaxed text-muted-foreground whitespace-pre-wrap">
                  {source.text}
                </p>
              )}
            </li>
          )
        })}
      </ol>
    </div>
  )
}
