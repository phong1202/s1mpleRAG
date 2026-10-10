'use client'

import { useEffect, useRef, useState } from 'react'
import { ArrowUp, MessageSquareText, RotateCcw, Square } from 'lucide-react'
import { useChat } from '@/hooks/use-chat'
import { useDocuments } from '@/hooks/use-documents'
import { ChatMessage } from './chat-message'
import { useLocale } from './locale-provider'

export function ChatPanel() {
  const { t, locale } = useLocale()
  const { documents } = useDocuments()
  const [input, setInput] = useState('')
  const { messages, send, status, stop, reset, error } = useChat()
  const scrollRef = useRef<HTMLDivElement>(null)

  const busy = status === 'submitted' || status === 'streaming'
  const hasDocs = documents.length > 0

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  function submit(text: string) {
    const trimmed = text.trim()
    if (!trimmed || busy) return
    send(trimmed, locale)
    setInput('')
  }

  const lastMessage = messages.at(-1)
  const awaitingFirstToken = status === 'submitted' && lastMessage?.role === 'user'

  return (
    <section className="flex h-full min-h-0 flex-col" aria-label="Chat">
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-5 py-8">
          {messages.length === 0 ? (
            <EmptyState hasDocs={hasDocs} onPick={submit} />
          ) : (
            messages.map((message) => (
              <ChatMessage
                key={message.id}
                message={message}
                isStreaming={status === 'streaming' && message.id === lastMessage?.id}
              />
            ))
          )}
          {awaitingFirstToken && (
            <p className="flex items-center gap-2 text-sm text-muted-foreground" aria-live="polite">
              <span className="size-2 animate-pulse rounded-full bg-primary" aria-hidden="true" />
              {t.thinking}
            </p>
          )}
          {error && (
            <p role="alert" className="rounded-lg bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {t.errors[error] ?? t.errorGeneric}
            </p>
          )}
        </div>
      </div>

      <div className="border-t bg-background/80 backdrop-blur">
        <form
          className="mx-auto flex w-full max-w-3xl flex-col gap-2 px-5 py-4"
          onSubmit={(e) => {
            e.preventDefault()
            submit(input)
          }}
        >
          <div className="flex items-end gap-2 rounded-2xl border bg-card p-2 focus-within:ring-2 focus-within:ring-ring/40">
            <label htmlFor="chat-input" className="sr-only">
              {t.placeholder}
            </label>
            <textarea
              id="chat-input"
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  if (e.nativeEvent.isComposing || e.keyCode === 229) return
                  e.preventDefault()
                  submit(input)
                }
              }}
              placeholder={t.placeholder}
              disabled={!hasDocs}
              className="max-h-40 min-h-10 flex-1 resize-none bg-transparent px-2 py-2 text-sm outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed field-sizing-content"
            />
            {busy ? (
              <button
                type="button"
                onClick={() => stop()}
                className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-secondary text-secondary-foreground"
                aria-label={t.stop}
              >
                <Square className="size-3.5 fill-current" aria-hidden="true" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim() || !hasDocs}
                className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground transition-opacity disabled:opacity-40"
                aria-label={t.send}
              >
                <ArrowUp className="size-4" aria-hidden="true" />
              </button>
            )}
          </div>
          {messages.length > 0 && (
            <button
              type="button"
              onClick={reset}
              disabled={busy}
              className="flex items-center gap-1.5 self-start text-xs text-muted-foreground transition-colors hover:text-foreground disabled:opacity-50"
            >
              <RotateCcw className="size-3" aria-hidden="true" />
              {t.newChat}
            </button>
          )}
        </form>
      </div>
    </section>
  )
}

function EmptyState({ hasDocs, onPick }: { hasDocs: boolean; onPick: (text: string) => void }) {
  const { t } = useLocale()
  return (
    <div className="flex flex-col items-center gap-6 py-16 text-center">
      <div className="flex size-12 items-center justify-center rounded-2xl bg-primary/10 text-primary">
        <MessageSquareText className="size-5" aria-hidden="true" />
      </div>
      <div className="flex max-w-md flex-col gap-2">
        <h2 className="text-2xl font-semibold tracking-tight text-balance">{t.emptyTitle}</h2>
        <p className="text-sm leading-relaxed text-muted-foreground text-pretty">
          {hasDocs ? t.emptyBody : t.emptyNoDocs}
        </p>
      </div>
      {hasDocs && (
        <div className="flex flex-wrap justify-center gap-2">
          {t.suggestions.map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              onClick={() => onPick(suggestion)}
              className="rounded-full border bg-card/60 px-3.5 py-1.5 text-xs transition-colors hover:border-primary/50 hover:text-primary"
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
