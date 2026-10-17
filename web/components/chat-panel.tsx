'use client'

import { useEffect, useRef, useState } from 'react'
import { ArrowUp, Loader2, MessageSquareText, RotateCcw, Square } from 'lucide-react'
import { useDocuments } from '@/hooks/use-documents'
import { MAX_QUESTION_CHARS } from '@/lib/backend'
import { cn } from '@/lib/utils'
import { ChatMessage } from './chat-message'
import { useChat } from './chat-provider'
import { useLocale } from './locale-provider'

export function ChatPanel() {
  const { t } = useLocale()
  const { documents } = useDocuments()
  const [input, setInput] = useState('')
  const { messages, send, retry, status, stop, error } = useChat()
  const scrollRef = useRef<HTMLDivElement>(null)

  const busy = status === 'submitted'
  const loading = status === 'loading'
  const hasDocs = documents.length > 0
  const question = input.trim()
  const tooLong = question.length > MAX_QUESTION_CHARS

  // The waiting line and the error sit below the last message too.
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, status, error])

  function submit() {
    if (!question || tooLong || busy || loading) return
    send(question)
    setInput('')
  }

  return (
    <section className="flex h-full min-h-0 flex-col" aria-label="Chat">
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-5 py-8">
          {loading ? (
            <p className="flex items-center gap-2 text-sm text-muted-foreground" aria-live="polite">
              <Loader2 className="size-4 animate-spin" aria-hidden="true" />
              {t.loadingConversation}
            </p>
          ) : messages.length === 0 ? (
            error ? null : <EmptyState hasDocs={hasDocs} />
          ) : (
            messages.map((message) => <ChatMessage key={message.id} message={message} />)
          )}
          {busy && (
            <p className="flex items-center gap-2 text-sm text-muted-foreground" aria-live="polite">
              <span className="size-2 animate-pulse rounded-full bg-primary" aria-hidden="true" />
              {t.thinking}
            </p>
          )}
          {error && (
            <div
              role="alert"
              className="flex flex-col gap-1.5 rounded-lg bg-destructive/10 px-3 py-2 text-sm text-destructive"
            >
              <p>{(error.key && t.errors[error.key]) ?? t.errorGeneric}</p>
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
                <button
                  type="button"
                  onClick={retry}
                  className="flex items-center gap-1.5 font-medium underline-offset-2 hover:underline"
                >
                  <RotateCcw className="size-3" aria-hidden="true" />
                  {t.retry}
                </button>
                {/* What to quote when reporting it: the id the backend
                    logged this failure under. */}
                {error.requestId && (
                  <span className="font-mono text-destructive/70 select-all">
                    {t.requestId}: {error.requestId}
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="border-t bg-background/80 backdrop-blur">
        <form
          className="mx-auto flex w-full max-w-3xl flex-col gap-2 px-5 py-4"
          onSubmit={(e) => {
            e.preventDefault()
            submit()
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
                  submit()
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
                disabled={!question || tooLong || !hasDocs || loading}
                className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground transition-opacity disabled:opacity-40"
                aria-label={t.send}
              >
                <ArrowUp className="size-4" aria-hidden="true" />
              </button>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
            {/* Only once the limit is near: the backend refuses a longer
                question, so the count shows before the send does. */}
            {question.length > MAX_QUESTION_CHARS * 0.9 && (
              <p className={cn('ml-auto font-mono tabular-nums', tooLong && 'font-semibold text-destructive')}>
                {question.length}/{MAX_QUESTION_CHARS}
              </p>
            )}
          </div>
        </form>
      </div>
    </section>
  )
}

function EmptyState({ hasDocs }: { hasDocs: boolean }) {
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
    </div>
  )
}
