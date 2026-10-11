'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { askQuestion, BackendError, type ErrorKey } from '@/lib/backend'
import type { Message } from '@/lib/types'

// No 'streaming': POST /query answers in one piece, so a question is either
// in flight or it is not.
export type ChatStatus = 'ready' | 'submitted'

export type ChatError = { key: ErrorKey | null; requestId: string | null }

// The conversation lives here and nowhere else: the backend answers each
// question on its own, so nothing but the question is sent, and the history
// is only what the screen shows.
export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [status, setStatus] = useState<ChatStatus>('ready')
  const [error, setError] = useState<ChatError | null>(null)
  // The request in flight, if any. Also how a late answer finds out it was
  // stopped or superseded: it is no longer the current one.
  const inFlight = useRef<AbortController | null>(null)
  const lastQuestion = useRef('')

  const ask = useCallback(async (question: string) => {
    inFlight.current?.abort()
    const controller = new AbortController()
    inFlight.current = controller
    lastQuestion.current = question
    setError(null)
    setStatus('submitted')

    try {
      const { answer, citations } = await askQuestion(question, controller.signal)
      if (inFlight.current !== controller) return
      setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: 'assistant', text: answer, citations }])
    } catch (cause) {
      if (inFlight.current !== controller) return
      setError({
        key: cause instanceof BackendError ? cause.key : null,
        requestId: cause instanceof BackendError ? cause.requestId : null,
      })
    }
    inFlight.current = null
    setStatus('ready')
  }, [])

  const cancel = useCallback(() => {
    inFlight.current?.abort()
    inFlight.current = null
  }, [])

  const send = useCallback(
    (text: string) => {
      setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: 'user', text }])
      void ask(text)
    },
    [ask],
  )

  // Asks the failed question again without repeating it in the conversation.
  const retry = useCallback(() => {
    if (lastQuestion.current) void ask(lastQuestion.current)
  }, [ask])

  // Stops waiting. The backend may still finish the work it started: there
  // is no cancel on its side, only on ours.
  const stop = useCallback(() => {
    cancel()
    setStatus('ready')
  }, [cancel])

  const reset = useCallback(() => {
    cancel()
    lastQuestion.current = ''
    setMessages([])
    setError(null)
    setStatus('ready')
  }, [cancel])

  useEffect(() => cancel, [cancel])

  return { messages, status, error, send, retry, stop, reset }
}
