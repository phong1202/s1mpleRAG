'use client'

import { useState } from 'react'
import type { Locale, Message } from '@/lib/types'

export type ChatStatus = 'ready' | 'submitted' | 'streaming'

// UI-only until the backend has a chat endpoint: it records the question and
// reports chat as unavailable. Wiring the endpoint means rewriting send() and
// stop() here -- ChatPanel depends only on this hook's shape.
export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [status, setStatus] = useState<ChatStatus>('ready')
  const [error, setError] = useState<string | null>(null)

  function send(text: string, _locale: Locale) {
    setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: 'user', text }])
    setError('chat_unavailable')
  }

  function stop() {
    setStatus('ready')
  }

  function reset() {
    setMessages([])
    setError(null)
  }

  return { messages, status, error, send, stop, reset }
}
