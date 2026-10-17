'use client'

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { askQuestion, BackendError, fetchConversation, type ErrorKey } from '@/lib/backend'
import { isConversationId, toMessage } from '@/lib/conversations'
import type { Message } from '@/lib/types'

// No 'streaming': POST /query answers in one piece. 'loading' is a stored
// conversation being read back.
export type ChatStatus = 'ready' | 'loading' | 'submitted'

export type ChatError = { key: ErrorKey | null; requestId: string | null }

type ChatContextValue = {
  messages: Message[]
  status: ChatStatus
  error: ChatError | null
  // The conversation on screen, null for a new chat not yet answered.
  conversationId: string | null
  // Bumped whenever the list of conversations may have changed.
  revision: number
  send: (text: string) => void
  retry: () => void
  stop: () => void
  newChat: () => void
  open: (id: string) => void
  // Called once a conversation was deleted.
  forget: (id: string) => void
}

const ChatContext = createContext<ChatContextValue | null>(null)

// The conversation's id lives in the URL, ?c=<id>: a reload opens it again,
// and Back returns to the one before. The backend keeps the turns; this
// keeps the id and shows the turns.
function writeUrl(id: string | null, mode: 'push' | 'replace') {
  const url = id ? `?c=${encodeURIComponent(id)}` : window.location.pathname
  if (mode === 'push') window.history.pushState(null, '', url)
  else window.history.replaceState(null, '', url)
}

function urlConversation() {
  return new URLSearchParams(window.location.search).get('c')
}

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const urlId = useSearchParams().get('c')
  const [messages, setMessages] = useState<Message[]>([])
  const [status, setStatus] = useState<ChatStatus>('ready')
  const [error, setError] = useState<ChatError | null>(null)
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [revision, setRevision] = useState(0)
  // The request in flight, if any. Also how a late answer finds out it was
  // stopped or superseded: it is no longer the current one.
  const inFlight = useRef<AbortController | null>(null)
  // The same id as conversationId, for the async code that must not read
  // a stale one from its closure.
  const conversation = useRef<string | null>(null)
  // What "Try again" repeats: the last question, or reading the
  // conversation back.
  const retryAction = useRef<(() => void) | null>(null)

  const adopt = useCallback((id: string | null) => {
    conversation.current = id
    setConversationId(id)
  }, [])

  const begin = useCallback(() => {
    inFlight.current?.abort()
    const controller = new AbortController()
    inFlight.current = controller
    setError(null)
    return controller
  }, [])

  const finish = useCallback((controller: AbortController) => {
    if (inFlight.current !== controller) return
    inFlight.current = null
    setStatus('ready')
  }, [])

  const cancel = useCallback(() => {
    inFlight.current?.abort()
    inFlight.current = null
  }, [])

  // An id the backend no longer has (its database was reset, say) is
  // dropped, and the same question asked again as the opening one. Nothing
  // to tell the user: they get their answer.
  const askOrStartOver = useCallback(async (question: string, signal: AbortSignal) => {
    const sent = conversation.current
    try {
      return await askQuestion(question, sent, signal)
    } catch (cause) {
      if (!sent || !(cause instanceof BackendError) || cause.key !== 'conversation_not_found') throw cause
      return askQuestion(question, null, signal)
    }
  }, [])

  const ask = useCallback(
    async (question: string) => {
      const controller = begin()
      retryAction.current = () => void ask(question)
      setStatus('submitted')

      try {
        const { answer, citations, conversation_id } = await askOrStartOver(question, controller.signal)
        if (inFlight.current !== controller) return
        adopt(conversation_id)
        // The first answer of a new chat names it. Replace, not push: Back
        // should not lead to the same chat without its id.
        if (urlConversation() !== conversation_id) writeUrl(conversation_id, 'replace')
        setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: 'assistant', text: answer, citations }])
        setRevision((n) => n + 1)
      } catch (cause) {
        if (inFlight.current !== controller) return
        // A failed question is not added to the conversation, so the id stays
        // and a retry continues it.
        const key = cause instanceof BackendError ? cause.key : null
        setError({
          // A 404 left over here came without an id, or after dropping it:
          // nothing the user can act on, so the generic failure.
          key: key === 'conversation_not_found' ? 'query_failed' : key,
          requestId: cause instanceof BackendError ? cause.requestId : null,
        })
      }
      finish(controller)
    },
    [adopt, askOrStartOver, begin, finish],
  )

  const load = useCallback(
    async (id: string) => {
      const controller = begin()
      retryAction.current = () => void load(id)
      adopt(id)
      setMessages([])
      setStatus('loading')

      try {
        if (!isConversationId(id)) throw new BackendError('conversation_not_found', 'Not a conversation id')
        const detail = await fetchConversation(id, controller.signal)
        if (inFlight.current !== controller) return
        setMessages(detail.messages.map(toMessage))
      } catch (cause) {
        if (inFlight.current !== controller) return
        const key = cause instanceof BackendError ? cause.key : null
        if (key === 'conversation_not_found') {
          // Deleted, or the database was reset: an empty new chat, quietly.
          adopt(null)
          writeUrl(null, 'replace')
        } else {
          setError({ key, requestId: cause instanceof BackendError ? cause.requestId : null })
        }
      }
      finish(controller)
    },
    [adopt, begin, finish],
  )

  const clear = useCallback(() => {
    cancel()
    retryAction.current = null
    adopt(null)
    setMessages([])
    setError(null)
    setStatus('ready')
  }, [adopt, cancel])

  // The URL leads: a reload, a link, Back and Forward all end up here.
  useEffect(() => {
    if (urlId === conversation.current) return
    if (urlId) void load(urlId)
    else clear()
  }, [urlId, load, clear])

  // Leaving drops whatever is in flight, and forgets which conversation was
  // being shown: mounted again (React mounts twice in development), the
  // effect above must read it again rather than take it as on screen while
  // its read was just cancelled.
  useEffect(
    () => () => {
      cancel()
      conversation.current = null
    },
    [cancel],
  )

  const send = useCallback(
    (text: string) => {
      setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: 'user', text }])
      void ask(text)
    },
    [ask],
  )

  const retry = useCallback(() => retryAction.current?.(), [])

  // Stops waiting. The backend may still finish the work it started, and
  // then keeps the exchange in the conversation: there is no cancel on its
  // side, only on ours (issue #2).
  const stop = useCallback(() => {
    cancel()
    setStatus('ready')
  }, [cancel])

  // A new chat is a new conversation: the next question goes without an id.
  const newChat = useCallback(() => {
    clear()
    if (urlConversation()) writeUrl(null, 'push')
  }, [clear])

  const open = useCallback((id: string) => {
    if (id !== urlConversation()) writeUrl(id, 'push')
  }, [])

  const forget = useCallback(
    (id: string) => {
      setRevision((n) => n + 1)
      if (conversation.current !== id) return
      clear()
      writeUrl(null, 'replace')
    },
    [clear],
  )

  const value = useMemo(
    () => ({ messages, status, error, conversationId, revision, send, retry, stop, newChat, open, forget }),
    [messages, status, error, conversationId, revision, send, retry, stop, newChat, open, forget],
  )
  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>
}

export function useChat() {
  const context = useContext(ChatContext)
  if (!context) throw new Error('useChat must be used within ChatProvider')
  return context
}
