'use client'

import { useEffect } from 'react'
import useSWRInfinite from 'swr/infinite'
import { conversationsKey, fetchConversations, type Paginated } from '@/lib/backend'
import type { ConversationSummary } from '@/lib/types'

// Page by page, as GET /conversations serves them. `revision` comes from
// the chat: each answer can add a conversation or move one to the top, so
// every loaded page is read again.
export function useConversations(revision: number) {
  const { data, error, isLoading, size, setSize, mutate } = useSWRInfinite(
    (page: number, previous: Paginated<ConversationSummary> | null) =>
      previous && previous.offset + previous.items.length >= previous.total ? null : conversationsKey(page),
    fetchConversations,
  )

  useEffect(() => {
    if (revision > 0) void mutate()
  }, [revision, mutate])

  // An answer moves its conversation to the top, so pages read at different
  // times can both hold it: the first sighting wins.
  const seen = new Set<string>()
  const conversations = (data ?? [])
    .flatMap((page) => page.items)
    .filter((c) => !seen.has(c.id) && Boolean(seen.add(c.id)))
  const total = data?.[0]?.total ?? 0

  return {
    conversations,
    error,
    isLoading,
    hasMore: conversations.length < total,
    loadMore: () => setSize(size + 1),
    mutate,
  }
}
