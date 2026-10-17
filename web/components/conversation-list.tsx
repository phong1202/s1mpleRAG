'use client'

import { useState } from 'react'
import { Loader2, MessageSquare, SquarePen, Trash2 } from 'lucide-react'
import { useConversations } from '@/hooks/use-conversations'
import { BackendError, deleteConversation } from '@/lib/backend'
import { formatUpdated } from '@/lib/conversations'
import type { ConversationSummary } from '@/lib/types'
import { cn } from '@/lib/utils'
import { useChat } from './chat-provider'
import { ConfirmDialog } from './confirm-dialog'
import { useLocale } from './locale-provider'
import { useToast } from './toast-provider'

export function ConversationList() {
  const { t, locale } = useLocale()
  const { notify } = useToast()
  const { conversationId, revision, status, open, forget, newChat } = useChat()
  const { conversations, error, isLoading, hasMore, loadMore, mutate } = useConversations(revision)
  const [pendingDelete, setPendingDelete] = useState<ConversationSummary | null>(null)

  const titleOf = (c: ConversationSummary) => c.title?.trim() || t.untitled

  async function remove(conversation: ConversationSummary) {
    try {
      await deleteConversation(conversation.id)
    } catch (cause) {
      // Already gone is what was asked for.
      if (!(cause instanceof BackendError && cause.key === 'conversation_not_found')) {
        notify(`${titleOf(conversation)}: ${t.deleteConversationFailed}`)
        return
      }
    }
    forget(conversation.id)
    void mutate()
  }

  return (
    <div className="flex h-full flex-col gap-3 overflow-hidden p-5">
      {/* Above the list it starts a new entry of: the conversation the next
          question opens. */}
      <button
        type="button"
        onClick={newChat}
        disabled={status === 'submitted'}
        className="flex shrink-0 items-center justify-center gap-2 rounded-lg bg-primary px-3 py-2 text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40 focus-visible:ring-offset-2 disabled:opacity-50"
      >
        <SquarePen className="size-4" aria-hidden="true" />
        {t.newChat}
      </button>
      <p className="text-xs text-muted-foreground">{t.conversationsHint}</p>

      {isLoading ? (
        <p className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" aria-hidden="true" />
          {t.loadingConversations}
        </p>
      ) : error && conversations.length === 0 ? (
        <p role="alert" className="text-sm text-destructive">
          {(error instanceof BackendError && error.key && t.errors[error.key]) || t.errors.conversation_unavailable}
        </p>
      ) : conversations.length === 0 ? (
        <p className="text-sm text-muted-foreground">{t.noConversations}</p>
      ) : (
        <ul className="flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto" aria-label={t.conversations}>
          {conversations.map((conversation) => {
            const active = conversation.id === conversationId
            return (
              <li
                key={conversation.id}
                className={cn(
                  'group flex items-center gap-1 rounded-lg border transition-colors',
                  active ? 'border-primary/40 bg-primary/5' : 'border-transparent hover:bg-muted/60',
                )}
              >
                <button
                  type="button"
                  onClick={() => open(conversation.id)}
                  aria-current={active ? 'true' : undefined}
                  className="flex min-w-0 flex-1 items-start gap-2.5 rounded-lg p-2 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
                >
                  <MessageSquare
                    className={cn('mt-0.5 size-4 shrink-0', active ? 'text-primary' : 'text-muted-foreground')}
                    aria-hidden="true"
                  />
                  <span className="min-w-0 flex-1">
                    <span className="line-clamp-2 text-sm break-words">{titleOf(conversation)}</span>
                    <span className="text-xs text-muted-foreground">
                      {formatUpdated(conversation.updated_at, locale)}
                    </span>
                  </span>
                </button>
                <button
                  type="button"
                  onClick={() => setPendingDelete(conversation)}
                  className="mr-1 rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
                  aria-label={`${t.remove} ${titleOf(conversation)}`}
                >
                  <Trash2 className="size-4" aria-hidden="true" />
                </button>
              </li>
            )
          })}
          {hasMore && (
            <li>
              <button
                type="button"
                onClick={loadMore}
                className="w-full rounded-lg p-2 text-xs text-muted-foreground transition-colors hover:bg-muted/60 hover:text-foreground"
              >
                {t.loadMore}
              </button>
            </li>
          )}
        </ul>
      )}

      <ConfirmDialog
        open={pendingDelete !== null}
        title={t.deleteConversationTitle}
        description={
          <>
            <span className="mb-1 block break-words font-medium text-foreground">
              {pendingDelete && titleOf(pendingDelete)}
            </span>
            {t.deleteConversationBody}
          </>
        }
        cancelLabel={t.cancel}
        confirmLabel={t.confirmDelete}
        onCancel={() => setPendingDelete(null)}
        onConfirm={() => {
          if (pendingDelete) void remove(pendingDelete)
          setPendingDelete(null)
        }}
      />
    </div>
  )
}
