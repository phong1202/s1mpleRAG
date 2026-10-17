'use client'

import { useRef, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { cn } from '@/lib/utils'
import { ConversationList } from './conversation-list'
import { DocumentPanel } from './document-panel'
import { useLocale } from './locale-provider'

type Tab = 'documents' | 'conversations'
const TABS: Tab[] = ['documents', 'conversations']

// Two tabs in the one column there is room for. Both panels stay mounted:
// an upload in progress lives in the documents panel's state, and switching
// tabs must not lose it.
export function Sidebar() {
  const { t } = useLocale()
  // Opened on a conversation (a reload, a link), the list it belongs to and
  // its New chat button come first.
  const opensConversation = useSearchParams().has('c')
  const [tab, setTab] = useState<Tab>(opensConversation ? 'conversations' : 'documents')
  const buttons = useRef<Record<Tab, HTMLButtonElement | null>>({ documents: null, conversations: null })
  const label: Record<Tab, string> = { documents: t.knowledgeBase, conversations: t.conversations }

  // The tablist pattern: arrows move between tabs, and only the selected one
  // is in the tab order.
  function onKeyDown(e: React.KeyboardEvent) {
    const step = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0
    if (!step) return
    e.preventDefault()
    const next = TABS[(TABS.indexOf(tab) + step + TABS.length) % TABS.length]
    setTab(next)
    buttons.current[next]?.focus()
  }

  return (
    <aside className="flex h-full flex-col overflow-hidden border-b bg-card/40 md:border-r md:border-b-0">
      <div role="tablist" aria-orientation="horizontal" onKeyDown={onKeyDown} className="flex shrink-0 gap-1 border-b px-3">
        {TABS.map((id) => (
          <button
            key={id}
            ref={(el) => {
              buttons.current[id] = el
            }}
            type="button"
            role="tab"
            id={`tab-${id}`}
            aria-selected={tab === id}
            aria-controls={`panel-${id}`}
            tabIndex={tab === id ? 0 : -1}
            onClick={() => setTab(id)}
            className={cn(
              '-mb-px border-b-2 px-3 py-3 text-sm font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40',
              tab === id
                ? 'border-primary text-foreground'
                : 'border-transparent text-muted-foreground hover:text-foreground',
            )}
          >
            {label[id]}
          </button>
        ))}
      </div>
      {TABS.map((id) => (
        <div
          key={id}
          role="tabpanel"
          id={`panel-${id}`}
          aria-labelledby={`tab-${id}`}
          hidden={tab !== id}
          className="min-h-0 flex-1"
        >
          {id === 'documents' ? <DocumentPanel /> : <ConversationList />}
        </div>
      ))}
    </aside>
  )
}
