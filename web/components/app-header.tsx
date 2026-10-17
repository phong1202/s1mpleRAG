'use client'

import { Library, UserRound } from 'lucide-react'
import { useMe } from '@/hooks/use-me'
import { LanguageToggle } from './language-toggle'
import { useLocale } from './locale-provider'

export function AppHeader() {
  const { t } = useLocale()
  const me = useMe()
  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b px-5">
      <div className="flex items-center gap-3">
        <div className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
          <Library className="size-4" aria-hidden="true" />
        </div>
        <div className="flex flex-col leading-tight">
          <h1 className="text-sm font-semibold">{t.appName}</h1>
          <p className="hidden text-xs text-muted-foreground sm:block">{t.tagline}</p>
        </div>
      </div>
      <div className="flex items-center gap-3">
        {/* Who the backend takes this to be; nothing while it does not say. */}
        {me && (
          <p className="flex items-center gap-1.5 text-xs text-muted-foreground" title={me.username}>
            <UserRound className="size-4" aria-hidden="true" />
            <span className="sr-only">{t.signedInAs} </span>
            <span className="hidden max-w-40 truncate sm:inline">{me.display_name}</span>
          </p>
        )}
        <LanguageToggle />
      </div>
    </header>
  )
}
