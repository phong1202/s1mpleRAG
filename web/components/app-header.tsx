'use client'

import { Library } from 'lucide-react'
import { LanguageToggle } from './language-toggle'
import { useLocale } from './locale-provider'

export function AppHeader() {
  const { t } = useLocale()
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
      <LanguageToggle />
    </header>
  )
}
