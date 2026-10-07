'use client'

import { cn } from '@/lib/utils'
import type { Locale } from '@/lib/types'
import { useLocale } from './locale-provider'

const OPTIONS: { value: Locale; label: string; full: string }[] = [
  { value: 'en', label: 'EN', full: 'English' },
  { value: 'vi', label: 'VI', full: 'Tiếng Việt' },
]

export function LanguageToggle() {
  const { locale, setLocale, t } = useLocale()

  return (
    <div
      role="radiogroup"
      aria-label={t.language}
      className="flex items-center rounded-lg border bg-muted/40 p-0.5"
    >
      {OPTIONS.map((option) => {
        const active = option.value === locale
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={active}
            aria-label={option.full}
            onClick={() => setLocale(option.value)}
            className={cn(
              'rounded-md px-2.5 py-1 text-xs font-medium transition-colors',
              active
                ? 'bg-primary text-primary-foreground'
                : 'text-muted-foreground hover:text-foreground',
            )}
          >
            {option.label}
          </button>
        )
      })}
    </div>
  )
}
