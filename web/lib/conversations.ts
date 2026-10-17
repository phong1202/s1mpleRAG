import type { Locale, Message, StoredMessage } from '@/lib/types'

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

// The id comes from the URL, which anyone can edit: anything that is not a
// UUID is treated as a conversation that does not exist, without asking.
export function isConversationId(value: string) {
  return UUID.test(value)
}

// A stored turn as the chat renders it. Answers stored before the backend
// kept citations come back with null: shown as plain text.
export function toMessage(stored: StoredMessage): Message {
  return {
    id: `stored-${stored.id}`,
    role: stored.role,
    text: stored.text,
    citations: stored.citations ?? undefined,
  }
}

const MINUTE = 60_000
const HOUR = 60 * MINUTE
const DAY = 24 * HOUR

// "5 minutes ago" for the recent past, a date beyond a week: past that, how
// long ago matters less than which day.
export function formatUpdated(iso: string, locale: Locale, now = Date.now()) {
  const elapsed = now - new Date(iso).getTime()
  const relative = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' })
  if (elapsed < MINUTE) return relative.format(0, 'second')
  if (elapsed < HOUR) return relative.format(-Math.floor(elapsed / MINUTE), 'minute')
  if (elapsed < DAY) return relative.format(-Math.floor(elapsed / HOUR), 'hour')
  if (elapsed < 7 * DAY) return relative.format(-Math.floor(elapsed / DAY), 'day')
  return new Intl.DateTimeFormat(locale, { day: 'numeric', month: 'short', year: 'numeric' }).format(new Date(iso))
}
