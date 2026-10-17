// Stand-in for GET /conversations. See ../shared.ts.
import { envelope, listConversations, MOCK_ENABLED } from '../shared'

export function GET(request: Request) {
  if (!MOCK_ENABLED) return envelope(404, 'Not Found', null)
  const params = new URL(request.url).searchParams
  const limit = Number(params.get('limit') ?? 20)
  const offset = Number(params.get('offset') ?? 0)
  if (!Number.isInteger(limit) || limit < 1 || limit > 100 || !Number.isInteger(offset) || offset < 0) {
    return envelope(422, 'Validation failed', { errors: [{ field: 'limit', message: 'Out of range' }] })
  }
  const all = listConversations()
  const items = all.slice(offset, offset + limit).map(({ messages, ...summary }) => ({
    ...summary,
    message_count: messages.length,
  }))
  return envelope(200, 'Success', { items, total: all.length, limit, offset })
}
