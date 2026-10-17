// What the mock routes share: the switch, the envelope, and the
// conversations, kept in this server's memory the way the backend keeps them
// in Postgres. They last until the server restarts.

export const MOCK_ENABLED = process.env.QUERY_MOCK === '1'

export const MOCK_USER = {
  id: '00000000-0000-4000-8000-000000000001',
  username: 'admin',
  display_name: 'Admin',
  is_admin: true,
}

export function envelope(status: number, message: string, data: unknown, requestId = crypto.randomUUID()) {
  return Response.json({ code: status, message, data }, { status, headers: { 'X-Request-ID': requestId } })
}

export type MockMessage = {
  id: number
  role: 'user' | 'assistant'
  text: string
  citations: unknown[] | null
  created_at: string
}

export type MockConversation = {
  id: string
  title: string | null
  created_at: string
  updated_at: string
  messages: MockMessage[]
}

// On globalThis: each route is bundled on its own, and a module-level map
// would give each of them a store of its own.
type Store = { conversations: Map<string, MockConversation>; nextMessageId: number }
const scope = globalThis as typeof globalThis & { __s1mpleragMock?: Store }
const store = (scope.__s1mpleragMock ??= { conversations: new Map(), nextMessageId: 1 })

export function createConversation(): MockConversation {
  const now = new Date().toISOString()
  const conversation = { id: crypto.randomUUID(), title: null, created_at: now, updated_at: now, messages: [] }
  store.conversations.set(conversation.id, conversation)
  return conversation
}

export function getConversation(id: string) {
  return store.conversations.get(id)
}

export function deleteConversation(id: string) {
  return store.conversations.delete(id)
}

// Most recently updated first, as GET /conversations orders them.
export function listConversations() {
  return [...store.conversations.values()].sort((a, b) => b.updated_at.localeCompare(a.updated_at))
}

export function addExchange(conversation: MockConversation, question: string, answer: string, citations: unknown[]) {
  const now = new Date().toISOString()
  conversation.title ??= question.slice(0, 120)
  conversation.updated_at = now
  conversation.messages.push(
    { id: store.nextMessageId++, role: 'user', text: question, citations: null, created_at: now },
    { id: store.nextMessageId++, role: 'assistant', text: answer, citations, created_at: now },
  )
}
