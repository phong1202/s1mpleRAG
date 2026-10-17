// Stand-in for GET and DELETE /conversations/{id}. See ../../shared.ts.
import { deleteConversation, envelope, getConversation, MOCK_ENABLED } from '../../shared'

type Context = { params: Promise<{ id: string }> }

export async function GET(_request: Request, { params }: Context) {
  if (!MOCK_ENABLED) return envelope(404, 'Not Found', null)
  const conversation = getConversation((await params).id)
  if (!conversation) return envelope(404, 'Conversation not found', null)
  return envelope(200, 'Success', conversation)
}

export async function DELETE(_request: Request, { params }: Context) {
  if (!MOCK_ENABLED) return envelope(404, 'Not Found', null)
  if (!deleteConversation((await params).id)) return envelope(404, 'Conversation not found', null)
  return envelope(200, 'Success', null)
}
