// Stand-in for GET /me: the backend's seeded admin. See ../shared.ts.
import { envelope, MOCK_ENABLED, MOCK_USER } from '../shared'

export function GET() {
  if (!MOCK_ENABLED) return envelope(404, 'Not Found', null)
  return envelope(200, 'Success', MOCK_USER)
}
