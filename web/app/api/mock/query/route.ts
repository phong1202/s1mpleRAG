// A stand-in for the backend's POST /query, for building the chat without
// paying for it: every real question is an OpenAI embedding and a chat
// completion, and the backend's stub provider has no answer to give (it
// answers 500). With QUERY_MOCK=1, next.config.mjs sends /backend/query
// here; the client code under test is exactly the code that talks to the
// real backend. Without it this route does not exist.
//
// The question picks the case, by its first word:
//   /none   the corpus has no answer: 200 with no citations
//   /html   HTML and Markdown in the answer and in a quote
//   /500    the provider failed
//   /slow   20 seconds, to try the stop button
//   (else)  an answer citing two documents, ordinals 1 and 3, once as a
//           list ([1, 3]), plus a [7] that has no citation behind it

const API_URL = process.env.API_URL ?? 'http://localhost:8000'
const MAX_QUESTION_CHARS = 2000

type Doc = { id: string; filename: string; status: string; page_count?: number | null }

type Citation = {
  ordinal: number
  kind: 'document'
  document_id: string
  filename: string
  page_number: number
  quote: string
  url: null
  title: null
}

// Cited when the backend has no ready document to borrow: the UI then shows
// a source that is no longer in the knowledge base.
const ABSENT: Doc = {
  id: '5a0c1c4e-0000-4000-8000-000000000000',
  filename: 'nd-168-2024_xu-phat-vi-pham-giao-thong-duong-bo.pdf',
  status: 'COMPLETED',
  page_count: 60,
}

const QUOTES = {
  plain:
    'Phạt tiền từ 18.000.000 đồng đến 20.000.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây: không chấp hành hiệu lệnh của đèn tín hiệu giao thông; không chấp hành hiệu lệnh, hướng dẫn của người điều khiển giao thông hoặc người kiểm soát giao thông; đi ngược chiều của đường',
  points: 'Ngoài việc bị phạt tiền, người điều khiển xe thực hiện hành vi vi phạm còn bị trừ điểm giấy phép lái xe như sau: thực hiện hành vi quy định tại điểm b khoản 9 Điều này bị trừ 04 điểm.',
  markup:
    '## Điều 6\n\n<table><tr><th>Hành vi</th><th>Mức phạt</th></tr><tr><td>**Vượt đèn đỏ**</td><td>18 - 20 triệu</td></tr></table>\n\n<img src=x onerror="document.title=\'pwned-quote\'"><script>document.title="pwned-quote"</script>',
}

const ANSWERS = {
  cited:
    'Người điều khiển xe ô tô **không chấp hành hiệu lệnh của đèn tín hiệu giao thông** bị phạt tiền từ 18 đến 20 triệu đồng [1].\n\nNgoài ra:\n\n- Bị trừ **04 điểm** giấy phép lái xe [3].\n- Mức phạt với xe máy thấp hơn [7].\n\nCả hai áp dụng cùng lúc [1, 3].\n\nTrong mã, dấu `[1]` không phải là trích dẫn.',
  none: 'Các nguồn được cung cấp không trả lời được câu hỏi này.',
  markup:
    'Mức phạt nằm ở bảng dưới đây [1].\n\n<img src=x onerror="document.title=\'pwned-answer\'">\n\n<script>document.title="pwned-answer"</script>\n\n<b onclick="document.title=\'pwned-answer\'">in đậm bằng HTML</b>\n\n[bấm vào đây](javascript:document.title=\'pwned-answer\')\n\n![ảnh ngoài](https://example.com/pixel.png)\n\n| Hành vi | Mức phạt |\n| --- | --- |\n| Vượt đèn đỏ | 18 - 20 triệu [1] |',
}

function envelope(status: number, message: string, data: unknown, requestId: string) {
  return Response.json({ code: status, message, data }, { status, headers: { 'X-Request-ID': requestId } })
}

async function readyDocuments(): Promise<Doc[]> {
  try {
    const res = await fetch(`${API_URL}/documents?limit=100`, { cache: 'no-store' })
    const body = (await res.json()) as { data: { items: Doc[] } }
    return body.data.items.filter((doc) => doc.status === 'COMPLETED')
  } catch {
    return []
  }
}

function cite(ordinal: number, doc: Doc, page: number, quote: string): Citation {
  return {
    ordinal,
    kind: 'document',
    document_id: doc.id,
    filename: doc.filename,
    page_number: Math.min(page, doc.page_count ?? page),
    quote: quote.slice(0, 280),
    url: null,
    title: null,
  }
}

export async function POST(request: Request) {
  const requestId = crypto.randomUUID()
  if (process.env.QUERY_MOCK !== '1') return envelope(404, 'Not Found', null, requestId)

  const started = Date.now()
  const body = (await request.json().catch(() => null)) as { question?: unknown } | null
  const question = typeof body?.question === 'string' ? body.question.trim() : ''
  if (!question || question.length > MAX_QUESTION_CHARS) {
    const errors = [{ field: 'question', message: 'String should have between 1 and 2000 characters' }]
    return envelope(422, 'Validation failed', { errors }, requestId)
  }

  const [command] = question.toLowerCase().split(/\s/)
  await new Promise((resolve) => setTimeout(resolve, command === '/slow' ? 20_000 : 1200))
  if (command === '/500') return envelope(500, 'Internal server error', null, requestId)

  const docs = await readyDocuments()
  const first = docs[0] ?? ABSENT
  const second = docs[1] ?? first

  let answer = ANSWERS.cited
  // The second source is cited on its last page: the far end of the jump.
  let citations = [cite(1, first, 2, QUOTES.plain), cite(3, second, second.page_count ?? 41, QUOTES.points)]
  if (command === '/none') {
    answer = ANSWERS.none
    citations = []
  } else if (command === '/html') {
    answer = ANSWERS.markup
    citations = [cite(1, first, 1, QUOTES.markup)]
  }

  return envelope(200, 'Success', { answer, citations, latency_ms: Date.now() - started }, requestId)
}
