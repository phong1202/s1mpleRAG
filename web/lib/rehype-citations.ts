// Turns the [n] markers of an answer into <cite-ref> elements, which the chat
// renders as buttons that open the cited page (components/chat-message.tsx).
// A list, [1, 2], becomes one element per number.
//
// A rehype plugin, run after sanitizing, rather than a rewrite of the answer
// into Markdown links before parsing: working on the parsed tree is what
// lets a marker inside code be told from one in prose, and nothing here ever
// builds markup out of the model's text.

export const CITATION_TAG = 'cite-ref'

// The hast nodes this touches, typed locally: hast's own types are not a
// dependency of this app.
type Node = {
  type: string
  tagName?: string
  value?: string
  properties?: Record<string, unknown>
  children?: Node[]
}

// The same pattern the backend maps citations with (app/core/generator.py):
// [1], and [1, 2], which models write as readily as [1][2].
const MARKER = /\[(\d+(?:\s*,\s*\d+)*)\]/g

// Code is quoted, not cited; and a button cannot sit inside a link.
const SKIPPED = new Set(['code', 'pre', 'a'])

function split(text: string): Node[] | null {
  const nodes: Node[] = []
  let end = 0
  for (const match of text.matchAll(MARKER)) {
    if (match.index > end) nodes.push({ type: 'text', value: text.slice(end, match.index) })
    for (const ordinal of match[1].split(',')) {
      nodes.push({
        type: 'element',
        tagName: CITATION_TAG,
        properties: { ordinal: Number(ordinal) },
        // Kept as the fallback: a number with no citation behind it is
        // rendered as the [n] it stood for.
        children: [{ type: 'text', value: `[${ordinal.trim()}]` }],
      })
    }
    end = match.index + match[0].length
  }
  if (nodes.length === 0) return null
  if (end < text.length) nodes.push({ type: 'text', value: text.slice(end) })
  return nodes
}

function walk(node: Node) {
  if (!node.children) return
  node.children = node.children.flatMap((child) => {
    if (child.type === 'text' && child.value) return split(child.value) ?? [child]
    if (child.type === 'element' && !SKIPPED.has(child.tagName ?? '')) walk(child)
    return [child]
  })
}

export function rehypeCitations() {
  return walk
}
