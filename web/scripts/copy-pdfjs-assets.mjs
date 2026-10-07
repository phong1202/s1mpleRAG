// Copies the pdf.js data files the viewer fetches at runtime into public/,
// so they are served from this app's own origin rather than a CDN. They are
// build output (gitignored), regenerated from the installed pdfjs-dist so
// they can never drift from the worker's version.
import { cpSync, mkdirSync, rmSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'

const require = createRequire(import.meta.url)
const source = dirname(require.resolve('pdfjs-dist/package.json'))
const target = join(import.meta.dirname, '..', 'public', 'pdfjs')

rmSync(target, { recursive: true, force: true })
mkdirSync(target, { recursive: true })

// cmaps: CJK and other non-Latin encodings. standard_fonts: the 14 base
// fonts that PDFs reference without embedding (the text-layer legal PDFs
// here use Helvetica/Times). wasm: JBIG2 and JPEG 2000 decoders for scans.
// Deliberately NOT copied: quickjs-eval.*, the PDF scripting sandbox, which
// this viewer never enables.
cpSync(join(source, 'cmaps'), join(target, 'cmaps'), { recursive: true })
cpSync(join(source, 'standard_fonts'), join(target, 'standard_fonts'), { recursive: true })
for (const name of ['jbig2.wasm', 'jbig2_nowasm_fallback.js', 'openjpeg.wasm', 'openjpeg_nowasm_fallback.js']) {
  cpSync(join(source, 'wasm', name), join(target, 'wasm', name))
}
