import { connection } from 'next/server'
import { AppHeader } from '@/components/app-header'
import { LocaleProvider } from '@/components/locale-provider'
import { Workspace } from '@/components/workspace'

export default async function Page() {
  // Rendered per request: the CSP nonce set in proxy.ts only reaches
  // Next's scripts during a dynamic render, never a prerendered page.
  await connection()

  return (
    <LocaleProvider>
      <div className="flex h-dvh flex-col">
        <AppHeader />
        <Workspace />
      </div>
    </LocaleProvider>
  )
}
