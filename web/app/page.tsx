import { AppHeader } from '@/components/app-header'
import { ChatPanel } from '@/components/chat-panel'
import { DocumentPanel } from '@/components/document-panel'
import { LocaleProvider } from '@/components/locale-provider'

export default function Page() {
  return (
    <LocaleProvider>
      <div className="flex h-dvh flex-col">
        <AppHeader />
        <main className="flex min-h-0 flex-1 flex-col md:flex-row">
          <div className="max-h-[45dvh] shrink-0 md:max-h-none md:w-80 lg:w-96">
            <DocumentPanel />
          </div>
          <div className="min-h-0 flex-1">
            <ChatPanel />
          </div>
        </main>
      </div>
    </LocaleProvider>
  )
}
