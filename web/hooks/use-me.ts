'use client'

import useSWR from 'swr'
import { fetchMe, ME_KEY } from '@/lib/backend'

// Who the backend says this is: its seeded admin, until login exists.
export function useMe() {
  const { data } = useSWR(ME_KEY, fetchMe, { revalidateOnFocus: false, shouldRetryOnError: false })
  return data ?? null
}
