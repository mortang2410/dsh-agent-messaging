/**
 * Peer titles come from durable projection checkpoints, not from log folds.
 *
 * Discovery lists every session in the store. Folding a title per id means one
 * whole-log decode per session, so on a large store a single listing call costs
 * minutes; the cache's zero-I/O faces exist precisely to avoid that.
 *
 * The cache matches a record to the caller's lifecycle through the listed
 * header's identity fields (`version`, `createdAt`, `cwd`, `isSeeded`), so the
 * fakes here match on exactly those, as the real cache does: a witness carrying
 * any other field name makes every lookup miss and silently degrades every
 * peer name to a directory slug.
 */
import { describe, expect, it, vi } from 'vitest'

import { SessionQueryPeerDirectory, type CachedTitleSource } from '../src/adapters/directory.ts'

/** A listed header, shaped as the store lists it. */
const header = (id: string) => ({
  version: 4,
  id,
  createdAt: 1,
  isSeeded: false,
  cwd: '/work',
})

const logger = { warn: () => {}, info: () => {}, error: () => {}, debug: () => {} }

/** Match one witness the way the real cache's lifecycle identity does. */
function witnessMatches(meta: { version: number; createdAt: number; cwd?: string; isSeeded?: boolean }): boolean {
  return meta.version === 4 && meta.createdAt === 1 && meta.cwd === '/work' && (meta.isSeeded ?? false) === false
}

function directory(options: {
  cache?: CachedTitleSource
  foldCalls?: { count: number }
  seen?: { version: unknown }[]
}) {
  const foldCalls = options.foldCalls ?? { count: 0 }
  const sessionQuery = {
    listSessions: async () => [{ header: header('s1'), revision: 'r1' }, { header: header('s2'), revision: 'r2' }],
    readTitleSnapshots: vi.fn(async (ids: readonly string[]) => {
      foldCalls.count += 1
      return ids.map((id) => ({
        status: 'fulfilled' as const,
        value: { session: header(id), title: { title: `folded ${id}` } },
      }))
    }),
  }
  return new SessionQueryPeerDirectory({
    sessionQuery: sessionQuery as never,
    agents: { get: () => undefined } as never,
    presence: { readPeers: async () => [] } as never,
    logger: logger as never,
    includeSubagents: false,
    ...(options.cache === undefined ? {} : { projectionCache: options.cache }),
  })
}

const caching = (titles: (meta: { id: string }) => string | undefined): CachedTitleSource => ({
  cachedSnapshot: (meta, keys) => {
    const title = keys?.includes('title') === true && witnessMatches(meta) ? titles(meta) : undefined
    return title === undefined ? undefined : { values: { title } }
  },
  cachedPredecessorTitle: (meta) => {
    const title = witnessMatches(meta) ? titles(meta) : undefined
    return title === undefined ? undefined : { values: { title } }
  },
})

describe('peer titles', () => {
  it('reads titles from the projection cache and never folds a log', async () => {
    const foldCalls = { count: 0 }
    const cache = caching((meta) => `cached ${meta.id}`)
    const peers = await directory({ cache, foldCalls }).list()
    expect(peers.map((peer) => peer.title)).toEqual(['cached s1', 'cached s2'])
    expect(foldCalls.count).toBe(0)
  })

  it('passes the listed header itself as the identity witness', async () => {
    const seen: { version: unknown }[] = []
    const cache: CachedTitleSource = {
      cachedSnapshot: (meta) => {
        seen.push(meta)
        return undefined
      },
      cachedPredecessorTitle: () => undefined,
    }
    await directory({ cache }).list()
    expect(seen).toEqual([header('s1'), header('s2')])
  })

  it('serves a stale-record title from the predecessor checkpoint', async () => {
    const cache: CachedTitleSource = {
      cachedSnapshot: () => undefined,
      cachedPredecessorTitle: (meta) => ({ values: { title: `predecessor ${meta.id}` } }),
    }
    const peers = await directory({ cache }).list()
    expect(peers.map((peer) => peer.title)).toEqual(['predecessor s1', 'predecessor s2'])
  })

  it('omits a title the cache cannot serve instead of folding the log', async () => {
    const foldCalls = { count: 0 }
    const peers = await directory({ cache: caching(() => undefined), foldCalls }).list()
    expect(peers.every((peer) => peer.title === undefined)).toBe(true)
    expect(peers.map((peer) => peer.name)).toEqual(['work-s1', 'work-s2'])
    expect(foldCalls.count).toBe(0)
  })

  it('folds through the corpus once when the host mounts no cache', async () => {
    const foldCalls = { count: 0 }
    const peers = await directory({ foldCalls }).list()
    expect(peers.map((peer) => peer.title)).toEqual(['folded s1', 'folded s2'])
    expect(foldCalls.count).toBe(1)
  })
})
