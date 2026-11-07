/**
 * Peer titles come from durable projection checkpoints, not from log folds.
 *
 * Discovery lists every session in the store. Folding a title per id means one
 * whole-log decode per session, so on a large store a single listing call costs
 * minutes; the cache's zero-I/O faces exist precisely to avoid that.
 */
import { describe, expect, it, vi } from 'vitest'

import { SessionQueryPeerDirectory, type CachedTitleSource } from '../src/adapters/directory.ts'

const header = (id: string) => ({
  id,
  version: 4,
  createdAt: 1,
  isSeeded: false,
  formatVersion: 4,
  cwd: '/work',
})

const logger = { warn: () => {}, info: () => {}, error: () => {}, debug: () => {} }

function directory(options: {
  titles?: Readonly<Record<string, string | null>>
  cache?: CachedTitleSource
  foldCalls?: { count: number }
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

describe('peer titles', () => {
  it('reads titles from the projection cache and never folds a log', async () => {
    const foldCalls = { count: 0 }
    const cache: CachedTitleSource = {
      cachedSnapshot: (meta, keys) =>
        keys?.includes('title') === true ? { values: { title: `cached ${meta.id}` } } : { values: {} },
      cachedPredecessorTitle: () => undefined,
    }
    const peers = await directory({ cache, foldCalls }).list()
    expect(peers.map((peer) => peer.title)).toEqual(['cached s1', 'cached s2'])
    expect(foldCalls.count).toBe(0)
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
    const cache: CachedTitleSource = {
      cachedSnapshot: () => undefined,
      cachedPredecessorTitle: () => undefined,
    }
    const peers = await directory({ cache, foldCalls }).list()
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
